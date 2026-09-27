import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置
# ============================================================
MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"
DATA_PATH = "../dataset/wikitext-02/train.jsonl"

MAX_ARTICLES = 9007
MAX_TOKENS_PER_ART = 256
DEVICE = "cuda"
SEED = 42

ATTN_BATCH = 16
RIDGE_BATCH = 100000

torch.manual_seed(SEED)


# ============================================================
# 数据
# ============================================================
df = pd.read_json(DATA_PATH, lines=True)
texts = df["text"].iloc[:MAX_ARTICLES].tolist()
tok = AutoTokenizer.from_pretrained(MODEL_A)

article_ids = []
for t in texts:
    ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    ids = ids[:MAX_TOKENS_PER_ART]
    if len(ids) >= 16:
        article_ids.append(ids)

max_len = max(len(x) for x in article_ids)
padded = torch.zeros(len(article_ids), max_len, dtype=torch.long)
for i, ids in enumerate(article_ids):
    padded[i, :len(ids)] = ids

print(f"文章数: {len(article_ids)}, 每篇最多 {max_len} token")


# ============================================================
# 工具
# ============================================================
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x, cos, sin):
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)


def cos_sim_float64(A, B):
    A = A.double().reshape(-1)
    B = B.double().reshape(-1)
    dot = (A * B).sum()
    nA = (A ** 2).sum().sqrt()
    nB = (B ** 2).sum().sqrt()
    return (dot / (nA * nB + 1e-12)).item()


# ============================================================
# 提取单层 Q/K/V
# ============================================================
@torch.no_grad()
def extract_qkv(model, layer_idx, padded, need_q=True):
    attn = model.model.layers[layer_idx].self_attn
    n_q = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // n_q)

    all_q, all_k, all_v = [], [], []
    batch_size = 64

    for start in range(0, len(padded), batch_size):
        batch = padded[start:start+batch_size].to(DEVICE)
        B, T = batch.shape

        out = model.model(input_ids=batch, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[layer_idx]
        normed = model.model.layers[layer_idx].input_layernorm(hidden)

        k = attn.k_proj(normed).view(B, T, n_kv, head_dim)
        k = attn.k_norm(k).transpose(1, 2)
        all_k.append(k.transpose(1, 2).reshape(B, T, -1).float().cpu())

        v = attn.v_proj(normed).view(B, T, n_kv, head_dim).transpose(1, 2)
        all_v.append(v.transpose(1, 2).reshape(B, T, -1).float().cpu())

        if need_q:
            q = attn.q_proj(normed).view(B, T, n_q, head_dim)
            q = attn.q_norm(q).transpose(1, 2)
            pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1).cpu()
            cos, sin = model.model.rotary_emb(q.cpu(), pos_ids)
            cos, sin = cos.to(DEVICE), sin.to(DEVICE)
            q = apply_rotary(q, cos, sin)
            all_q.append(q.transpose(1, 2).reshape(B, T, -1).float().cpu())

        del out, hidden, normed
        torch.cuda.empty_cache()

    def cat(x):
        if not x:
            return None
        return torch.cat(x, dim=0).reshape(-1, x[0].shape[-1])

    return cat(all_q), cat(all_k), cat(all_v)


# ============================================================
# Ridge 分 batch
# ============================================================
def fit_ridge_batched(X, Y, lam=100.0, batch_size=RIDGE_BATCH):
    D_X = X.shape[-1]
    D_Y = Y.shape[-1]
    XtX = torch.zeros(D_X, D_X, dtype=torch.float32)
    XtY = torch.zeros(D_X, D_Y, dtype=torch.float32)
    N = X.shape[0]

    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        x = X[s:e].float()
        y = Y[s:e].float()
        XtX += x.T @ x
        XtY += x.T @ y

    return torch.linalg.solve(XtX + lam * torch.eye(D_X), XtY)


def apply_W_batched(X, W, batch_size=RIDGE_BATCH):
    N = X.shape[0]
    outs = []
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        outs.append((X[s:e].float() @ W).cpu())
    return torch.cat(outs, 0)


# ============================================================
# Attention 分 batch
# ============================================================
def attention_single(Q, K, V):
    n_rep = Q.shape[1] // K.shape[1]
    K = repeat_kv(K, n_rep)
    V = repeat_kv(V, n_rep)

    scale = 1.0 / (Q.shape[-1] ** 0.5)
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale

    Tq, Tk = scores.shape[-2], scores.shape[-1]
    mask = torch.triu(torch.ones(Tq, Tk, device=Q.device, dtype=torch.bool), 1)
    scores = scores.masked_fill(mask[None, None], torch.finfo(scores.dtype).min)

    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, V), probs


@torch.no_grad()
def attention_batched(Q, K, V, batch_size=ATTN_BATCH):
    B = Q.shape[0]
    outs, probs_list = [], []
    for s in range(0, B, batch_size):
        e = min(s + batch_size, B)
        q = Q[s:e].to(DEVICE).float()
        k = K[s:e].to(DEVICE).float()
        v = V[s:e].to(DEVICE).float()
        o, p = attention_single(q, k, v)
        outs.append(o.cpu())
        probs_list.append(p.cpu())
        del q, k, v, o, p
        torch.cuda.empty_cache()
    return torch.cat(outs, 0), torch.cat(probs_list, 0)


# ============================================================
# 加载模型
# ============================================================
print("\n加载模型...")
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()

model_A.model.rotary_emb.inv_freq = model_A.model.rotary_emb.inv_freq.to(DEVICE)
model_B.model.rotary_emb.inv_freq = model_B.model.rotary_emb.inv_freq.to(DEVICE)

n_layers = model_A.config.num_hidden_layers
n_q = model_B.config.num_attention_heads
n_kv = model_B.config.num_key_value_heads
head_dim = getattr(model_B.config, "head_dim",
                   model_B.config.hidden_size // n_q)

B = len(padded)
T = max_len
pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1).cpu()


# ============================================================
# 逐层扫描
# ============================================================
print(f"\n扫描 {n_layers} 层...")
print("=" * 90)
print(f"{'Layer':>5}  {'K R²':>8}  {'V R²':>8}  {'AttnCos':>9}  {'OutCos':>9}  {'[0,64)':>8}  {'[192,256)':>10}")
print("-" * 90)

results = []

for LAYER in range(n_layers):
    # 提取
    Q_B, K_B, V_B = extract_qkv(model_B, LAYER, padded, need_q=True)
    _,  K_A, V_A = extract_qkv(model_A, LAYER, padded, need_q=False)

    # Ridge
    W_K = fit_ridge_batched(K_A, K_B, lam=100.0)
    W_V = fit_ridge_batched(V_A, V_B, lam=100.0)

    K_mapped = apply_W_batched(K_A, W_K)
    V_mapped = apply_W_batched(V_A, W_V)

    K_r2 = 1 - ((K_mapped - K_B.float())**2).sum() / (K_B.float()**2).sum()
    V_r2 = 1 - ((V_mapped - V_B.float())**2).sum() / (V_B.float()**2).sum()

    # 重建维度
    Q_B = Q_B.view(B, T, n_q, head_dim).transpose(1, 2)
    K_B = K_B.view(B, T, n_kv, head_dim).transpose(1, 2)
    V_B = V_B.view(B, T, n_kv, head_dim).transpose(1, 2)
    K_mapped = K_mapped.view(B, T, n_kv, head_dim).transpose(1, 2)
    V_mapped = V_mapped.view(B, T, n_kv, head_dim).transpose(1, 2)

    # RoPE
    cos, sin = model_B.model.rotary_emb(K_B.cpu(), pos_ids)
    cos = cos.to(DEVICE); sin = sin.to(DEVICE)

    K_B_rope = apply_rotary(K_B.to(DEVICE), cos, sin).cpu()
    K_mapped_rope = apply_rotary(K_mapped.to(DEVICE), cos, sin).cpu()
    V_B = V_B.cpu()
    V_mapped = V_mapped.cpu()

    # Attention
    O_real, A_real = attention_batched(Q_B, K_B_rope, V_B)
    O_trans, A_trans = attention_batched(Q_B, K_mapped_rope, V_mapped)

    # 指标
    O_real_flat = O_real.transpose(1, 2).reshape(B, T, -1)
    O_trans_flat = O_trans.transpose(1, 2).reshape(B, T, -1)

    cos_attn = cos_sim_float64(A_real, A_trans)
    cos_output = cos_sim_float64(O_real_flat, O_trans_flat)

    # 位置分段
    bucket = T // 4
    cos_first = cos_sim_float64(O_real_flat[:, :bucket], O_trans_flat[:, :bucket])
    cos_last = cos_sim_float64(O_real_flat[:, 3*bucket:], O_trans_flat[:, 3*bucket:])

    print(f"{LAYER:5d}  {K_r2.item():8.4f}  {V_r2.item():8.4f}  "
          f"{cos_attn:9.4f}  {cos_output:9.4f}  {cos_first:8.4f}  {cos_last:10.4f}")

    results.append({
        "layer": LAYER,
        "K_r2": K_r2.item(),
        "V_r2": V_r2.item(),
        "attn_cos": cos_attn,
        "out_cos": cos_output,
        "out_cos_first_quarter": cos_first,
        "out_cos_last_quarter": cos_last,
    })

    # 清理
    del Q_B, K_B, V_B, K_A, V_A, K_mapped, V_mapped
    del O_real, A_real, O_trans, A_trans, K_B_rope, K_mapped_rope
    torch.cuda.empty_cache()


# ============================================================
# 汇总
# ============================================================
print()
print("=" * 90)
print("Summary")
print("=" * 90)

K_r2_mean = np.mean([r["K_r2"] for r in results])
V_r2_mean = np.mean([r["V_r2"] for r in results])
attn_cos_mean = np.mean([r["attn_cos"] for r in results])
out_cos_mean = np.mean([r["out_cos"] for r in results])

print(f"K R² mean         : {K_r2_mean:.4f}")
print(f"V R² mean         : {V_r2_mean:.4f}")
print(f"Attention cos mean: {attn_cos_mean:.4f}")
print(f"Output cos mean   : {out_cos_mean:.4f}")

best = max(results, key=lambda x: x["out_cos"])
worst = min(results, key=lambda x: x["out_cos"])
print(f"\nBest layer  : {best['layer']}  out_cos={best['out_cos']:.4f}")
print(f"Worst layer : {worst['layer']}  out_cos={worst['out_cos']:.4f}")

# 保存
with open("layer_scan_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nSaved: layer_scan_results.json")