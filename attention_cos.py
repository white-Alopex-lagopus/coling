import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置
# ============================================================
MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"
DATA_PATH = "../dataset/wikitext-02/train.jsonl"

MAX_ARTICLES = 2000
MAX_TOKENS_PER_ART = 128
LAYER = 14
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
print(f"总 token: {len(article_ids) * max_len}")


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
    """标准 GQA 展开：[k0,k1,...] → [k0,k1,...,k0,k1,...]"""
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, H, n_rep, T, D)
        .reshape(B, H * n_rep, T, D)
    )


def cos_sim_float64(A, B):
    """float64 手动 cosine，避免精度问题"""
    A = A.double().reshape(-1)
    B = B.double().reshape(-1)
    dot = (A * B).sum()
    nA = (A ** 2).sum().sqrt()
    nB = (B ** 2).sum().sqrt()
    return (dot / (nA * nB + 1e-12)).item()


# ============================================================
# 提取 Q/K/V
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

        # K (pre-RoPE)
        k = attn.k_proj(normed).view(B, T, n_kv, head_dim)
        k = attn.k_norm(k).transpose(1, 2)
        all_k.append(k.transpose(1, 2).reshape(B, T, -1).float().cpu())

        # V
        v = attn.v_proj(normed).view(B, T, n_kv, head_dim).transpose(1, 2)
        all_v.append(v.transpose(1, 2).reshape(B, T, -1).float().cpu())

        # Q (post-RoPE)
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
# 加载模型
# ============================================================
print("\n加载模型...")
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()

model_A.model.rotary_emb.inv_freq = model_A.model.rotary_emb.inv_freq.to(DEVICE)
model_B.model.rotary_emb.inv_freq = model_B.model.rotary_emb.inv_freq.to(DEVICE)


# ============================================================
# 提取
# ============================================================
print(f"\n提取第 {LAYER} 层 Q/K/V...")

Q_B, K_B, V_B = extract_qkv(model_B, LAYER, padded, need_q=True)
_,  K_A, V_A = extract_qkv(model_A, LAYER, padded, need_q=False)

print(f"  Q_B: {Q_B.shape}")
print(f"  K_A: {K_A.shape}, K_B: {K_B.shape}")
print(f"  V_A: {V_A.shape}, V_B: {V_B.shape}")


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

    W = torch.linalg.solve(XtX + lam * torch.eye(D_X), XtY)
    return W


def apply_W_batched(X, W, batch_size=RIDGE_BATCH):
    N = X.shape[0]
    outs = []
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        outs.append((X[s:e].float() @ W).cpu())
    return torch.cat(outs, 0)


print("\n学 W_K...")
W_K = fit_ridge_batched(K_A, K_B, lam=100.0)

print("学 W_V...")
W_V = fit_ridge_batched(V_A, V_B, lam=100.0)

K_A_mapped = apply_W_batched(K_A, W_K)
V_A_mapped = apply_W_batched(V_A, W_V)

K_r2 = 1 - ((K_A_mapped - K_B.float())**2).sum() / (K_B.float()**2).sum()
V_r2 = 1 - ((V_A_mapped - V_B.float())**2).sum() / (V_B.float()**2).sum()
print(f"  K R²: {K_r2.item():.4f}")
print(f"  V R²: {V_r2.item():.4f}")


# ============================================================
# 重建维度 + RoPE
# ============================================================
n_q = model_B.config.num_attention_heads
n_kv = model_B.config.num_key_value_heads
head_dim = getattr(model_B.config, "head_dim",
                   model_B.config.hidden_size // n_q)

B = len(padded)
T = max_len

Q_B = Q_B.view(B, T, n_q, head_dim).transpose(1, 2)
K_B = K_B.view(B, T, n_kv, head_dim).transpose(1, 2)
V_B = V_B.view(B, T, n_kv, head_dim).transpose(1, 2)
K_mapped = K_A_mapped.view(B, T, n_kv, head_dim).transpose(1, 2)
V_mapped = V_A_mapped.view(B, T, n_kv, head_dim).transpose(1, 2)

pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1).cpu()
cos, sin = model_B.model.rotary_emb(K_B.cpu(), pos_ids)
cos = cos.to(DEVICE)
sin = sin.to(DEVICE)

K_B = K_B.to(DEVICE)
K_mapped = K_mapped.to(DEVICE)

K_B_rope = apply_rotary(K_B, cos, sin)
K_mapped_rope = apply_rotary(K_mapped, cos, sin)

K_B_rope = K_B_rope.cpu()
K_mapped_rope = K_mapped_rope.cpu()
V_B = V_B.cpu()
V_mapped = V_mapped.cpu()

# ============================================================
# 调试打印
# ============================================================
print("\n" + "=" * 70)
print("Sanity check")
print("=" * 70)
for name, t in [("Q_B", Q_B), ("K_B_rope", K_B_rope), ("V_B", V_B),
                ("K_mapped_rope", K_mapped_rope), ("V_mapped", V_mapped)]:
    print(f"{name:20s}  shape={tuple(t.shape)}  "
          f"nan={torch.isnan(t).sum().item()}  "
          f"inf={torch.isinf(t).sum().item()}  "
          f"min={t.min().item():.3f}  max={t.max().item():.3f}")


# ============================================================
# Attention（分 batch，标准 repeat_kv）
# ============================================================
def attention_single(Q, K, V):
    """
    Q: [b, H_q, T, D]  GPU float32
    K: [b, H_kv, T, D] GPU float32
    V: [b, H_kv, T, D] GPU float32
    """
    n_rep = Q.shape[1] // K.shape[1]
    K = repeat_kv(K, n_rep)
    V = repeat_kv(V, n_rep)

    scale = 1.0 / (Q.shape[-1] ** 0.5)
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale

    Tq, Tk = scores.shape[-2], scores.shape[-1]
    mask = torch.triu(torch.ones(Tq, Tk, device=Q.device, dtype=torch.bool), 1)
    scores = scores.masked_fill(mask[None, None], torch.finfo(scores.dtype).min)

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, V)
    return out, probs


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


print("\n计算真实 attention...")
O_real, A_real = attention_batched(Q_B, K_B_rope, V_B)

print("计算传递 attention...")
O_trans, A_trans = attention_batched(Q_B, K_mapped_rope, V_mapped)


# ============================================================
# 指标（float64 手动 cos）
# ============================================================
O_real_flat = O_real.transpose(1, 2).reshape(B, T, -1)
O_trans_flat = O_trans.transpose(1, 2).reshape(B, T, -1)

cos_output = cos_sim_float64(O_real_flat, O_trans_flat)
cos_attn = cos_sim_float64(A_real, A_trans)

print()
print("=" * 70)
print(f"Layer {LAYER} KV Transfer Result")
print("=" * 70)
print(f"K mapping R²     : {K_r2.item():.4f}")
print(f"V mapping R²     : {V_r2.item():.4f}")
print(f"Attention cosine : {cos_attn:.4f}")
print(f"Output cosine    : {cos_output:.4f}")
print("=" * 70)

# 位置分段
print("\n位置分段 output cosine:")
bucket = T // 4
for b in range(4):
    lo = b * bucket
    hi = (b + 1) * bucket if b < 3 else T
    c = cos_sim_float64(O_real_flat[:, lo:hi], O_trans_flat[:, lo:hi])
    print(f"  [{lo:4d}, {hi:4d}): {c:.4f}")