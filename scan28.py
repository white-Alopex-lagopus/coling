import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置
# ============================================================
MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"
DATA_PATH = "../dataset/wikitext-02/train.jsonl"

MAX_ARTICLES = 1981
MAX_TOKENS_PER_ART = 128
DEVICE = "cuda"
SEED = 42

RIDGE_BATCH = 50000
ATTN_BATCH = 16           # 按文章数
TRAIN_RATIO = 0.8

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

B = len(padded)
T = max_len
N = B * T
print(f"文章数: {B}, 每篇最多 {T} token, 总 token: {N}")


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
    B_, H, T_, D = x.shape
    return x[:, :, None, :, :].expand(B_, H, n_rep, T_, D).reshape(B_, H * n_rep, T_, D)


def r2_score(pred, target):
    pred = pred.float()
    target = target.float()
    return (1 - ((pred - target)**2).sum() / (target**2).sum()).item()


# ============================================================
# Ridge
# ============================================================
def fit_ridge_batched(X, Y, idx=None, lam=100.0, batch_size=RIDGE_BATCH):
    D_X = X.shape[-1]
    D_Y = Y.shape[-1]
    XtX = torch.zeros(D_X, D_X, dtype=torch.float32)
    XtY = torch.zeros(D_X, D_Y, dtype=torch.float32)

    if idx is None:
        N_ = X.shape[0]
        for s in range(0, N_, batch_size):
            e = min(s + batch_size, N_)
            x = X[s:e].float(); y = Y[s:e].float()
            XtX += x.T @ x
            XtY += x.T @ y
    else:
        N_ = len(idx)
        for s in range(0, N_, batch_size):
            e = min(s + batch_size, N_)
            bi = idx[s:e]
            x = X[bi].float(); y = Y[bi].float()
            XtX += x.T @ x
            XtY += x.T @ y

    return torch.linalg.solve(XtX + lam * torch.eye(D_X), XtY)


def apply_W_batched(X, W, batch_size=RIDGE_BATCH):
    N_ = X.shape[0]
    outs = []
    for s in range(0, N_, batch_size):
        e = min(s + batch_size, N_)
        outs.append((X[s:e].float() @ W).cpu())
    return torch.cat(outs, 0)


# ============================================================
# 提取 Q/K/V（fp16 存）
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
        B_b, T_b = batch.shape

        out = model.model(input_ids=batch, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[layer_idx]
        normed = model.model.layers[layer_idx].input_layernorm(hidden)

        k = attn.k_proj(normed).view(B_b, T_b, n_kv, head_dim)
        k = attn.k_norm(k).transpose(1, 2)
        all_k.append(k.transpose(1, 2).reshape(B_b, T_b, -1).to(torch.float16).cpu())

        v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
        all_v.append(v.transpose(1, 2).reshape(B_b, T_b, -1).to(torch.float16).cpu())

        if need_q:
            q = attn.q_proj(normed).view(B_b, T_b, n_q, head_dim)
            q = attn.q_norm(q).transpose(1, 2)
            pos_ids = torch.arange(T_b).unsqueeze(0).expand(B_b, -1).cpu()
            cos, sin = model.model.rotary_emb(q.cpu(), pos_ids)
            cos, sin = cos.to(DEVICE), sin.to(DEVICE)
            q = apply_rotary(q, cos, sin)
            all_q.append(q.transpose(1, 2).reshape(B_b, T_b, -1).to(torch.float16).cpu())

        del out, hidden, normed
        torch.cuda.empty_cache()

    def cat(x):
        if not x:
            return None
        return torch.cat(x, dim=0).reshape(-1, x[0].shape[-1])

    return cat(all_q), cat(all_k), cat(all_v)


# ============================================================
# Attention（单 batch）
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


# ============================================================
# 流式 cosine（按文章 batch）
# ============================================================
@torch.no_grad()
def streaming_cos_layer(Q_B_rope, K_A, V_A, K_B, V_B, W_K, W_V,
                        cos_full, sin_full, n_q, n_kv, head_dim,
                        B, T, DEVICE, batch_articles=16):
    """
    三层映射：Identity / Linear V
    流式累积 cos，不保留全量 A/O
    """
    acc = {}
    for name in ["Identity", "Linear"]:
        acc[name] = {"dO": 0.0, "nOr": 0.0, "nOt": 0.0,
                     "dA": 0.0, "nAr": 0.0, "nAt": 0.0}

    for b_start in range(0, B, batch_articles):
        b_end = min(b_start + batch_articles, B)
        b = b_end - b_start

        # 取当前 batch 的切片
        s = b_start * T
        e = b_end * T

        q = Q_B_rope[s:e].to(DEVICE).float()           # 已经带 RoPE，[b*T, n_q*D]
        k_ref = K_B[s:e].to(DEVICE).float()            # pre-RoPE
        v_ref = V_B[s:e].to(DEVICE).float()
        k_src = K_A[s:e].to(DEVICE).float()
        v_src = V_A[s:e].to(DEVICE).float()

        # reshape
        q = q.view(b, T, n_q, head_dim).transpose(1, 2)
        k_ref = k_ref.view(b, T, n_kv, head_dim).transpose(1, 2)
        v_ref = v_ref.view(b, T, n_kv, head_dim).transpose(1, 2)
        k_src = k_src.view(b, T, n_kv, head_dim).transpose(1, 2)
        v_src = v_src.view(b, T, n_kv, head_dim).transpose(1, 2)

        # 当前 batch 的 RoPE
        c = cos_full[b_start:b_end].to(DEVICE).float()
        sn = sin_full[b_start:b_end].to(DEVICE).float()

        # Real attention
        k_ref_rope = apply_rotary(k_ref, c, sn)
        O_r, A_r = attention_single(q, k_ref_rope, v_ref)

        # ----- Identity -----
        k_id_rope = apply_rotary(k_src, c, sn)
        O_t, A_t = attention_single(q, k_id_rope, v_src)
        _accumulate(acc["Identity"], O_r, A_r, O_t, A_t)
        del k_id_rope, O_t, A_t

        # ----- Linear V -----
        # 映射在 [b, T, n_kv*head_dim] 上做
        k_flat = k_src.transpose(1, 2).reshape(b, T, -1)         # [b, T, n_kv*D]
        v_flat = v_src.transpose(1, 2).reshape(b, T, -1)

        k_lin_flat = (k_flat.reshape(-1, k_flat.shape[-1]).cpu().float() @ W_K).to(DEVICE)
        v_lin_flat = (v_flat.reshape(-1, v_flat.shape[-1]).cpu().float() @ W_V).to(DEVICE)

        k_lin = k_lin_flat.view(b, T, n_kv, head_dim).transpose(1, 2)
        v_lin = v_lin_flat.view(b, T, n_kv, head_dim).transpose(1, 2)

        k_lin_rope = apply_rotary(k_lin, c, sn)
        O_t, A_t = attention_single(q, k_lin_rope, v_lin)
        _accumulate(acc["Linear"], O_r, A_r, O_t, A_t)

        del q, k_ref, v_ref, k_src, v_src, k_ref_rope, O_r, A_r
        del k_flat, v_flat, k_lin_flat, v_lin_flat, k_lin, v_lin, k_lin_rope
        del O_t, A_t
        torch.cuda.empty_cache()

    out = {}
    for name, a in acc.items():
        cos_O = a["dO"] / (np.sqrt(a["nOr"]) * np.sqrt(a["nOt"]) + 1e-12)
        cos_A = a["dA"] / (np.sqrt(a["nAr"]) * np.sqrt(a["nAt"]) + 1e-12)
        out[name] = (cos_O, cos_A)
    return out


def _accumulate(a, O_r, A_r, O_t, A_t):
    O_r_f = O_r.reshape(O_r.shape[0], -1).double()
    O_t_f = O_t.reshape(O_t.shape[0], -1).double()
    A_r_f = A_r.reshape(A_r.shape[0], -1).double()
    A_t_f = A_t.reshape(A_t.shape[0], -1).double()
    a["dO"] += (O_r_f * O_t_f).sum().item()
    a["nOr"] += (O_r_f ** 2).sum().item()
    a["nOt"] += (O_t_f ** 2).sum().item()
    a["dA"] += (A_r_f * A_t_f).sum().item()
    a["nAr"] += (A_r_f ** 2).sum().item()
    a["nAt"] += (A_t_f ** 2).sum().item()


# ============================================================
# 加载模型
# ============================================================
print("\n加载模型...")
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()

model_A.model.rotary_emb.inv_freq = model_A.model.rotary_emb.inv_freq.to(DEVICE)
model_B.model.rotary_emb.inv_freq = model_B.model.rotary_emb.inv_freq.to(DEVICE)

n_q = model_B.config.num_attention_heads
n_kv = model_B.config.num_key_value_heads
head_dim = getattr(model_B.config, "head_dim",
                   model_B.config.hidden_size // n_q)
n_layers = model_B.config.num_hidden_layers

print(f"Layers: {n_layers}, n_q={n_q}, n_kv={n_kv}, head_dim={head_dim}")


# ============================================================
# 预计算 RoPE（1.7B，所有文章）
# ============================================================
pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1).cpu()
_cos, _sin = model_B.model.rotary_emb(
    torch.zeros(B, T, head_dim), pos_ids
)
cos_full = _cos.float().cpu()
sin_full = _sin.float().cpu()
print(f"RoPE cached: cos={cos_full.shape}")


# ============================================================
# 逐层扫描
# ============================================================
print()
print("=" * 95)
print(f"{'Layer':>5}  {'K R²':>8}  {'V R²':>8}  {'AttnCos':>9}  {'OutCos':>9}  {'[0,64)':>8}  {'[64,128)':>10}")
print("-" * 95)

results = []

for LAYER in range(n_layers):
    try:
        # 提取
        Q_B, K_B, V_B = extract_qkv(model_B, LAYER, padded, need_q=True)
        _,  K_A, V_A = extract_qkv(model_A, LAYER, padded, need_q=False)

        # train 划分
        perm = torch.randperm(N)
        n_train = int(N * TRAIN_RATIO)
        train_idx = perm[:n_train]

        # 训练 W_K, W_V
        W_K = fit_ridge_batched(K_A, K_B, idx=train_idx, lam=100.0)
        W_V = fit_ridge_batched(V_A, V_B, idx=train_idx, lam=100.0)

        # K/V R²
        K_lin = apply_W_batched(K_A, W_K)
        V_lin = apply_W_batched(V_A, W_V)
        K_r2 = r2_score(K_lin, K_B.float())
        V_r2 = r2_score(V_lin, V_B.float())

        # reshape Q
        Q_B_ = Q_B.view(B, T, n_q, head_dim).transpose(1, 2)

        # 流式 cos
        out = streaming_cos_layer(
            Q_B, K_A, V_A, K_B, V_B, W_K, W_V,
            cos_full, sin_full, n_q, n_kv, head_dim,
            B, T, DEVICE, batch_articles=16
        )

        outcos_id = out["Identity"][0]
        outcos_lin = out["Linear"][0]
        attncos_lin = out["Linear"][1]

        # 位置分段（用 Linear 结果，需重新跑一遍带位置累积，简化为全量）
        # 简化为只输出整体
        cos_first = None
        cos_last = None

        print(f"{LAYER:5d}  {K_r2:8.4f}  {V_r2:8.4f}  {attncos_lin:9.4f}  "
              f"{outcos_lin:9.4f}  {'-':>8}  {'-':>10}")

        results.append({
            "layer": LAYER,
            "K_r2": K_r2,
            "V_r2": V_r2,
            "attn_cos": attncos_lin,
            "out_cos": outcos_lin,
            "out_cos_identity": outcos_id,
        })

        # 清理
        del Q_B, K_B, V_B, K_A, V_A, K_lin, V_lin, W_K, W_V
        del Q_B_, out
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"{LAYER:5d}  ERROR: {e}")
        results.append({"layer": LAYER, "error": str(e)})
        gc.collect()
        torch.cuda.empty_cache()


# ============================================================
# 汇总
# ============================================================
print()
print("=" * 95)
print("Summary")
print("=" * 95)

ok = [r for r in results if "error" not in r]
if ok:
    K_r2_mean = np.mean([r["K_r2"] for r in ok])
    V_r2_mean = np.mean([r["V_r2"] for r in ok])
    attn_cos_mean = np.mean([r["attn_cos"] for r in ok])
    out_cos_mean = np.mean([r["out_cos"] for r in ok])
    out_cos_id_mean = np.mean([r["out_cos_identity"] for r in ok])

    print(f"K R² mean         : {K_r2_mean:.4f}")
    print(f"V R² mean         : {V_r2_mean:.4f}")
    print(f"AttnCos mean      : {attn_cos_mean:.4f}")
    print(f"OutCos mean       : {out_cos_mean:.4f}")
    print(f"OutCos identity   : {out_cos_id_mean:.4f}")

    best = max(ok, key=lambda x: x["out_cos"])
    worst = min(ok, key=lambda x: x["out_cos"])
    print(f"\nBest  layer: {best['layer']}  out_cos={best['out_cos']:.4f}")
    print(f"Worst layer: {worst['layer']}  out_cos={worst['out_cos']:.4f}")

    with open("layer_scan_28.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved: layer_scan_28.json")