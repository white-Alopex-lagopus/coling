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

MAX_ARTICLES = 2000
MAX_TOKENS_PER_ART = 256
LAYER = 14                   # 先用中间层测试
DEVICE = "cuda"
SEED = 42

RIDGE_BATCH = 50000

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
print(f"文章数: {B}, 每篇最多 {T} token")


# ============================================================
# 工具
# ============================================================
def cos_sim_batched(A, B, batch_size=64):
    A = A.reshape(A.shape[0], -1)
    B = B.reshape(B.shape[0], -1)
    N = A.shape[0]
    dot = 0.0
    nA = 0.0
    nB = 0.0
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        a = A[s:e].double()
        b = B[s:e].double()
        dot += (a * b).sum().item()
        nA += (a ** 2).sum().item()
        nB += (b ** 2).sum().item()
        del a, b
    return dot / (np.sqrt(nA) * np.sqrt(nB) + 1e-12)


# ============================================================
# Ridge 分 batch
# ============================================================
def fit_ridge(X, Y, lam=100.0, batch_size=RIDGE_BATCH):
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


def apply_W(X, W, batch_size=RIDGE_BATCH):
    N = X.shape[0]
    outs = []
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        outs.append((X[s:e].float() @ W).cpu())
    return torch.cat(outs, 0)


# ============================================================
# 提取 V（单层）
# ============================================================
@torch.no_grad()
def extract_v(model, layer_idx, padded):
    attn = model.model.layers[layer_idx].self_attn
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)

    all_v = []
    batch_size = 64

    for start in range(0, len(padded), batch_size):
        batch = padded[start:start+batch_size].to(DEVICE)
        B_b, T_b = batch.shape

        out = model.model(input_ids=batch, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[layer_idx]
        normed = model.model.layers[layer_idx].input_layernorm(hidden)

        v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
        all_v.append(v.transpose(1, 2).reshape(B_b, T_b, -1).float().cpu())

        del out, hidden, normed
        torch.cuda.empty_cache()

    return torch.cat(all_v, dim=0).reshape(-1, all_v[0].shape[-1])


# ============================================================
# 加载模型
# ============================================================
print("\n加载模型...")
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()


# ============================================================
# 提取 V
# ============================================================
print(f"\n提取第 {LAYER} 层 V...")
V_A = extract_v(model_A, LAYER, padded)   # [N, D]
V_B = extract_v(model_B, LAYER, padded)
N = V_A.shape[0]
print(f"  V_A: {V_A.shape}, V_B: {V_B.shape}")

# 位置标记（每个 token 属于哪个位置）
positions = torch.arange(T).unsqueeze(0).expand(B, -1).reshape(-1)   # [N]


# ============================================================
# 方法 1：全局单 W
# ============================================================
print("\n[方法 1] 全局单 W...")
W_global = fit_ridge(V_A, V_B, lam=100.0)
V_global = apply_W(V_A, W_global)
r2_global = 1 - ((V_global - V_B.float())**2).sum() / (V_B.float()**2).sum()
print(f"  R² = {r2_global.item():.4f}")


# ============================================================
# 方法 2：分段 W（每 32 位置一个）
# ============================================================
print("\n[方法 2] 分段 W（每 32 位置）...")
SEG = 32
n_seg = (T + SEG - 1) // SEG
V_seg = torch.zeros_like(V_B)
r2_seg_list = []

for s in range(n_seg):
    lo = s * SEG
    hi = min((s + 1) * SEG, T)
    mask = (positions >= lo) & (positions < hi)

    if mask.sum() < 100:
        continue

    V_A_seg = V_A[mask]
    V_B_seg = V_B[mask]

    W_seg = fit_ridge(V_A_seg, V_B_seg, lam=100.0)
    V_seg_pred = apply_W(V_A_seg, W_seg)
    V_seg[mask] = V_seg_pred

    r2_seg = 1 - ((V_seg_pred - V_B_seg.float())**2).sum() / (V_B_seg.float()**2).sum()
    r2_seg_list.append(r2_seg.item())
    print(f"  位置 [{lo:3d}, {hi:3d}): R² = {r2_seg.item():.4f}  (n={mask.sum().item()})")

r2_seg_overall = 1 - ((V_seg - V_B.float())**2).sum() / (V_B.float()**2).sum()
print(f"  整体 R² = {r2_seg_overall.item():.4f}")


# ============================================================
# 方法 3：位置条件 W（W = W_base + pos * W_slope）
# ============================================================
print("\n[方法 3] 位置条件 W...")
# 构造特征：X 和 X * pos_norm 拼接
pos_norm = (positions.float() / T).unsqueeze(-1)   # [N, 1]
V_A_aug = torch.cat([V_A, V_A * pos_norm], dim=-1)  # [N, 2*D]

W_aug = fit_ridge(V_A_aug, V_B, lam=100.0)
V_aug_pred = apply_W(V_A_aug, W_aug)
r2_aug = 1 - ((V_aug_pred - V_B.float())**2).sum() / (V_B.float()**2).sum()
print(f"  R² = {r2_aug.item():.4f}")


# ============================================================
# 汇总
# ============================================================
print()
print("=" * 70)
print(f"V Mapping Comparison (Layer {LAYER})")
print("=" * 70)
print(f"{'Method':>30}  {'R²':>10}")
print("-" * 70)
print(f"{'Global single W':>30}  {r2_global.item():10.4f}")
print(f"{'Segmented W (32 positions)':>30}  {r2_seg_overall.item():10.4f}")
print(f"{'Position-conditioned W':>30}  {r2_aug.item():10.4f}")
print("=" * 70)