import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
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
LAYER = 14
DEVICE = "cuda"
SEED = 42

RIDGE_BATCH = 50000
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
print(f"文章数: {B}, 每篇最多 {T} token, 总 token: {B * T}")


# ============================================================
# 工具
# ============================================================
def r2_score(pred, target):
    pred = pred.float()
    target = target.float()
    return (1 - ((pred - target)**2).sum() / (target**2).sum()).item()


def fit_ridge(X, Y, lam=100.0, batch_size=RIDGE_BATCH):
    D_X = X.shape[-1]
    D_Y = Y.shape[-1]
    XtX = torch.zeros(D_X, D_X, dtype=torch.float32)
    XtY = torch.zeros(D_X, D_Y, dtype=torch.float32)
    N_ = X.shape[0]
    for s in range(0, N_, batch_size):
        e = min(s + batch_size, N_)
        x = X[s:e].float(); y = Y[s:e].float()
        XtX += x.T @ x
        XtY += x.T @ y
    return torch.linalg.solve(XtX + lam * torch.eye(D_X), XtY)


# ============================================================
# 提取 V
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

n_kv = model_B.config.num_key_value_heads
head_dim = getattr(model_B.config, "head_dim",
                   model_B.config.hidden_size // model_B.config.num_attention_heads)
print(f"n_kv={n_kv}, head_dim={head_dim}, total_dim={n_kv * head_dim}")


# ============================================================
# 提取 V
# ============================================================
print(f"\n提取第 {LAYER} 层 V...")
V_A = extract_v(model_A, LAYER, padded)
V_B = extract_v(model_B, LAYER, padded)
N = V_A.shape[0]
D = V_A.shape[1]
print(f"  V_A: {V_A.shape}, V_B: {V_B.shape}")

# train/test
perm = torch.randperm(N)
n_train = int(N * TRAIN_RATIO)
train_idx = perm[:n_train]
test_idx = perm[n_train:]
print(f"  train: {len(train_idx)}, test: {len(test_idx)}")


# ============================================================
# 方法 0：Baseline 全局线性 W
# ============================================================
print("\n[0] Baseline 全局线性 W...")
W_global = fit_ridge(V_A[train_idx], V_B[train_idx])
V_global_test = V_A[test_idx].float() @ W_global
r2_global = r2_score(V_global_test, V_B[test_idx])
print(f"  Test R² = {r2_global:.4f}")


# ============================================================
# 方法 1：逐头 V 映射
# ============================================================
print("\n[1] 逐头 V 映射...")

V_head_test = torch.zeros_like(V_B[test_idx])
W_head_list = []

for h in range(n_kv):
    s_h = h * head_dim
    e_h = (h + 1) * head_dim

    V_A_h = V_A[:, s_h:e_h]     # [N, 128]
    V_B_h = V_B[:, s_h:e_h]

    W_h = fit_ridge(V_A_h[train_idx], V_B_h[train_idx])
    W_head_list.append(W_h)

    V_h_test = V_A_h[test_idx].float() @ W_h
    r2_h = r2_score(V_h_test, V_B_h[test_idx])
    print(f"  Head {h}: R² = {r2_h:.4f}")

    V_head_test[:, s_h:e_h] = V_h_test

r2_per_head = r2_score(V_head_test, V_B[test_idx])
print(f"  整体 Test R² = {r2_per_head:.4f}")


# ============================================================
# 方法 2：分组逐头（每 2 个头共享一个 W）
# ============================================================
print("\n[2] 分组逐头（每 2 个头共享）...")

group_size = 2
n_groups = n_kv // group_size
V_group_test = torch.zeros_like(V_B[test_idx])

for g in range(n_groups):
    s_g = g * group_size * head_dim
    e_g = (g + 1) * group_size * head_dim

    V_A_g = V_A[:, s_g:e_g]
    V_B_g = V_B[:, s_g:e_g]

    W_g = fit_ridge(V_A_g[train_idx], V_B_g[train_idx])
    V_g_test = V_A_g[test_idx].float() @ W_g
    V_group_test[:, s_g:e_g] = V_g_test

r2_group = r2_score(V_group_test, V_B[test_idx])
print(f"  Test R² = {r2_group:.4f}")


# ============================================================
# 方法 3：逐头 + 全局残差
#    V_pred = V_head + (V_global - V_head) * α  （可选）
# 简化：逐头映射 + 全局映射的平均
# ============================================================
print("\n[3] 逐头 + 全局 平均...")
V_avg_test = 0.5 * V_head_test + 0.5 * V_global_test
r2_avg = r2_score(V_avg_test, V_B[test_idx])
print(f"  Test R² = {r2_avg:.4f}")


# ============================================================
# 汇总
# ============================================================
print()
print("=" * 70)
print(f"Per-Head V Mapping (Layer {LAYER})")
print("=" * 70)
print(f"{'Method':>40}  {'Test R²':>10}")
print("-" * 70)
print(f"{'0. Global Linear W':>40}  {r2_global:10.4f}")
print(f"{'1. Per-Head (8 W)':>40}  {r2_per_head:10.4f}")
print(f"{'2. Grouped Per-Head (2 heads, 4 W)':>40}  {r2_group:10.4f}")
print(f"{'3. Per-Head + Global Average':>40}  {r2_avg:10.4f}")
print("=" * 70)


# ============================================================
# 参数量对比
# ============================================================
print()
print("参数量对比：")
print(f"  Global:   {D}x{D} = {D*D:,}")
print(f"  Per-Head: 8 x {head_dim}x{head_dim} = {8*head_dim*head_dim:,}")
print(f"  Grouped:  4 x {2*head_dim}x{2*head_dim} = {4*(2*head_dim)*(2*head_dim):,}")