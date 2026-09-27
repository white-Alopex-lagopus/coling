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
# 方法 0：Baseline 线性 W
# ============================================================
print("\n[0] Baseline 线性 W...")

# 在 train 上拟合
XtX = torch.zeros(D, D)
XtY = torch.zeros(D, D)
for s in range(0, len(train_idx), RIDGE_BATCH):
    e = min(s + RIDGE_BATCH, len(train_idx))
    idx = train_idx[s:e]
    x = V_A[idx].float(); y = V_B[idx].float()
    XtX += x.T @ x
    XtY += x.T @ y

W_linear = torch.linalg.solve(XtX + 100.0 * torch.eye(D), XtY)

# 在 test 上评估
V_linear_test = V_A[test_idx].float() @ W_linear
r2_linear = r2_score(V_linear_test, V_B[test_idx])
print(f"  Test R² = {r2_linear:.4f}")


# ============================================================
# 方法 1：方向/幅度分解
# ============================================================
print("\n[1] 方向/幅度分解...")

eps = 1e-8

# 分解（在 train + test 都算，因为分解是纯数学操作，不涉及拟合）
V_A_mag = V_A.norm(dim=-1, keepdim=True)                # [N, 1]
V_A_dir = V_A / (V_A_mag + eps)                          # [N, D]

V_B_mag = V_B.norm(dim=-1, keepdim=True)                # [N, 1]
V_B_dir = V_B / (V_B_mag + eps)                          # [N, D]

# ---- 幅度映射：简单线性 y = a*x + b ----
x_mag = V_A_mag[train_idx].squeeze(-1).float()           # [n_train]
y_mag = V_B_mag[train_idx].squeeze(-1).float()           # [n_train]

A_mat = torch.stack([x_mag, torch.ones_like(x_mag)], dim=1)   # [n_train, 2]
coef = torch.linalg.lstsq(A_mat, y_mag.unsqueeze(1)).solution  # [2, 1]
a, b = coef[0].item(), coef[1].item()
print(f"  幅度映射: y = {a:.4f} * x + {b:.4f}")

# test 幅度预测
V_mag_pred_test = a * V_A_mag[test_idx].squeeze(-1).float() + b
V_mag_r2 = r2_score(V_mag_pred_test, V_B_mag[test_idx].squeeze(-1).float())
print(f"  幅度 Test R² = {V_mag_r2:.4f}")

# ---- 方向映射：Ridge ----
XtX_d = torch.zeros(D, D)
XtY_d = torch.zeros(D, D)
for s in range(0, len(train_idx), RIDGE_BATCH):
    e = min(s + RIDGE_BATCH, len(train_idx))
    idx = train_idx[s:e]
    x = V_A_dir[idx].float(); y = V_B_dir[idx].float()
    XtX_d += x.T @ x
    XtY_d += x.T @ y

W_dir = torch.linalg.solve(XtX_d + 100.0 * torch.eye(D), XtY_d)

# test 方向预测
V_dir_pred_test = V_A_dir[test_idx].float() @ W_dir
V_dir_r2 = r2_score(V_dir_pred_test, V_B_dir[test_idx].float())
print(f"  方向 Test R² = {V_dir_r2:.4f}")

# ---- 重组 ----
V_decomp_test = V_mag_pred_test.unsqueeze(-1) * V_dir_pred_test
r2_decomp = r2_score(V_decomp_test, V_B[test_idx])
print(f"  重组 Test R² = {r2_decomp:.4f}")


# ============================================================
# 方法 2：归一化 + Ridge（只映射方向，幅度直接沿用）
# ============================================================
print("\n[2] 方向 Ridge + 幅度沿用...")

# 方向用 Ridge，幅度不映射，直接用 A 的幅度
V_norm_test = V_A_mag[test_idx].float() * V_dir_pred_test
r2_norm = r2_score(V_norm_test, V_B[test_idx])
print(f"  Test R² = {r2_norm:.4f}")


# ============================================================
# 方法 3：逐头方向/幅度分解
# ============================================================
print("\n[3] 逐头方向/幅度分解...")

n_kv = 8
head_dim = 128

V_decomp_head = torch.zeros_like(V_B[test_idx])

for h in range(n_kv):
    s_h = h * head_dim
    e_h = (h + 1) * head_dim

    # 分解当前 head
    V_A_h = V_A[:, s_h:e_h]                          # [N, 128]
    V_B_h = V_B[:, s_h:e_h]

    V_A_h_mag = V_A_h.norm(dim=-1, keepdim=True)
    V_A_h_dir = V_A_h / (V_A_h_mag + eps)

    V_B_h_mag = V_B_h.norm(dim=-1, keepdim=True)
    V_B_h_dir = V_B_h / (V_B_h_mag + eps)

    # 幅度映射
    x = V_A_h_mag[train_idx].squeeze(-1).float()
    y = V_B_h_mag[train_idx].squeeze(-1).float()
    A_mat = torch.stack([x, torch.ones_like(x)], dim=1)
    coef = torch.linalg.lstsq(A_mat, y.unsqueeze(1)).solution
    a_h, b_h = coef[0].item(), coef[1].item()

    V_h_mag_pred = a_h * V_A_h_mag[test_idx].squeeze(-1).float() + b_h

    # 方向映射
    XtX_h = torch.zeros(head_dim, head_dim)
    XtY_h = torch.zeros(head_dim, head_dim)
    for s in range(0, len(train_idx), RIDGE_BATCH):
        ee = min(s + RIDGE_BATCH, len(train_idx))
        idx = train_idx[s:ee]
        xx = V_A_h_dir[idx].float(); yy = V_B_h_dir[idx].float()
        XtX_h += xx.T @ xx
        XtY_h += xx.T @ yy

    W_h_dir = torch.linalg.solve(XtX_h + 100.0 * torch.eye(head_dim), XtY_h)

    V_h_dir_pred = V_A_h_dir[test_idx].float() @ W_h_dir

    # 重组
    V_decomp_head[:, s_h:e_h] = V_h_mag_pred.unsqueeze(-1) * V_h_dir_pred

r2_decomp_head = r2_score(V_decomp_head, V_B[test_idx])
print(f"  Test R² = {r2_decomp_head:.4f}")


# ============================================================
# 汇总
# ============================================================
print()
print("=" * 70)
print(f"V Mapping: Direction/Magnitude Decomposition (Layer {LAYER})")
print("=" * 70)
print(f"{'Method':>40}  {'Test R²':>10}")
print("-" * 70)
print(f"{'0. Baseline Linear W':>40}  {r2_linear:10.4f}")
print(f"{'1. Global Dir/Mag Decomposition':>40}  {r2_decomp:10.4f}")
print(f"{'2. Dir Ridge + Mag from A':>40}  {r2_norm:10.4f}")
print(f"{'3. Per-Head Dir/Mag Decomposition':>40}  {r2_decomp_head:10.4f}")
print("-" * 70)
print(f"{'   [for reference] Mag R²':>40}  {V_mag_r2:10.4f}")
print(f"{'   [for reference] Dir R²':>40}  {V_dir_r2:10.4f}")
print("=" * 70)