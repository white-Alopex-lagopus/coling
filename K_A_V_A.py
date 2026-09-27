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

N_ARTICLES = 2000
MAX_TOKENS_PER_ART = 128
LAYER = 14                  # 先用单层快速验证

DEVICE = "cuda"
SEED = 42
RIDGE_BATCH = 50000
RIDGE_LAMBDA = 100.0
TRAIN_RATIO = 0.8

torch.manual_seed(SEED)


# ============================================================
# 数据
# ============================================================
df = pd.read_json(DATA_PATH, lines=True)
texts = df["text"].iloc[:N_ARTICLES].tolist()
tok = AutoTokenizer.from_pretrained(MODEL_A)

ids_list = []
for t in texts:
    ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    ids = ids[:MAX_TOKENS_PER_ART]
    if len(ids) >= 16:
        ids_list.append(ids)

L = max(len(x) for x in ids_list)
padded = torch.zeros(len(ids_list), L, dtype=torch.long)
for i, ids in enumerate(ids_list):
    padded[i, :len(ids)] = ids

B = len(padded)
T = L
print(f"文章数: {B}, 每篇最多 {T} token, 总 token: {B * T}")


# ============================================================
# 工具
# ============================================================
def r2_score(pred, target):
    pred = pred.float()
    target = target.float()
    return (1 - ((pred - target)**2).sum() / (target**2).sum()).item()


def fit_ridge(X, Y, lam=RIDGE_LAMBDA, batch_size=RIDGE_BATCH):
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
# 提取 K/V
# ============================================================
@torch.no_grad()
def extract_kv(model, layer_idx, padded):
    attn = model.model.layers[layer_idx].self_attn
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)

    all_k, all_v = [], []
    batch_size = 64

    for start in range(0, len(padded), batch_size):
        batch = padded[start:start+batch_size].to(DEVICE)
        B_b, T_b = batch.shape

        out = model.model(input_ids=batch, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[layer_idx]
        normed = model.model.layers[layer_idx].input_layernorm(hidden)

        k = attn.k_proj(normed).view(B_b, T_b, n_kv, head_dim)
        k = attn.k_norm(k).transpose(1, 2)
        k = k.transpose(1, 2).reshape(B_b, T_b, -1)
        all_k.append(k.float().cpu())

        v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
        v = v.transpose(1, 2).reshape(B_b, T_b, -1)
        all_v.append(v.float().cpu())

        del out, hidden, normed
        torch.cuda.empty_cache()

    K = torch.cat(all_k, 0).reshape(-1, all_k[0].shape[-1])
    V = torch.cat(all_v, 0).reshape(-1, all_v[0].shape[-1])
    return K, V


# ============================================================
# 加载模型
# ============================================================
print("\n加载模型...")
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()


# ============================================================
# 提取
# ============================================================
print(f"\n提取第 {LAYER} 层 K/V...")
K_A, V_A = extract_kv(model_A, LAYER, padded)
K_B, V_B = extract_kv(model_B, LAYER, padded)
print(f"  K_A: {K_A.shape}, V_A: {V_A.shape}")
print(f"  K_B: {K_B.shape}, V_B: {V_B.shape}")

N = K_A.shape[0]
D = K_A.shape[1]

# 划分 train/test
perm = torch.randperm(N)
n_train = int(N * TRAIN_RATIO)
train_idx = perm[:n_train]
test_idx = perm[n_train:]
print(f"  train: {len(train_idx)}, test: {len(test_idx)}")


# ============================================================
# Baseline：只用 V_A
# ============================================================
print("\n" + "=" * 70)
print(f"K 辅助 V 映射实验（Layer {LAYER}）")
print("=" * 70)

print("\n[Baseline] 只用 V_A...")
W_v_base = fit_ridge(V_A[train_idx], V_B[train_idx])
V_pred_base = V_A[test_idx].float() @ W_v_base
r2_base = r2_score(V_pred_base, V_B[test_idx])
print(f"  V R² = {r2_base:.4f}")


# ============================================================
# 方案 1：拼接 [V_A, K_A] + Ridge
# ============================================================
print("\n[方案 1] 拼接 [V_A, K_A] + Ridge...")
X1_train = torch.cat([V_A[train_idx], K_A[train_idx]], dim=-1)   # [N, 2D]
X1_test = torch.cat([V_A[test_idx], K_A[test_idx]], dim=-1)

W1 = fit_ridge(X1_train, V_B[train_idx])
V_pred1 = X1_test.float() @ W1
r2_1 = r2_score(V_pred1, V_B[test_idx])
print(f"  V R² = {r2_1:.4f}  (vs baseline: {r2_1 - r2_base:+.4f})")


# ============================================================
# 方案 3：加权求和 V_A@W_v + α * K_A@W_k
# ============================================================
print("\n[方案 3] 加权求和...")
W_v = fit_ridge(V_A[train_idx], V_B[train_idx])
W_k = fit_ridge(K_A[train_idx], V_B[train_idx])

V_from_V = V_A[test_idx].float() @ W_v
V_from_K = K_A[test_idx].float() @ W_k

best_alpha = 0.0
best_r2_3 = r2_base
print("  α 搜索：")
for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    V_pred = V_from_V + alpha * V_from_K
    r2_a = r2_score(V_pred, V_B[test_idx])
    marker = ""
    if r2_a > best_r2_3:
        best_r2_3 = r2_a
        best_alpha = alpha
        marker = " ←"
    print(f"    α={alpha:>4.1f}  R² = {r2_a:.4f}{marker}")

print(f"  最优 α = {best_alpha}")
print(f"  V R² = {best_r2_3:.4f}  (vs baseline: {best_r2_3 - r2_base:+.4f})")


# ============================================================
# 方案 5：K 先映射，再拼接
# ============================================================
print("\n[方案 5] K 先映射，再拼接...")
W_K = fit_ridge(K_A[train_idx], K_B[train_idx])
K_A_mapped_train = K_A[train_idx].float() @ W_K
K_A_mapped_test = K_A[test_idx].float() @ W_K

K_mapped_r2 = r2_score(K_A_mapped_test, K_B[test_idx])
print(f"  K 映射 R² = {K_mapped_r2:.4f}")

X5_train = torch.cat([V_A[train_idx], K_A_mapped_train], dim=-1)
X5_test = torch.cat([V_A[test_idx], K_A_mapped_test], dim=-1)

W5 = fit_ridge(X5_train, V_B[train_idx])
V_pred5 = X5_test.float() @ W5
r2_5 = r2_score(V_pred5, V_B[test_idx])
print(f"  V R² = {r2_5:.4f}  (vs baseline: {r2_5 - r2_base:+.4f})")


# ============================================================
# 汇总
# ============================================================
print()
print("=" * 70)
print(f"K 辅助 V 映射结果（Layer {LAYER}）")
print("=" * 70)
print(f"{'方法':>35}  {'V R²':>10}  {'Δ vs Baseline':>15}")
print("-" * 70)
print(f"{'Baseline: V_A only':>35}  {r2_base:>10.4f}  {'—':>15}")
print(f"{'方案 1: concat[V_A, K_A]':>35}  {r2_1:>10.4f}  {r2_1 - r2_base:>+15.4f}")
print(f"{'方案 3: V_A@Wv + α*K_A@Wk':>35}  {best_r2_3:>10.4f}  {best_r2_3 - r2_base:>+15.4f}")
print(f"{'方案 5: concat[V_A, K_A@Wk]':>35}  {r2_5:>10.4f}  {r2_5 - r2_base:>+15.4f}")
print("=" * 70)

print()
print("解读：")
print(f"  方案 3 最优 α = {best_alpha}")
if best_alpha == 0:
    print("  → K 没有辅助作用")
elif best_alpha < 0.5:
    print("  → K 有微弱辅助")
else:
    print("  → K 有明显辅助作用")

print()
print("如果任一方案的 V R² 明显 > 0.55，值得扩展到全层 PPL 验证。")