import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置
# ============================================================
MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"
DATA_PATH = "../dataset/wikitext-02/train.jsonl"

MAX_ARTICLES = 500
MAX_TOKENS_PER_ART = 256
MAX_OFFSET = 3          # 只算 |i - j| <= 3
DEVICE = "cuda"

torch.manual_seed(42)

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

print(f"用了 {len(article_ids)} 篇文章")

max_len = max(len(x) for x in article_ids)
padded = torch.zeros(len(article_ids), max_len, dtype=torch.long)
for i, ids in enumerate(article_ids):
    padded[i, :len(ids)] = ids

print(f"输入: {tuple(padded.shape)}")

# ============================================================
# 提取 hidden
# ============================================================
@torch.no_grad()
def extract(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16
    ).to(DEVICE).eval()

    base = model.model          # 关键：跳过 lm_head
    n_layers = model.config.num_hidden_layers
    all_hs = [[] for _ in range(n_layers)]

    batch_size = 64             # 更保守
    for start in range(0, len(padded), batch_size):
        batch = padded[start:start+batch_size].to(DEVICE)
        out = base(input_ids=batch, output_hidden_states=True, use_cache=False)

        for l in range(n_layers):
            h = out.hidden_states[l].reshape(-1, out.hidden_states[l].shape[-1])
            # all_hs[l].append(h.float().cpu())
            all_hs[l].append(h.to(torch.float16).cpu())

    for l in range(n_layers):
        all_hs[l] = torch.cat(all_hs[l], dim=0)

    del model, base
    torch.cuda.empty_cache()
    return all_hs

print("\n提取 0.6B...")
hs_A = extract(MODEL_A)
print("提取 1.7B...")
hs_B = extract(MODEL_B)

L_A, D_A = len(hs_A), hs_A[0].shape[-1]
L_B, D_B = len(hs_B), hs_B[0].shape[-1]
N = hs_A[0].shape[0]
print(f"0.6B: {L_A} layers, D={D_A}, N={N}")
print(f"1.7B: {L_B} layers, D={D_B}, N={N}")

# ============================================================
# Ridge CV R²
# ============================================================
def ridge_cv(X, Y, lam=100.0, n_folds=5):
    N = X.shape[0]
    fold_size = N // n_folds
    perm = torch.randperm(N)
    X, Y = X[perm], Y[perm]

    r2_list = []
    for k in range(n_folds):
        te_s, te_e = k * fold_size, (k + 1) * fold_size
        X_te, Y_te = X[te_s:te_e], Y[te_s:te_e]
        X_tr = torch.cat([X[:te_s], X[te_e:]], dim=0)
        Y_tr = torch.cat([Y[:te_s], Y[te_e:]], dim=0)

        X_m, X_s = X_tr.mean(0, keepdim=True), X_tr.std(0, keepdim=True).clamp_min(1e-6)
        Y_m, Y_s = Y_tr.mean(0, keepdim=True), Y_tr.std(0, keepdim=True).clamp_min(1e-6)

        X_tr = (X_tr - X_m) / X_s
        Y_tr = (Y_tr - Y_m) / Y_s
        X_te = (X_te - X_m) / X_s
        Y_te = (Y_te - Y_m) / Y_s

        D = X_tr.shape[1]
        XtX = X_tr.T @ X_tr + lam * torch.eye(D, dtype=X_tr.dtype)
        W = torch.linalg.solve(XtX, X_tr.T @ Y_tr)

        XW = X_te @ W
        r2 = (1 - ((XW - Y_te) ** 2).sum() / (Y_te ** 2).sum()).item()
        r2_list.append(r2)

    return float(np.mean(r2_list))

# ============================================================
# 只算对角线附近
# ============================================================
print(f"\n计算 Ridge CV R²（只算 |i-j| <= {MAX_OFFSET}）...")

ridge = np.full((L_A, L_B), np.nan)

n_pairs = 0
for i in range(L_A):
    for j in range(max(0, i - MAX_OFFSET), min(L_B, i + MAX_OFFSET + 1)):
        ridge[i, j] = ridge_cv(hs_A[i], hs_B[j], lam=100.0)
        n_pairs += 1
    if (i + 1) % 7 == 0:
        print(f"  row {i+1}/{L_A}  (cumulative pairs: {n_pairs})")

np.save("ridge_cv_diagband.npy", ridge)

# ============================================================
# 统计
# ============================================================
diag = np.array([ridge[i, i] for i in range(min(L_A, L_B)) if not np.isnan(ridge[i, i])])

# off-diagonal 只在计算过的范围内
off_diag_vals = []
for i in range(L_A):
    for j in range(max(0, i - MAX_OFFSET), min(L_B, i + MAX_OFFSET + 1)):
        if i != j and not np.isnan(ridge[i, j]):
            off_diag_vals.append(ridge[i, j])
off_diag_vals = np.array(off_diag_vals)

print()
print("=" * 70)
print(f"Ridge CV R² (diagonal band ±{MAX_OFFSET})")
print("=" * 70)
print(f"Diagonal mean   : {diag.mean():.4f}")
print(f"Off-diag mean   : {off_diag_vals.mean():.4f}")
print(f"Diag - offdiag  : {diag.mean() - off_diag_vals.mean():+.4f}")

# 每个偏移量的平均
print()
print("每层偏移的平均 R²：")
for offset in range(-MAX_OFFSET, MAX_OFFSET + 1):
    vals = []
    for i in range(L_A):
        j = i + offset
        if 0 <= j < L_B and not np.isnan(ridge[i, j]):
            vals.append(ridge[i, j])
    if vals:
        print(f"  offset {offset:+d}: mean = {np.mean(vals):.4f}  (n={len(vals)})")

# 每层的 argmax
print()
print(f"{'0.6B':>5}  {'best 1.7B':>10}  {'R²':>8}  {'Δ':>5}")
print("-" * 40)
for i in range(L_A):
    row = ridge[i]
    valid = ~np.isnan(row)
    if valid.any():
        j_valid = np.where(valid)[0]
        j = j_valid[np.argmax(row[valid])]
        print(f"{i:5d}  {j:10d}  {row[j]:8.4f}  {j-i:+5d}")

# ============================================================
# 可视化
# ============================================================
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 9))
masked = np.ma.masked_invalid(ridge)
im = ax.imshow(masked, cmap="viridis", aspect="auto", vmin=0, vmax=1)

# 对角线
for i in range(min(L_A, L_B)):
    ax.plot(i, i, 'r+', markersize=10, markeredgewidth=1.5)

ax.set_xlabel("1.7B layer")
ax.set_ylabel("0.6B layer")
ax.set_title(f"Ridge CV R² (band ±{MAX_OFFSET})")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("ridge_cv_diagband.png", dpi=150)
print("\nSaved: ridge_cv_diagband.png")