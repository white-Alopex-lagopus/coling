import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置
# ============================================================
MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"
DATA_PATH = "../dataset/wikitext-02/train.jsonl"
CACHE_DIR = "./cache"

NUM_SAMPLES = 50
SEQ_LEN = 512
MAX_TOKENS = 2000
SEED = 42
DEVICE = "cuda"
K_VALUES = [1, 2, 4, 8]
TRAIN_RATIO = 0.75

os.makedirs(CACHE_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# 数据
# ============================================================
df = pd.read_json(DATA_PATH, lines=True)
text = df["text"].str.cat(sep="\n")
tok = AutoTokenizer.from_pretrained(MODEL_A)
ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
need = NUM_SAMPLES * SEQ_LEN
ids = ids[:need].reshape(NUM_SAMPLES, SEQ_LEN).to(DEVICE)

token_idx = torch.randperm(NUM_SAMPLES * SEQ_LEN)[:MAX_TOKENS]

# ============================================================
# 提取 hidden states（带缓存）
# ============================================================
def cache_path(tag, suffix):
    return os.path.join(CACHE_DIR, f"{tag}_{suffix}.pt")

@torch.no_grad()
def extract_hidden(model_path, tag):
    path = cache_path(tag, "hs")
    if os.path.exists(path):
        print(f"Loading cached {tag}...")
        return torch.load(path)

    print(f"Extracting {tag}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16
    ).to(DEVICE).eval()

    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = []
    for l in range(len(out.hidden_states)):
        h = out.hidden_states[l].reshape(-1, out.hidden_states[l].shape[-1])
        h = h[token_idx].float().cpu()
        hs.append(h)

    del model
    torch.cuda.empty_cache()
    torch.save(hs, path)
    return hs

hs_A = extract_hidden(MODEL_A, f"A_{os.path.basename(MODEL_A)}")
hs_B = extract_hidden(MODEL_B, f"B_{os.path.basename(MODEL_B)}")

L_A, D_A = len(hs_A), hs_A[0].shape[-1]
L_B, D_B = len(hs_B), hs_B[0].shape[-1]
print(f"Model A: {L_A} layers, D={D_A}")
print(f"Model B: {L_B} layers, D={D_B}")

# ============================================================
# 训练/测试划分
# ============================================================
N = hs_A[0].shape[0]
gen = torch.Generator().manual_seed(SEED)
perm = torch.randperm(N, generator=gen)
n_train = int(N * TRAIN_RATIO)
train_idx = perm[:n_train]
test_idx  = perm[n_train:]
print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

# ============================================================
# 基础工具
# ============================================================
def standardize_fit(X):
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (X - mean) / std, mean, std

def standardize_apply(X, mean, std):
    return (X - mean) / std

def fit_and_eval(X_train, Y_train, X_test, Y_test, lam=1.0):
    """在训练集拟合 Ridge，在测试集评估 R²。"""
    X_train_n, mX, sX = standardize_fit(X_train)
    Y_train_n, mY, sY = standardize_fit(Y_train)
    X_train_n = X_train_n.to(DEVICE)
    Y_train_n = Y_train_n.to(DEVICE)

    D = X_train_n.shape[1]
    XtX = X_train_n.T @ X_train_n + lam * torch.eye(D, dtype=X_train_n.dtype, device=DEVICE)
    XtY = X_train_n.T @ Y_train_n
    W = torch.linalg.solve(XtX, XtY)

    X_test_n = standardize_apply(X_test, mX, sX).to(DEVICE)
    Y_test_n = standardize_apply(Y_test, mY, sY).to(DEVICE)
    XW = X_test_n @ W
    residual = ((XW - Y_test_n) ** 2).sum()
    total = (Y_test_n ** 2).sum()
    return (1 - residual / total).item()

# ============================================================
# CKA（保留，虽然饱和，但比 Procrustes 强）
# ============================================================
def cka(X, Y, device=DEVICE):
    X = X.to(device)
    Y = Y.to(device)
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    Gx = X @ X.T
    Gy = Y @ Y.T
    n = Gx.shape[0]
    H = torch.eye(n, dtype=Gx.dtype, device=device) - torch.ones(n, n, dtype=Gx.dtype, device=device) / n
    Gx = H @ Gx @ H
    Gy = H @ Gy @ H
    num = (Gx * Gy).sum()
    den = torch.sqrt((Gx * Gx).sum() * (Gy * Gy).sum()).clamp_min(1e-12)
    return (num / den).item()

# ============================================================
# 计算矩阵（注意：矩阵也用训练集拟合、测试集评估）
# ============================================================
print(f"\nComputing {L_A} x {L_B} Ridge R² (train-fit, test-eval) and CKA...")

ridge_mat = np.zeros((L_A, L_B))
cka_mat = np.zeros((L_A, L_B))

for i in range(L_A):
    for j in range(L_B):
        # Ridge：训练集拟合，测试集评估
        ridge_mat[i, j] = fit_and_eval(
            hs_A[i][train_idx], hs_B[j][train_idx],
            hs_A[i][test_idx],  hs_B[j][test_idx]
        )
        # CKA：在全部 token 上算（CKA 不涉及拟合，不存在泄露）
        cka_mat[i, j] = cka(hs_A[i], hs_B[j])
    if (i + 1) % 7 == 0:
        print(f"  row {i+1}/{L_A}")

np.save("ridge_r2.npy", ridge_mat)
np.save("cka.npy", cka_mat)

# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for ax, mat, title, cmap in zip(
    axes, [ridge_mat, cka_mat], ["Ridge R² (test)", "CKA"], ["viridis", "magma"]
):
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    for i in range(min(L_A, L_B)):
        ax.plot(i, i, 'r+', markersize=8, markeredgewidth=1.2)
    ax.set_xlabel("Model B layer")
    ax.set_ylabel("Model A layer")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("metrics.png", dpi=150)
print("Saved: metrics.png")

# ============================================================
# Top-k 源层选择（训练集选源层，测试集评估）
# ============================================================
print("\n[Top-k] 源层选择...")

def topk_eval(ref_mat, k, hs_src, hs_tgt, train_idx, test_idx):
    """
    对每个目标层 i：
      1. 按 ref_mat 列选 top-k 源层
      2. 拼接后训练集拟合 Ridge
      3. 测试集评估 R²
    返回每个目标层的 test R²。
    """
    r2_list = []
    for i in range(L_B):
        order = np.argsort(-ref_mat[:, i])
        topk = order[:k]
        X = torch.cat([hs_src[j] for j in topk], dim=1)
        Y = hs_tgt[i]
        r2 = fit_and_eval(
            X[train_idx], Y[train_idx],
            X[test_idx],  Y[test_idx]
        )
        r2_list.append(r2)
    return np.array(r2_list)

# 基线：终端对齐 = k=1 且选 j=i
diag_r2 = np.array([ridge_mat[i, i] for i in range(min(L_A, L_B))])

# Top-k 用 Ridge R² 矩阵选源层
topk_ridge_results = {}
for k in K_VALUES:
    topk_ridge_results[k] = topk_eval(ridge_mat, k, hs_A, hs_B, train_idx, test_idx)
    print(f"  Top-{k:<2} by Ridge R²: mean test R² = {topk_ridge_results[k].mean():.4f}")

# Top-k 用 CKA 矩阵选源层
topk_cka_results = {}
for k in K_VALUES:
    topk_cka_results[k] = topk_eval(cka_mat, k, hs_A, hs_B, train_idx, test_idx)
    print(f"  Top-{k:<2} by CKA    : mean test R² = {topk_cka_results[k].mean():.4f}")

print(f"  Diagonal (j=i)       : mean test R² = {diag_r2.mean():.4f}")

# 可视化
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(L_B), diag_r2, 'k--', linewidth=2, label='Diagonal (j=i)')
for k in K_VALUES:
    ax.plot(range(L_B), topk_ridge_results[k], 'o-', label=f'Top-{k} by Ridge')
    ax.plot(range(L_B), topk_cka_results[k],   's--', label=f'Top-{k} by CKA')
ax.set_xlabel("Model B layer")
ax.set_ylabel("Test R²")
ax.set_title("Top-k source layer selection (test-set R²)")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig("topk.png", dpi=150)
print("Saved: topk.png")

# ============================================================
# 打印统计
# ============================================================
def summarize(mat, name):
    diag = np.array([mat[i, i] for i in range(min(L_A, L_B))])
    mask = ~np.eye(L_A, L_B, dtype=bool)
    off = mat[mask]
    best = [int(np.argmax(mat[i])) for i in range(L_A)]
    on_diag = sum(1 for i, j in enumerate(best) if i == j)
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Diagonal mean   : {diag.mean():.4f}")
    print(f"Off-diag mean   : {off.mean():.4f}")
    print(f"Diag - offdiag  : {diag.mean() - off.mean():+.4f}")
    print(f"Best match on diag: {on_diag}/{L_A}")
    print(f"Mean |best_j - i| : {np.mean([abs(j-i) for i,j in enumerate(best)]):.2f}")

summarize(ridge_mat, "Ridge R² (test)")
summarize(cka_mat, "CKA")

# ============================================================
# 加载 attention cosine（如果存在）
# ============================================================
if os.path.exists("attn_cos_real.npy"):
    attn_real = np.load("attn_cos_real.npy")
    if attn_real.shape == (L_A, L_B):
        print(f"\n{'='*60}")
        print("Attention-output cosine (real KV)")
        print(f"{'='*60}")
        summarize(attn_real, "Attention cosine (real)")
        diag_attn = np.array([attn_real[i, i] for i in range(min(L_A, L_B))])
        diag_ridge = np.array([ridge_mat[i, i] for i in range(min(L_A, L_B))])
        diag_cka   = np.array([cka_mat[i, i] for i in range(min(L_A, L_B))])
        def pearson(a, b):
            a = a - a.mean(); b = b - b.mean()
            return float((a * b).mean() / (a.std() * b.std() + 1e-12))
        print(f"\nDiagonal correlation:")
        print(f"  attn vs Ridge : r = {pearson(diag_attn, diag_ridge):+.3f}")
        print(f"  attn vs CKA   : r = {pearson(diag_attn, diag_cka):+.3f}")
    else:
        print(f"\nattn_cos_real.npy shape {attn_real.shape} != ({L_A},{L_B}), skip.")
else:
    print("\nattn_cos_real.npy not found, skip attention cosine.")