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
PROJ_DIM = 256   # Procrustes 随机投影维度

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
# 提取 hidden states（带缓存，复用你已有的）
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
# 指标函数
# ============================================================
def center(X):
    return X - X.mean(dim=0, keepdim=True)

def scale(X):
    return X / X.std(dim=0, keepdim=True).clamp_min(1e-6)

def ridge_r2(X, Y, lam=1.0, device="cuda"):
    X = scale(center(X)).to(device)
    Y = scale(center(Y)).to(device)
    D = X.shape[1]
    XtX = X.T @ X + lam * torch.eye(D, dtype=X.dtype, device=device)
    XtY = X.T @ Y
    W = torch.linalg.solve(XtX, XtY)
    XW = X @ W
    residual = ((XW - Y) ** 2).sum()
    total = (Y ** 2).sum()
    return (1 - residual / total).item()

def procrustes_sim(X, Y, dim=PROJ_DIM, seed=42):
    """
    X: [N, D_X], Y: [N, D_Y]
    先用随机高斯投影压到 dim 维，再算归一化内积（RV 系数）。
    用固定 seed 保证每一对 (i,j) 用的是同一投影。
    """
    gen = torch.Generator().manual_seed(seed)
    if X.shape[1] != dim:
        Px = torch.randn(X.shape[1], dim, generator=gen) / (X.shape[1] ** 0.5)
        X = X @ Px
    if Y.shape[1] != dim:
        Py = torch.randn(Y.shape[1], dim, generator=gen) / (Y.shape[1] ** 0.5)
        Y = Y @ Py
    X = center(X)
    Y = center(Y)
    num = (X * Y).sum()
    den = torch.sqrt((X * X).sum() * (Y * Y).sum()).clamp_min(1e-12)
    return (num / den).item()

# ============================================================
# 计算 Ridge 和 Procrustes 矩阵
# ============================================================
print(f"\nComputing {L_A} x {L_B} Ridge R² and Procrustes...")

ridge_mat = np.zeros((L_A, L_B))
proc_mat = np.zeros((L_A, L_B))

for i in range(L_A):
    for j in range(L_B):
        ridge_mat[i, j] = ridge_r2(hs_A[i], hs_B[j])
        proc_mat[i, j] = procrustes_sim(hs_A[i], hs_B[j])
    if (i + 1) % 7 == 0:
        print(f"  row {i+1}/{L_A}")

np.save("ridge_r2.npy", ridge_mat)
np.save("procrustes.npy", proc_mat)

# ============================================================
# 可视化：Ridge 和 Procrustes 热力图
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for ax, mat, title, cmap in zip(
    axes, [ridge_mat, proc_mat], ["Ridge R²", "Procrustes sim"], ["viridis", "magma"]
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
# 最佳匹配曲线
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))
for mat, label, marker in [(ridge_mat, "Ridge R²", 'o'),
                           (proc_mat, "Procrustes", 's')]:
    best = [int(np.argmax(mat[i])) for i in range(L_A)]
    ax.plot(range(L_A), best, marker + '-', label=label)
ax.plot(range(L_A), range(L_A), 'k--', label='Diagonal')
ax.set_xlabel("Model A layer")
ax.set_ylabel("Best matching Model B layer")
ax.set_title("Best layer match")
ax.legend()
plt.tight_layout()
plt.savefig("best_match.png", dpi=150)
print("Saved: best_match.png")

# ============================================================
# Top-k 源层选择
# ============================================================
print("\n[Top-k] 源层选择...")

def topk_ridge(L_src, L_tgt, hs_src, hs_tgt, ref_mat, k, lam=1.0):
    """
    对每个目标层 i，按 ref_mat 列选 top-k 源层，拼接后做 Ridge。
    ref_mat: [L_src, L_tgt]，用 proc_mat 或 ridge_mat 均可。
    返回每个目标层的 R²。
    """
    r2_list = []
    picks = []
    for i in range(L_tgt):
        order = np.argsort(-ref_mat[:, i])
        topk = order[:k]
        X = torch.cat([hs_src[j] for j in topk], dim=1)
        Y = hs_tgt[i]
        r2_list.append(ridge_r2(X, Y, lam=lam))
        picks.append(topk)
    return np.array(r2_list), picks

# 基线 1：终端对齐
diag_r2 = np.array([ridge_mat[i, i] for i in range(min(L_A, L_B))])

# 基线 2：Procrustes 最佳单层
best1_r2, _ = topk_ridge(L_A, L_B, hs_A, hs_B, proc_mat, k=1)

# Top-k
topk_results = {}
for k in K_VALUES:
    r2, _ = topk_ridge(L_A, L_B, hs_A, hs_B, proc_mat, k=k)
    topk_results[k] = r2
    print(f"  k={k:<2} (by Procrustes): mean R² = {r2.mean():.4f}")

print(f"  diagonal (j=i)          : mean R² = {diag_r2.mean():.4f}")
print(f"  best-1 by Procrustes    : mean R² = {best1_r2.mean():.4f}")

# 可视化
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(L_B), diag_r2, 'k--', label='Diagonal (j=i)')
ax.plot(range(L_B), best1_r2, 's-', label='Best-1 by Procrustes')
for k in K_VALUES:
    ax.plot(range(L_B), topk_results[k], 'o-', label=f'Top-{k}')
ax.set_xlabel("Model B layer")
ax.set_ylabel("Ridge R²")
ax.set_title("Top-k source layer selection")
ax.legend()
plt.tight_layout()
plt.savefig("topk.png", dpi=150)
print("Saved: topk.png")

# ============================================================
# 打印详细统计
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

summarize(ridge_mat, "Ridge R²")
summarize(proc_mat, "Procrustes sim")

# ============================================================
# 逐层最佳匹配（Procrustes）
# ============================================================
print(f"\n{'='*60}")
print("Procrustes: best match per Model A layer")
print(f"{'='*60}")
print(f"{'A layer':>8}  {'best B':>8}  {'value':>8}  {'delta':>6}  {'diag value':>10}")
print("-" * 50)
deltas = []
for i in range(L_A):
    j = int(np.argmax(proc_mat[i]))
    val = proc_mat[i, j]
    diag_val = proc_mat[i, i] if i < L_B else float("nan")
    delta = j - i
    deltas.append(abs(delta))
    print(f"{i:>8}  {j:>8}  {val:>8.4f}  {delta:>+6}  {diag_val:>10.4f}")

print(f"\n  On-diagonal best: {sum(1 for i in range(L_A) if int(np.argmax(proc_mat[i])) == i)}/{L_A}")
print(f"  Mean |delta|    : {np.mean(deltas):.2f}")
print(f"  Median |delta|  : {np.median(deltas):.2f}")

# ============================================================
# 加载真实 attention-output cosine（如果存在）
# ============================================================
if os.path.exists("attn_cos_real.npy"):
    attn_real = np.load("attn_cos_real.npy")
    if attn_real.shape == (L_A, L_B):
        print(f"\n{'='*60}")
        print("Attention-output cosine (real KV)")
        print(f"{'='*60}")
        summarize(attn_real, "Attention cosine (real)")
        # 相关性
        diag_attn = np.array([attn_real[i, i] for i in range(min(L_A, L_B))])
        diag_proc = np.array([proc_mat[i, i] for i in range(min(L_A, L_B))])
        diag_ridge = np.array([ridge_mat[i, i] for i in range(min(L_A, L_B))])
        def pearson(a, b):
            a = a - a.mean(); b = b - b.mean()
            return float((a * b).mean() / (a.std() * b.std() + 1e-12))
        print(f"\nDiagonal correlation:")
        print(f"  attn vs Procrustes : r = {pearson(diag_attn, diag_proc):+.3f}")
        print(f"  attn vs Ridge      : r = {pearson(diag_attn, diag_ridge):+.3f}")
    else:
        print(f"\nattn_cos_real.npy shape {attn_real.shape} != ({L_A},{L_B}), skip.")
else:
    print("\nattn_cos_real.npy not found, skip attention cosine.")