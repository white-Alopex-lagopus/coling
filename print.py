import numpy as np

# ============================================================
# 读取
# ============================================================
ridge = np.load("ridge_r2.npy")
cka   = np.load("cka.npy")
try:
    attn = np.load("attn_cos.npy")
    has_attn = True
except FileNotFoundError:
    has_attn = False
    print("attn_cos.npy not found, skipping.\n")

L_A, L_B = ridge.shape
print(f"Shape: Model A = {L_A} layers, Model B = {L_B} layers\n")


# ============================================================
# 1. 整体统计
# ============================================================
def summarize(mat, name):
    diag = np.array([mat[i, i] for i in range(min(L_A, L_B))])
    mask = ~np.eye(L_A, L_B, dtype=bool)
    off = mat[mask]

    print(f"{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"  Diagonal mean  : {diag.mean():.4f}")
    print(f"  Diagonal min   : {diag.min():.4f}  (layer {diag.argmin()})")
    print(f"  Diagonal max   : {diag.max():.4f}  (layer {diag.argmax()})")
    print(f"  Off-diag mean  : {off.mean():.4f}")
    print(f"  Off-diag max   : {off.max():.4f}")
    print(f"  Diag - offdiag : {diag.mean() - off.mean():+.4f}")
    print()

summarize(ridge, "Ridge R²")
summarize(cka, "CKA")
if has_attn:
    summarize(attn, "Attention-output cosine")


# ============================================================
# 2. 每一层的最佳匹配（逐行）
# ============================================================
def best_match_table(mat, name):
    print(f"{'='*60}")
    print(f"{name}: best match per Model A layer")
    print(f"{'='*60}")
    print(f"{'A layer':>8}  {'best B':>8}  {'value':>8}  {'delta':>6}  {'diag value':>10}")
    print("-" * 50)

    deltas = []
    for i in range(L_A):
        j = int(np.argmax(mat[i]))
        val = mat[i, j]
        diag_val = mat[i, i] if i < L_B else float("nan")
        delta = j - i
        deltas.append(abs(delta))
        print(f"{i:>8}  {j:>8}  {val:>8.4f}  {delta:>+6}  {diag_val:>10.4f}")

    print(f"\n  On-diagonal best: {sum(1 for i in range(L_A) if int(np.argmax(mat[i])) == i)}/{L_A}")
    print(f"  Mean |delta|    : {np.mean(deltas):.2f}")
    print(f"  Median |delta|  : {np.median(deltas):.2f}")
    print()

best_match_table(ridge, "Ridge R²")
best_match_table(cka, "CKA")
if has_attn:
    best_match_table(attn, "Attention-output cosine")


# ============================================================
# 3. 每一层的 Top-5 匹配
# ============================================================
def top5_table(mat, name):
    print(f"{'='*60}")
    print(f"{name}: Top-5 matches per Model A layer")
    print(f"{'='*60}")
    for i in range(L_A):
        order = np.argsort(-mat[i])[:5]
        vals = mat[i, order]
        pairs = "  ".join(f"B{j}({v:.3f})" for j, v in zip(order, vals))
        print(f"  A{i:>2}: {pairs}")
    print()

top5_table(cka, "CKA")
top5_table(ridge, "Ridge R²")


# ============================================================
# 4. 逐层对角线 vs 非对角线最大值
# ============================================================
def diag_vs_offdiag(mat, name):
    print(f"{'='*60}")
    print(f"{name}: diag vs best off-diag per layer")
    print(f"{'='*60}")
    print(f"{'A layer':>8}  {'diag':>8}  {'best off':>10}  {'off layer':>10}  {'diag - best_off':>16}")
    print("-" * 60)
    wins = 0
    for i in range(min(L_A, L_B)):
        diag_val = mat[i, i]
        row = mat[i].copy()
        row[i] = -np.inf
        j = int(np.argmax(row))
        off_val = row[j]
        if diag_val >= off_val:
            wins += 1
        print(f"{i:>8}  {diag_val:>8.4f}  {off_val:>10.4f}  {j:>10}  {diag_val - off_val:>+16.4f}")
    print(f"\n  Diagonal wins: {wins}/{min(L_A, L_B)}")
    print()

diag_vs_offdiag(ridge, "Ridge R²")
diag_vs_offdiag(cka, "CKA")