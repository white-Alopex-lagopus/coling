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
MODEL_A = "../model/Qwen3-0.6B-base"   # 源模型
MODEL_B = "../model/Qwen3-1.7B-base"   # 接收模型
DATA_PATH = "../dataset/wikitext-02/train.jsonl"
CACHE_DIR = "./cache"

NUM_SAMPLES = 50
SEQ_LEN = 512
MAX_TOKENS = 2000
SEED = 42
DEVICE = "cuda"
K_VALUES = [1, 2, 4, 8]   # Part B 的 top-k 候选

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

N_total = NUM_SAMPLES * SEQ_LEN
token_idx = torch.randperm(N_total)[:MAX_TOKENS]

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
    for l in range(len(out.hidden_states)):   # 含最后一层
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
def center_scale(X):
    X = X - X.mean(dim=0, keepdim=True)
    X = X / X.std(dim=0, keepdim=True).clamp_min(1e-6)
    return X

def ridge_r2(X, Y, lam=1.0, device="cuda"):
    X = center_scale(X).to(device)
    Y = center_scale(Y).to(device)
    D = X.shape[1]
    XtX = X.T @ X + lam * torch.eye(D, dtype=X.dtype, device=device)
    XtY = X.T @ Y
    W = torch.linalg.solve(XtX, XtY)
    XW = X @ W
    residual = ((XW - Y) ** 2).sum()
    total = (Y ** 2).sum()
    return (1 - residual / total).item()

def cka(X, Y, device="cuda"):
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
    den = torch.sqrt((Gx * Gx).sum() * (Gy * Gy).sum())
    return (num / den).item()

def attention_output_cosine(Q, K_A, V_A, K_B, V_B,
                             proj_dim=128, seed=42, causal=True, device="cuda"):
    """
    Q: 来自 B 的 query
    K_A, V_A: 来自 A
    K_B, V_B: 来自 B
    全部投影到 proj_dim 后，分别算 attention 输出，最后逐 token cosine。
    """
    Q = Q.to(device); K_A = K_A.to(device); V_A = V_A.to(device)
    K_B = K_B.to(device); V_B = V_B.to(device)
    gen = torch.Generator(device=device).manual_seed(seed)

    def proj(X, d_out):
        d_in = X.shape[1]
        P = torch.randn(d_in, d_out, generator=gen, device=device) / (d_in ** 0.5)
        return X @ P

    Q_p = proj(Q, proj_dim)
    K_A_p = proj(K_A, proj_dim)
    K_B_p = proj(K_B, proj_dim)
    V_A_p = proj(V_A, proj_dim)
    V_B_p = proj(V_B, proj_dim)

    N = Q_p.shape[0]
    scale = proj_dim ** 0.5

    if causal:
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)
        neg_inf = torch.finfo(Q_p.dtype).min
        attn_mask = torch.zeros(N, N, dtype=Q_p.dtype, device=device)
        attn_mask.masked_fill_(mask, neg_inf)
    else:
        attn_mask = torch.zeros(N, N, dtype=Q_p.dtype, device=device)

    logits_A = Q_p @ K_A_p.T / scale + attn_mask
    logits_B = Q_p @ K_B_p.T / scale + attn_mask

    out_A = torch.softmax(logits_A, dim=-1) @ V_A_p
    out_B = torch.softmax(logits_B, dim=-1) @ V_B_p

    out_A = out_A / out_A.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    out_B = out_B / out_B.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (out_A * out_B).sum(dim=-1).mean().item()

# ============================================================
# 预计算 Ridge 和 CKA 矩阵
# ============================================================
print(f"\nComputing {L_A} x {L_B} Ridge R² and CKA...")

ridge_mat = np.zeros((L_A, L_B))
cka_mat = np.zeros((L_A, L_B))

for i in range(L_A):
    for j in range(L_B):
        ridge_mat[i, j] = ridge_r2(hs_A[i], hs_B[j])
        cka_mat[i, j] = cka(hs_A[i], hs_B[j])
    if (i + 1) % 7 == 0:
        print(f"  row {i+1}/{L_A}")

np.save("ridge_r2.npy", ridge_mat)
np.save("cka.npy", cka_mat)

# ============================================================
# Part A：热力图 + 最佳匹配曲线
# ============================================================
print("\n[Part A] 可视化...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
for ax, mat, title, cmap in zip(
    axes, [ridge_mat, cka_mat], ["Ridge R²", "CKA"], ["viridis", "magma"]
):
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    for i in range(min(L_A, L_B)):
        ax.plot(i, i, 'r+', markersize=8, markeredgewidth=1.2)
    ax.set_xlabel("Model B layer")
    ax.set_ylabel("Model A layer")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("partA_heatmaps.png", dpi=150)
print("Saved: partA_heatmaps.png")

fig, ax = plt.subplots(figsize=(8, 8))
for mat, label, marker in [(ridge_mat, "Ridge R²", 'o'),
                           (cka_mat, "CKA", 's')]:
    best = [int(np.argmax(mat[i])) for i in range(L_A)]
    ax.plot(range(L_A), best, marker + '-', label=label)
ax.plot(range(L_A), range(L_A), 'k--', label='Diagonal')
ax.set_xlabel("Model A layer")
ax.set_ylabel("Best matching Model B layer")
ax.set_title("Best layer match")
ax.legend()
plt.tight_layout()
plt.savefig("partA_best_match.png", dpi=150)
print("Saved: partA_best_match.png")

# ============================================================
# Part B：Top-k 源层选择
# ============================================================
print("\n[Part B] Top-k 源层选择...")

def topk_ridge(L_src, L_tgt, hs_src, hs_tgt, cka_mat, k, lam=1.0):
    """
    对每个目标层 i，按 CKA 从源层里选 top-k，拼接后做 Ridge。
    返回每个目标层的 R²。
    """
    r2_list = []
    picks = []
    for i in range(L_tgt):
        order = np.argsort(-cka_mat[:, i])  # 按 CKA 降序
        topk = order[:k]
        X = torch.cat([hs_src[j] for j in topk], dim=1)
        Y = hs_tgt[i]
        r2_list.append(ridge_r2(X, Y, lam=lam))
        picks.append(topk)
    return np.array(r2_list), picks

# 基线 1：只取对角层（j=i）
diag_r2 = np.array([ridge_mat[i, i] for i in range(min(L_A, L_B))])

# 基线 2：只取最佳单层（CKA 最高）
best1_r2, best1_picks = topk_ridge(L_A, L_B, hs_A, hs_B, cka_mat, k=1)

# Top-k
topk_results = {}
for k in K_VALUES:
    topk_results[k], picks = topk_ridge(L_A, L_B, hs_A, hs_B, cka_mat, k=k)
    print(f"  k={k}: mean R² = {topk_results[k].mean():.4f}")

print(f"  diag (k=1, j=i): mean R² = {diag_r2.mean():.4f}")
print(f"  best-1 by CKA  : mean R² = {best1_r2.mean():.4f}")

# 可视化
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(L_B), diag_r2, 'k--', label='Diagonal (j=i)')
ax.plot(range(L_B), best1_r2, 's-', label='Best-1 by CKA')
for k in K_VALUES:
    ax.plot(range(L_B), topk_results[k], 'o-', label=f'Top-{k}')
ax.set_xlabel("Model B layer")
ax.set_ylabel("Ridge R²")
ax.set_title("Part B: Source layer selection")
ax.legend()
plt.tight_layout()
plt.savefig("partB_topk.png", dpi=150)
print("Saved: partB_topk.png")

# ============================================================
# Part C：Attention 输出余弦
# ============================================================
print("\n[Part C] Attention 输出余弦...")

# 计算所有 (i,j) 对的 attention 输出余弦
# i: Model B 的层；j: Model A 的层
# Q 用 Model B 第 i 层，K/V 分别来自 A 第 j 层和 B 第 i 层
attn_cos = np.zeros((L_A, L_B))
for i in range(L_B):
    for j in range(L_A):
        try:
            attn_cos[j, i] = attention_output_cosine(
                Q=hs_B[i], K_A=hs_A[j], V_A=hs_A[j],
                K_B=hs_B[i], V_B=hs_B[i],
                proj_dim=128, seed=42, causal=True
            )
        except Exception as e:
            print(f"  failed ({j},{i}): {e}")
            attn_cos[j, i] = 0.0
    if (i + 1) % 7 == 0:
        print(f"  col {i+1}/{L_B}")

np.save("attn_cos.npy", attn_cos)

# 可视化
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(attn_cos, cmap="coolwarm", aspect="auto", vmin=0, vmax=1)
for i in range(min(L_A, L_B)):
    ax.plot(i, i, 'r+', markersize=8, markeredgewidth=1.2)
ax.set_xlabel("Model B layer")
ax.set_ylabel("Model A layer")
ax.set_title("Attention-output cosine")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("partC_attn_cos.png", dpi=150)
print("Saved: partC_attn_cos.png")

# 与 CKA、Ridge 做相关
def pearson(a, b):
    a = a.flatten(); b = b.flatten()
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return float((a * b).mean())

diag_attn = np.array([attn_cos[i, i] for i in range(min(L_A, L_B))])
diag_cka  = np.array([cka_mat[i, i]  for i in range(min(L_A, L_B))])
diag_ridge = np.array([ridge_mat[i, i] for i in range(min(L_A, L_B))])

print("\nDiagonal correlation:")
print(f"  attn_cos vs CKA   : r = {pearson(diag_attn, diag_cka):+.3f}")
print(f"  attn_cos vs Ridge : r = {pearson(diag_attn, diag_ridge):+.3f}")
print(f"  CKA vs Ridge      : r = {pearson(diag_cka, diag_ridge):+.3f}")

# ============================================================
# Part D：换成不同层数的模型对
# ============================================================
# 想换模型时，只改上面的 MODEL_A / MODEL_B 即可。
# 例如：
# MODEL_A = "../model/Qwen3-0.6B-base"    # 28 层
# MODEL_B = "../model/Qwen3-4B-base"      # 36 层
# 然后重跑整个脚本。
#
# 缓存会自动带上模型名，不会互相覆盖。
# 热力图矩阵形状会自动变成 29 x 37 之类，代码不需要改。
# ============================================================

# ============================================================
# 汇总打印
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
summarize(cka_mat, "CKA")
summarize(attn_cos, "Attention-output cosine")

print("\n[Part B] 汇总")
print(f"  Diagonal (j=i)       : {diag_r2.mean():.4f}")
print(f"  Best-1 by CKA        : {best1_r2.mean():.4f}")
for k in K_VALUES:
    print(f"  Top-{k:<2} by CKA       : {topk_results[k].mean():.4f}")