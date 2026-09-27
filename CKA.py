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
def cache_path(name):
    return os.path.join(CACHE_DIR, f"{name}_hs.pt")

@torch.no_grad()
def extract_hidden(model_path, tag):
    path = cache_path(tag)
    if os.path.exists(path):
        print(f"Loading cached {tag}...")
        return torch.load(path)

    print(f"Extracting {tag}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16
    ).to(DEVICE).eval()

    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = []
    for l in range(len(out.hidden_states)):        # 修复：包含最后一层
        h = out.hidden_states[l].reshape(-1, out.hidden_states[l].shape[-1])
        h = h[token_idx].float().cpu()
        hs.append(h)

    del model
    torch.cuda.empty_cache()

    torch.save(hs, path)
    return hs

hs_A = extract_hidden(MODEL_A, "A_0.6B")
hs_B = extract_hidden(MODEL_B, "B_1.7B")

L_A, D_A = len(hs_A), hs_A[0].shape[-1]
L_B, D_B = len(hs_B), hs_B[0].shape[-1]
print(f"0.6B: {L_A} layers, D={D_A}")
print(f"1.7B: {L_B} layers, D={D_B}")

# ============================================================
# 三个指标
# ============================================================
def center_scale(X):
    X = X - X.mean(dim=0, keepdim=True)
    X = X / X.std(dim=0, keepdim=True).clamp_min(1e-6)
    return X

def ridge_r2(X, Y, lam=1.0):
    X = center_scale(X)
    Y = center_scale(Y)
    D = X.shape[1]
    XtX = X.T @ X + lam * torch.eye(D, dtype=X.dtype)
    XtY = X.T @ Y
    W = torch.linalg.solve(XtX, XtY)
    XW = X @ W
    residual = ((XW - Y) ** 2).sum()
    total = (Y ** 2).sum()
    return (1 - residual / total).item()

def cka(X, Y):
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    # Gram 矩阵
    Gx = X @ X.T
    Gy = Y @ Y.T
    # 中心化
    n = Gx.shape[0]
    H = torch.eye(n, dtype=Gx.dtype) - torch.ones(n, n, dtype=Gx.dtype) / n
    Gx = H @ Gx @ H
    Gy = H @ Gy @ H
    num = (Gx * Gy).sum()
    den = torch.sqrt((Gx * Gx).sum() * (Gy * Gy).sum())
    return (num / den).item()

def token_cosine(X, Y):
    # 逐 token 余弦相似度，再平均
    X = X / X.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    Y = Y / Y.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return (X * Y).sum(dim=-1).mean().item()

# ============================================================
# 计算
# ============================================================
print(f"\nComputing {L_A} x {L_B} metrics...")

ridge = np.zeros((L_A, L_B))
cka_mat = np.zeros((L_A, L_B))
cos_mat = np.zeros((L_A, L_B))

for i in range(L_A):
    for j in range(L_B):
        ridge[i, j] = ridge_r2(hs_A[i], hs_B[j])
        cka_mat[i, j] = cka(hs_A[i], hs_B[j])
        # cos_mat[i, j] = token_cosine(hs_A[i], hs_B[j])
        cos_mat[i, j] = 0.0
    if (i + 1) % 7 == 0:
        print(f"  row {i+1}/{L_A}")

np.save("ridge_r2.npy", ridge)
np.save("cka.npy", cka_mat)
np.save("token_cos.npy", cos_mat)

# ============================================================
# 可视化：三个热力图并排
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

titles = ["Ridge R²", "CKA", "Token Cosine"]
mats = [ridge, cka_mat, cos_mat]
cmaps = ["viridis", "magma", "coolwarm"]

for ax, mat, title, cmap in zip(axes, mats, titles, cmaps):
    im = ax.imshow(mat, cmap=cmap, aspect="auto")
    for i in range(min(L_A, L_B)):
        ax.plot(i, i, 'r+', markersize=8, markeredgewidth=1.2)
    ax.set_xlabel("1.7B layer")
    ax.set_ylabel("0.6B layer")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("metrics.png", dpi=150)
print("Saved: metrics.png")

# ============================================================
# 分析
# ============================================================
def analyze(mat, name):
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

analyze(ridge, "Ridge R²")
analyze(cka_mat, "CKA")
analyze(cos_mat, "Token Cosine")

# ============================================================
# 最佳匹配曲线
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))
best_ridge = [int(np.argmax(ridge[i])) for i in range(L_A)]
best_cka = [int(np.argmax(cka_mat[i])) for i in range(L_A)]
best_cos = [int(np.argmax(cos_mat[i])) for i in range(L_A)]

ax.plot(range(L_A), best_ridge, 'o-', label='Ridge R²')
ax.plot(range(L_A), best_cka, 's-', label='CKA')
ax.plot(range(L_A), best_cos, '^-', label='Token Cosine')
ax.plot(range(L_A), range(L_A), 'k--', label='Diagonal')
ax.set_xlabel("0.6B layer")
ax.set_ylabel("Best matching 1.7B layer")
ax.set_title("Best layer match")
ax.legend()
plt.tight_layout()
plt.savefig("best_match.png", dpi=150)
print("Saved: best_match.png")