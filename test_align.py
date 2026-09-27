import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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
DATA_PATH = "../dataset/wikitext-02/kvtest.jsonl"

NUM_SAMPLES = 50
SEQ_LEN = 512
MAX_TOKENS = 2000
SEED = 42

DEVICE = "cuda"
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
# 提取 hidden states
# ============================================================
@torch.no_grad()
def extract_hidden(model_path, token_idx):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16
    ).to(DEVICE).eval()
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    n_layers = model.config.num_hidden_layers
    hs = []
    for l in range(n_layers):
        h = out.hidden_states[l].reshape(-1, out.hidden_states[l].shape[-1])
        h = h[token_idx].float().cpu()
        hs.append(h)
    del model
    torch.cuda.empty_cache()
    return hs

print("Extracting 0.6B...")
hs_A = extract_hidden(MODEL_A, token_idx)
print("Extracting 1.7B...")
hs_B = extract_hidden(MODEL_B, token_idx)

L_A, D_A = len(hs_A), hs_A[0].shape[-1]
L_B, D_B = len(hs_B), hs_B[0].shape[-1]
print(f"0.6B: {L_A} layers, D={D_A}")
print(f"1.7B: {L_B} layers, D={D_B}")

# ============================================================
# 全局均值（去各向异性用）
# ============================================================
global_mean_A = torch.cat(hs_A, dim=0).mean(dim=0, keepdim=True)
global_mean_B = torch.cat(hs_B, dim=0).mean(dim=0, keepdim=True)

# Ridge 回归 R²
# ============================================================
def ridge_r2(X, Y, lam=1.0):
    """
    X: [N, D_X]
    Y: [N, D_Y]
    解 XW ≈ Y，返回 R²
    """
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # 标准化尺度，避免数值问题
    X = X / X.std(dim=0, keepdim=True).clamp_min(1e-6)
    Y = Y / Y.std(dim=0, keepdim=True).clamp_min(1e-6)

    D = X.shape[1]
    XtX = X.T @ X + lam * torch.eye(D, dtype=X.dtype)
    XtY = X.T @ Y
    W = torch.linalg.solve(XtX, XtY)

    XW = X @ W
    residual = ((XW - Y) ** 2).sum()
    total = (Y ** 2).sum()

    return (1 - residual / total).item()


# ============================================================
# 计算
# ============================================================
print(f"\nComputing {L_A} x {L_B} Ridge R²...")

ridge = np.zeros((L_A, L_B))

for i in range(L_A):
    for j in range(L_B):
        ridge[i, j] = ridge_r2(hs_A[i], hs_B[j])
    if (i + 1) % 7 == 0:
        print(f"  row {i+1}/{L_A}")

np.save("ridge_r2.npy", ridge)

# ============================================================
# 可视化
# ============================================================
fig, ax = plt.subplots(figsize=(10, 9))
im = ax.imshow(ridge, cmap="viridis", aspect="auto", vmin=0, vmax=1)
for i in range(min(L_A, L_B)):
    ax.plot(i, i, 'r+', markersize=10, markeredgewidth=1.5)
ax.set_xlabel("1.7B layer")
ax.set_ylabel("0.6B layer")
ax.set_title("Ridge regression R² (linear map)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("ridge_r2.png", dpi=150)
print("Saved: ridge_r2.png")

# ============================================================
# 统计
# ============================================================
diag = np.array([ridge[i, i] for i in range(min(L_A, L_B))])
mask = ~np.eye(L_A, L_B, dtype=bool)
off = ridge[mask]

print()
print("=" * 70)
print("Ridge R² statistics")
print("=" * 70)
print(f"Diagonal mean   : {diag.mean():.4f}")
print(f"Off-diag mean   : {off.mean():.4f}")
print(f"Diag - offdiag  : {diag.mean() - off.mean():+.4f}")

best = [int(np.argmax(ridge[i])) for i in range(L_A)]
on_diag = sum(1 for i, j in enumerate(best) if i == j)
print(f"Best match on diag: {on_diag}/{L_A}")
print(f"Mean |best_j - i| : {np.mean([abs(j-i) for i,j in enumerate(best)]):.2f}")

print()
print("Best 1.7B match (Ridge R²):")
print(f"{'0.6B':>5}  {'best 1.7B':>10}  {'R²':>8}  {'Δ':>5}")
print("-" * 40)
for i in range(L_A):
    j = int(np.argmax(ridge[i]))
    print(f"{i:5d}  {j:10d}  {ridge[i, j]:8.4f}  {j-i:+5d}")