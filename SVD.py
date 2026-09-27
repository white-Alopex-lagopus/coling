import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"
DATA_PATH = "../dataset/wikitext-02/train.jsonl"
CACHE_DIR = "./cache"

NUM_SAMPLES = 50
SEQ_LEN = 512
MAX_TOKENS = 5000
SEED = 42
DEVICE = "cuda"
R_VALUES = [8, 16, 32, 64, 128, 256]

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
ids = ids[:NUM_SAMPLES * SEQ_LEN].reshape(NUM_SAMPLES, SEQ_LEN).to(DEVICE)

token_idx = torch.randperm(NUM_SAMPLES * SEQ_LEN)[:MAX_TOKENS]

# ============================================================
# 提取 hidden states
# ============================================================
@torch.no_grad()
def extract_hidden(model_path, tag):
    path = os.path.join(CACHE_DIR, f"{tag}_hs.pt")
    if os.path.exists(path):
        print(f"Loading {tag}...")
        return torch.load(path)
    print(f"Extracting {tag}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16
    ).to(DEVICE).eval()
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hs = []
    for l in range(len(out.hidden_states)):
        h = out.hidden_states[l].reshape(-1, out.hidden_states[l].shape[-1])
        hs.append(h[token_idx].float().cpu())
    del model
    torch.cuda.empty_cache()
    torch.save(hs, path)
    return hs

hs_A = extract_hidden(MODEL_A, f"A_{os.path.basename(MODEL_A)}")
hs_B = extract_hidden(MODEL_B, f"B_{os.path.basename(MODEL_B)}")

L_A, D_A = len(hs_A), hs_A[0].shape[-1]
L_B, D_B = len(hs_B), hs_B[0].shape[-1]
print(f"A: {L_A} layers, D={D_A}")
print(f"B: {L_B} layers, D={D_B}")

# ============================================================
# 共享基：联合 SVD
# ============================================================
def find_shared_basis(X_S, X_T, r):
    """
    X_S: [N, d_S]
    X_T: [N, d_T]
    返回 V_S [d_S, r], V_T [d_T, r]
    """
    X_S = X_S - X_S.mean(dim=0, keepdim=True)
    X_T = X_T - X_T.mean(dim=0, keepdim=True)

    X_cat = torch.cat([X_S, X_T], dim=1)         # [N, d_S + d_T]
    U, S, Vh = torch.linalg.svd(X_cat, full_matrices=False)
    V = Vh[:r].T                                  # [(d_S+d_T), r]

    V_S = V[:X_S.shape[1]]
    V_T = V[X_S.shape[1]:]
    return V_S, V_T, S[:r]

def compress_decompress(X_S, X_T, V_S, V_T):
    """
    Sharer 投影 → 传 A → Receiver 重建
    返回重建误差
    """
    X_S_c = X_S - X_S.mean(dim=0, keepdim=True)
    X_T_c = X_T - X_T.mean(dim=0, keepdim=True)

    A = X_S_c @ V_S                                # [N, r]
    X_T_hat = A @ V_T.T + X_T.mean(dim=0, keepdim=True)

    residual = ((X_T_hat - X_T) ** 2).sum()
    total = ((X_T - X_T.mean(dim=0, keepdim=True)) ** 2).sum()
    r2 = (1 - residual / total).item()
    return r2, A

# ============================================================
# 对每对 (i, j) 测试不同 r
# ============================================================
print("\n=== 共享基评估 ===")

results = {}   # results[(i,j)][r] = R²

for i in range(L_A):
    for j in range(L_B):
        # 只在层数接近的地方测试，减少计算量
        if abs(i - j) > 4:
            continue
        X_S = hs_A[i]
        X_T = hs_B[j]
        for r in R_VALUES:
            if r > min(D_A, D_B):
                continue
            V_S, V_T, _ = find_shared_basis(X_S, X_T, r)
            r2, _ = compress_decompress(X_S, X_T, V_S, V_T)
            results.setdefault((i, j), {})[r] = r2

# ============================================================
# 打印对角层结果
# ============================================================
print(f"\n{'Layer':>8}  " + "  ".join(f"r={r:<4}" for r in R_VALUES))
print("-" * (10 + 8 * len(R_VALUES)))

for i in range(min(L_A, L_B)):
    row = f"{i:>8}  "
    for r in R_VALUES:
        val = results.get((i, i), {}).get(r, float("nan"))
        row += f"{val:>6.3f}  "
    print(row)

# ============================================================
# 压缩率 vs 重建质量
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(0, min(L_A, L_B), 4):
    vals = [results.get((i, i), {}).get(r, float("nan")) for r in R_VALUES]
    ax.plot(R_VALUES, vals, 'o-', label=f"Layer {i}")
ax.set_xlabel("Rank r (compression →)")
ax.set_ylabel("Reconstruction R²")
ax.set_title("Shared basis: rank vs reconstruction quality")
ax.legend()
plt.tight_layout()
plt.savefig("shared_basis_rank.png", dpi=150)
print("Saved: shared_basis_rank.png")

# ============================================================
# 关键指标：达到 95% R² 需要多大 r
# ============================================================
print("\n=== 达到 95% R² 所需 rank ===")
for i in range(min(L_A, L_B)):
    for r in R_VALUES:
        val = results.get((i, i), {}).get(r, float("nan"))
        if val >= 0.95:
            print(f"  Layer {i}: r = {r}  (R² = {val:.3f})")
            break