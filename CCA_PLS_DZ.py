import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置
# ============================================================
MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"
TRAIN_PATH = "../dataset/wikitext-02/train.jsonl"
TEST_PATH  = "../dataset/wikitext-02/test.jsonl"

MAX_ARTICLES_TRAIN = 2000
MAX_ARTICLES_TEST  = 1400
MAX_TOKENS_PER_ART = 256
LAYERS_TO_TEST = [0, 7, 14, 21, 27]
R_VALUES = [16, 32, 64, 128, 256]

DEVICE = "cuda"
SEED = 42
BATCH_SIZE = 32
COV_BATCH = 50000

torch.manual_seed(SEED)

# ============================================================
# 数据准备
# ============================================================
def prepare(path, max_articles, max_tokens):
    df = pd.read_json(path, lines=True)
    texts = df["text"].iloc[:max_articles].tolist()
    tok = AutoTokenizer.from_pretrained(MODEL_A)
    article_ids = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        ids = ids[:max_tokens]
        if len(ids) >= 16:
            article_ids.append(ids)
    max_len = max(len(x) for x in article_ids)
    padded = torch.zeros(len(article_ids), max_len, dtype=torch.long)
    mask = torch.zeros(len(article_ids), max_len, dtype=torch.bool)
    for i, ids in enumerate(article_ids):
        padded[i, :len(ids)] = ids
        mask[i, :len(ids)] = True
    print(f"  {path}: {len(article_ids)} 篇, max_len={max_len}, 有效 token={mask.sum().item()}")
    return padded, mask

print("准备数据...")
padded_train, mask_train = prepare(TRAIN_PATH, MAX_ARTICLES_TRAIN, MAX_TOKENS_PER_ART)
padded_test,  mask_test  = prepare(TEST_PATH,  MAX_ARTICLES_TEST,  MAX_TOKENS_PER_ART)

# ============================================================
# 提取 V cache
# ============================================================
@torch.no_grad()
def extract_v(model, layer_idx, padded):
    attn = model.model.layers[layer_idx].self_attn
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)
    all_v = []
    for start in range(0, len(padded), BATCH_SIZE):
        batch = padded[start:start+BATCH_SIZE].to(DEVICE)
        B_b, T_b = batch.shape
        out = model.model(input_ids=batch, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[layer_idx]
        normed = model.model.layers[layer_idx].input_layernorm(hidden)
        v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
        v = v.transpose(1, 2).reshape(B_b, T_b, -1).float().cpu()
        all_v.append(v)
        del out, hidden, normed, v
        torch.cuda.empty_cache()
    return torch.cat(all_v, dim=0).reshape(-1, all_v[0].shape[-1])

# ============================================================
# 加载模型
# ============================================================
print("\n加载模型...")
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()

# ============================================================
# 协方差计算
# ============================================================
def compute_cov(X, Y, reg=1e-3):
    N = X.shape[0]
    dX, dY = X.shape[1], Y.shape[1]
    Xm = X.mean(dim=0)
    Ym = Y.mean(dim=0)
    Cxx = torch.zeros(dX, dX)
    Cyy = torch.zeros(dY, dY)
    Cxy = torch.zeros(dX, dY)
    for s in range(0, N, COV_BATCH):
        e = min(s + COV_BATCH, N)
        x = (X[s:e].float() - Xm).to(DEVICE)
        y = (Y[s:e].float() - Ym).to(DEVICE)
        Cxx += (x.T @ x).cpu()
        Cyy += (y.T @ y).cpu()
        Cxy += (x.T @ y).cpu()
        del x, y
        torch.cuda.empty_cache()
    return Cxx / N, Cyy / N, Cxy / N, Xm, Ym

# ============================================================
# 方法 1：CCA 投影
# ============================================================
def cca_projections(Cxx, Cyy, Cxy, r, reg=1e-3):
    dX, dY = Cxx.shape[0], Cyy.shape[0]
    Cxx = Cxx + reg * torch.eye(dX)
    Cyy = Cyy + reg * torch.eye(dY)
    Lx = torch.linalg.cholesky(Cxx)
    Ly = torch.linalg.cholesky(Cyy)
    M = torch.linalg.solve_triangular(Lx, Cxy, upper=False)
    M = torch.linalg.solve_triangular(Ly, M.T, upper=False).T
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    W_S = torch.linalg.solve_triangular(Lx.T, U[:, :r], upper=True)
    W_T = torch.linalg.solve_triangular(Ly.T, Vh[:r].T, upper=True)
    W_S = W_S / W_S.norm(dim=0, keepdim=True).clamp_min(1e-6)
    W_T = W_T / W_T.norm(dim=0, keepdim=True).clamp_min(1e-6)
    return W_S, W_T, S[:r]

# ============================================================
# 方法 2：PLS 投影
# ============================================================
def pls_projections(Cxx, Cyy, Cxy, r, reg=1e-3):
    """
    PLS：直接对 Cxy 做 SVD。
    """
    U, S, Vh = torch.linalg.svd(Cxy, full_matrices=False)
    W_S = U[:, :r]
    W_T = Vh[:r].T
    W_S = W_S / W_S.norm(dim=0, keepdim=True).clamp_min(1e-6)
    W_T = W_T / W_T.norm(dim=0, keepdim=True).clamp_min(1e-6)
    return W_S, W_T, S[:r]

# ============================================================
# 方法 3：低秩回归
# ============================================================
def low_rank_regression(Cxx, Cxy, r, lam=1.0):
    """
    解 W_full = (Cxx + lam I)^-1 Cxy，再取低秩近似。
    返回 W_r: [dX, dY]，rank r。
    """
    dX, dY = Cxx.shape[0], Cxy.shape[1]
    W_full = torch.linalg.solve(Cxx + lam * torch.eye(dX), Cxy)
    U, S, Vh = torch.linalg.svd(W_full, full_matrices=False)
    W_r = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r]
    return W_r

# ============================================================
# 线性映射 g: A_S -> X_T
# ============================================================
def fit_linear_map(A, X_T, lam=1.0, batch_size=COV_BATCH):
    N = A.shape[0]
    dA = A.shape[1]
    dT = X_T.shape[1]
    AtA = torch.zeros(dA, dA)
    AtY = torch.zeros(dA, dT)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        a = A[s:e].float()
        y = X_T[s:e].float()
        AtA += a.T @ a
        AtY += a.T @ y
        del a, y
    M = torch.linalg.solve(AtA + lam * torch.eye(dA), AtY)
    return M

def eval_reconstruct(A_test, X_T_test, M, mean_T):
    X_hat = A_test.float() @ M + mean_T
    residual = ((X_hat - X_T_test.float()) ** 2).sum()
    total = ((X_T_test.float() - mean_T) ** 2).sum()
    return (1 - residual / total).item()

# ============================================================
# 主循环
# ============================================================
results = {}   # results[layer][method][r] = (r2, extra)

for layer in LAYERS_TO_TEST:
    print(f"\n{'='*60}")
    print(f"Layer {layer}")
    print(f"{'='*60}")

    VA_train = extract_v(model_A, layer, padded_train)
    VB_train = extract_v(model_B, layer, padded_train)
    VA_test  = extract_v(model_A, layer, padded_test)
    VB_test  = extract_v(model_B, layer, padded_test)

    mtr = mask_train.reshape(-1)
    mte = mask_test.reshape(-1)
    VA_train, VB_train = VA_train[mtr], VB_train[mtr]
    VA_test,  VB_test  = VA_test[mte],  VB_test[mte]

    print(f"  Train tokens: {VA_train.shape}, Test tokens: {VA_test.shape}")

    # 一次算协方差，三种方法共用
    Cxx, Cyy, Cxy, mS, mT = compute_cov(VA_train, VB_train)

    results[layer] = {"CCA": {}, "PLS": {}, "LRR": {}}

    for r in R_VALUES:
        # ---- CCA ----
        W_S, W_T, S = cca_projections(Cxx, Cyy, Cxy, r)
        A_tr = (VA_train - mS) @ W_S
        A_te = (VA_test  - mS) @ W_S
        M = fit_linear_map(A_tr, VB_train - mT)
        r2_cca = eval_reconstruct(A_te, VB_test, M, mT)
        results[layer]["CCA"][r] = (r2_cca, S[0].item())

        # ---- PLS ----
        W_S, W_T, S = pls_projections(Cxx, Cyy, Cxy, r)
        A_tr = (VA_train - mS) @ W_S
        A_te = (VA_test  - mS) @ W_S
        M = fit_linear_map(A_tr, VB_train - mT)
        r2_pls = eval_reconstruct(A_te, VB_test, M, mT)
        results[layer]["PLS"][r] = (r2_pls, S[0].item())

        # ---- 低秩回归 ----
        W_r = low_rank_regression(Cxx, Cxy, r)
        r2_lrr = eval_reconstruct(VA_test - mS, VB_test, W_r, mT)
        results[layer]["LRR"][r] = (r2_lrr, None)

        print(f"  r={r:>4}: CCA={r2_cca:.4f}  PLS={r2_pls:.4f}  LRR={r2_lrr:.4f}")

    del VA_train, VB_train, VA_test, VB_test
    torch.cuda.empty_cache()

# ============================================================
# 汇总
# ============================================================
for method in ["CCA", "PLS", "LRR"]:
    print(f"\n{'='*80}")
    print(f"[{method}] test R²")
    print(f"{'='*80}")
    header = f"{'Layer':>6}  " + "  ".join(f"r={r:<5}" for r in R_VALUES)
    print(header)
    print("-" * len(header))
    for layer in LAYERS_TO_TEST:
        row = f"{layer:>6}  "
        for r in R_VALUES:
            r2, _ = results[layer][method][r]
            row += f"{r2:>7.4f}  "
        print(row)

# ============================================================
# 三种方法对比（每个 rank）
# ============================================================
print(f"\n{'='*80}")
print("三种方法逐层对比（test R²）")
print(f"{'='*80}")
for r in R_VALUES:
    print(f"\n--- r = {r} ---")
    header = f"{'Layer':>6}  {'CCA':>8}  {'PLS':>8}  {'LRR':>8}  {'Best':>8}"
    print(header)
    print("-" * len(header))
    for layer in LAYERS_TO_TEST:
        vals = {
            "CCA": results[layer]["CCA"][r][0],
            "PLS": results[layer]["PLS"][r][0],
            "LRR": results[layer]["LRR"][r][0],
        }
        best = max(vals, key=vals.get)
        print(f"{layer:>6}  {vals['CCA']:>8.4f}  {vals['PLS']:>8.4f}  {vals['LRR']:>8.4f}  {best:>8}")