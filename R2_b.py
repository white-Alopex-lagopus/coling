import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

MAX_ARTICLES = 9007         # 可以调大，内存不受影响
MAX_TOKENS_PER_ART = 256
MAX_OFFSET = 3
DEVICE = "cuda"
N_FOLDS = 5

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

N_total = len(article_ids) * max_len

# fold 划分：按 token 索引（你已验证 token 近似独立）
perm = torch.randperm(N_total)
fold_ids = torch.zeros(N_total, dtype=torch.long)
for k in range(N_FOLDS):
    fold_ids[perm[k::N_FOLDS]] = k


# ============================================================
# 分 batch 累加 Ridge 所需矩阵
# ============================================================
def compute_ridge_cv_batch(model_A, model_B, layer_A, layer_B,
                            padded, fold_ids, n_folds, lam=100.0,
                            batch_size=32):
    """
    不存全部 hidden，分 batch 累加 XtX 和 XtY，然后解 W，算 CV R²。
    """
    # 每个 fold 一个累加器
    # 第一次 forward 后才知道 D_A, D_B，所以先延迟初始化
    XtX_list = None   # list of [D_A, D_A]
    XtY_list = None   # list of [D_A, D_B]
    XtX_test_list = None
    XtY_test_list = None

    y_norm_list = None   # 每个 fold 的 ||Y_test||²

    for start in range(0, len(padded), batch_size):
        batch = padded[start:start+batch_size].to(DEVICE)
        bsz = batch.shape[0]

        with torch.no_grad():
            out_A = model_A.model(input_ids=batch, output_hidden_states=True, use_cache=False)
            out_B = model_B.model(input_ids=batch, output_hidden_states=True, use_cache=False)

        h_A = out_A.hidden_states[layer_A].reshape(-1, out_A.hidden_states[layer_A].shape[-1])
        h_B = out_B.hidden_states[layer_B].reshape(-1, out_B.hidden_states[layer_B].shape[-1])

        # 当前 batch 的 token 在全局的索引
        idx = torch.arange(start * max_len, (start + bsz) * max_len)
        fids = fold_ids[idx]

        D_A = h_A.shape[-1]
        D_B = h_B.shape[-1]

        if XtX_list is None:
            XtX_list = [torch.zeros(D_A, D_A, dtype=torch.float32) for _ in range(n_folds)]
            XtY_list = [torch.zeros(D_A, D_B, dtype=torch.float32) for _ in range(n_folds)]
            XtX_test_list = [torch.zeros(D_A, D_A, dtype=torch.float32) for _ in range(n_folds)]
            XtY_test_list = [torch.zeros(D_A, D_B, dtype=torch.float32) for _ in range(n_folds)]
            y_norm_list = [0.0 for _ in range(n_folds)]

        h_A = h_A.float().cpu()
        h_B = h_B.float().cpu()

        for k in range(n_folds):
            tr_mask = fids != k
            te_mask = fids == k

            if tr_mask.sum() > 0:
                X_tr = h_A[tr_mask]
                Y_tr = h_B[tr_mask]
                XtX_list[k] += X_tr.T @ X_tr
                XtY_list[k] += X_tr.T @ Y_tr

            if te_mask.sum() > 0:
                X_te = h_A[te_mask]
                Y_te = h_B[te_mask]
                XtX_test_list[k] += X_te.T @ X_te
                XtY_test_list[k] += X_te.T @ Y_te
                y_norm_list[k] += (Y_te ** 2).sum().item()

        del out_A, out_B, h_A, h_B
        torch.cuda.empty_cache()

    # 每个 fold 解 W，算 test R²
    r2_list = []
    for k in range(n_folds):
        XtX = XtX_list[k]
        XtY = XtY_list[k]
        D_A = XtX.shape[0]

        # 注意：这里没有做标准化，为了简化。
        # 如果要更严格，需要先算 mean/std，但这需要两遍 forward。
        # 这里直接用 ridge + 足够大的 λ 控制。

        W = torch.linalg.solve(XtX + lam * torch.eye(D_A), XtY)

        # test R²
        # ||X_te W - Y_te||² = ||X_te W||² - 2<X_te W, Y_te> + ||Y_te||²
        #                   = tr(W^T XtX_test W) - 2 tr(W^T XtY_test) + ||Y_te||²
        XtX_te = XtX_test_list[k]
        XtY_te = XtY_test_list[k]

        term1 = (W.T @ XtX_te @ W).trace().item()
        term2 = (W.T @ XtY_te).trace().item()
        y_norm = y_norm_list[k]

        residual = term1 - 2 * term2 + y_norm
        r2 = 1 - residual / (y_norm + 1e-8)
        r2_list.append(r2)

    return float(np.mean(r2_list))


# ============================================================
# 加载模型
# ============================================================
print("\n加载模型...")
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()

L_A = model_A.config.num_hidden_layers
L_B = model_B.config.num_hidden_layers

# ============================================================
# 算对角线附近
# ============================================================
print(f"\n计算 Ridge CV R²（|i-j| <= {MAX_OFFSET}）...")

ridge = np.full((L_A, L_B), np.nan)
n_pairs = 0

for i in range(L_A):
    for j in range(max(0, i - MAX_OFFSET), min(L_B, i + MAX_OFFSET + 1)):
        ridge[i, j] = compute_ridge_cv_batch(
            model_A, model_B, i, j,
            padded, fold_ids, N_FOLDS, lam=100.0, batch_size=32
        )
        n_pairs += 1
    print(f"  row {i+1}/{L_A}  (pairs: {n_pairs})")

np.save("ridge_cv_diagband.npy", ridge)

# ============================================================
# 统计
# ============================================================
diag = np.array([ridge[i, i] for i in range(min(L_A, L_B)) if not np.isnan(ridge[i, i])])
off = np.array([ridge[i, j]
                for i in range(L_A)
                for j in range(max(0, i-MAX_OFFSET), min(L_B, i+MAX_OFFSET+1))
                if i != j and not np.isnan(ridge[i, j])])

print()
print("=" * 70)
print("Ridge CV R²")
print("=" * 70)
print(f"Diagonal mean   : {diag.mean():.4f}")
print(f"Off-diag mean   : {off.mean():.4f}")
print(f"Diag - offdiag  : {diag.mean() - off.mean():+.4f}")

print("\n每层偏移平均：")
for offset in range(-MAX_OFFSET, MAX_OFFSET + 1):
    vals = [ridge[i, i+offset] for i in range(L_A)
            if 0 <= i+offset < L_B and not np.isnan(ridge[i, i+offset])]
    if vals:
        print(f"  offset {offset:+d}: {np.mean(vals):.4f}  (n={len(vals)})")