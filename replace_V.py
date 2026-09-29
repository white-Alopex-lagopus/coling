import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import torch.nn.functional as F
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

LAYER = 14
RANK = 64
N_TRAIN_ARTICLES = 200
N_TEST_ARTICLES = 100
MAX_LEN = 128

DEVICE = "cuda"
SEED = 42
BATCH_SIZE = 8
COV_BATCH = 20000

torch.manual_seed(SEED)

# ============================================================
# 加载模型
# ============================================================
print("加载模型...")
tok = AutoTokenizer.from_pretrained(MODEL_A)
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()

# ============================================================
# 数据
# ============================================================
def load_texts(path, n):
    df = pd.read_json(path, lines=True)
    return df["text"].iloc[:n].tolist()

train_texts = load_texts(TRAIN_PATH, N_TRAIN_ARTICLES)
test_texts  = load_texts(TEST_PATH, N_TEST_ARTICLES)
print(f"Train texts: {len(train_texts)}, Test texts: {len(test_texts)}")

def tokenize(texts, max_len=MAX_LEN):
    ids_list = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if len(ids) >= 32:
            ids_list.append(ids[:max_len])
    return ids_list

train_ids = tokenize(train_texts)
test_ids  = tokenize(test_texts)

# ============================================================
# 提取 V（第 L 层）
# ============================================================
@torch.no_grad()
def extract_v(model, input_ids):
    """input_ids: [T] → V: [T, d]"""
    attn = model.model.layers[LAYER].self_attn
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)
    ids = input_ids.unsqueeze(0).to(DEVICE)
    out = model.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states[LAYER]
    normed = model.model.layers[LAYER].input_layernorm(hidden)
    v = attn.v_proj(normed).view(1, -1, n_kv, head_dim).transpose(1, 2)
    v = v.transpose(1, 2).reshape(-1, n_kv * head_dim).float().cpu()
    return v

# ============================================================
# 学 CCA / PLS / LRR 参数（用 train 集）
# ============================================================
print("\n提取 train 集 V...")
V_A_train = []
V_B_train = []
with torch.no_grad():
    for ids in train_ids:
        V_A_train.append(extract_v(model_A, ids))
        V_B_train.append(extract_v(model_B, ids))
V_A_train = torch.cat(V_A_train, dim=0)
V_B_train = torch.cat(V_B_train, dim=0)
print(f"V_A_train: {tuple(V_A_train.shape)}, V_B_train: {tuple(V_B_train.shape)}")

def compute_cov(X, Y, batch_size=COV_BATCH):
    N = X.shape[0]
    dX, dY = X.shape[1], Y.shape[1]
    mX = X.mean(dim=0)
    mY = Y.mean(dim=0)
    Cxx = torch.zeros(dX, dX)
    Cyy = torch.zeros(dY, dY)
    Cxy = torch.zeros(dX, dY)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        x = X[s:e].float() - mX
        y = Y[s:e].float() - mY
        Cxx += x.T @ x
        Cyy += y.T @ y
        Cxy += x.T @ y
    return Cxx / N, Cyy / N, Cxy / N, mX, mY

Cxx, Cyy, Cxy, mS, mT = compute_cov(V_A_train, V_B_train)

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
    return W_S, S[:r]

def pls_projections(Cxy, r):
    U, S, Vh = torch.linalg.svd(Cxy, full_matrices=False)
    W_S = U[:, :r]
    W_S = W_S / W_S.norm(dim=0, keepdim=True).clamp_min(1e-6)
    return W_S, S[:r]

def low_rank_regression(Cxx, Cxy, r, lam=1.0):
    dX = Cxx.shape[0]
    W_full = torch.linalg.solve(Cxx + lam * torch.eye(dX), Cxy)
    U, S, Vh = torch.linalg.svd(W_full, full_matrices=False)
    W_r = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r]
    return W_r

print("\n学投影矩阵...")
W_S_cca, _ = cca_projections(Cxx, Cyy, Cxy, RANK)
W_S_pls, _ = pls_projections(Cxy, RANK)
W_r_lrr = low_rank_regression(Cxx, Cxy, RANK)

# 学映射 M: A_S -> V_T
def fit_M(A, Y, mean_Y, lam=1.0, batch_size=COV_BATCH):
    N = A.shape[0]
    r = A.shape[1]
    dY = Y.shape[1]
    AtA = torch.zeros(r, r)
    AtY = torch.zeros(r, dY)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        a = A[s:e].float()
        y = Y[s:e].float() - mean_Y
        AtA += a.T @ a
        AtY += a.T @ y
    return torch.linalg.solve(AtA + lam * torch.eye(r), AtY)

A_train_cca = (V_A_train - mS) @ W_S_cca
M_cca = fit_M(A_train_cca, V_B_train, mT)

A_train_pls = (V_A_train - mS) @ W_S_pls
M_pls = fit_M(A_train_pls, V_B_train, mT)

# LRR 直接用 W_r_lrr

# ============================================================
# 替换实验：在 Receiver forward 时替换第 L 层的 V
# ============================================================
def compute_loss(model, input_ids, new_v=None):
    """
    在 Receiver 上算 next-token loss。
    如果 new_v 不为 None，替换第 L 层的 V。
    input_ids: [T]
    """
    ids = input_ids.unsqueeze(0).to(DEVICE)
    labels = ids.clone()

    hook_handle = None
    if new_v is not None:
        # new_v: [T, d]，转成 v_proj 的输出形状 [1, T, d]
        v_tensor = new_v.unsqueeze(0).to(DEVICE).to(torch.float16)
        def hook(module, inp, out):
            print(f"[hook] v_proj output shape: {out.shape}, new_v shape: {v_tensor.shape}")
            return v_tensor
        hook_handle = model.model.layers[LAYER].self_attn.v_proj.register_forward_hook(hook)

    try:
        with torch.no_grad():
            out = model(input_ids=ids, labels=labels)
            loss = out.loss.item()
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    return loss

def reconstruct_v(V_A, method):
    """V_A: [T, d] → 重建 V_B_hat: [T, d]"""
    if method == "cca":
        A = (V_A - mS) @ W_S_cca
        return A @ M_cca + mT
    elif method == "pls":
        A = (V_A - mS) @ W_S_pls
        return A @ M_pls + mT
    elif method == "lrr":
        return (V_A - mS) @ W_r_lrr + mT
    elif method == "sharer":
        return V_A  # 直接用 Sharer 的 V（形状一致的话）
    elif method == "zero":
        return torch.zeros_like(V_A)
    else:
        raise ValueError(method)

# ============================================================
# 主循环：逐段 test 文本
# ============================================================
print(f"\n开始替换实验（Layer {LAYER}, rank {RANK}）...")

losses = {"baseline": [], "cca": [], "pls": [], "lrr": [], "zero": []}

with torch.no_grad():
    for i, ids in enumerate(test_ids):
        # Baseline
        loss_base = compute_loss(model_B, ids)
        losses["baseline"].append(loss_base)

        # 提取 Sharer V
        V_A = extract_v(model_A, ids)

        # 各种方法重建
        V_cca = reconstruct_v(V_A, "cca")
        V_pls = reconstruct_v(V_A, "pls")
        V_lrr = reconstruct_v(V_A, "lrr")
        V_zero = reconstruct_v(V_A, "zero")

        # 替换
        losses["cca"].append(compute_loss(model_B, ids, V_cca))
        losses["pls"].append(compute_loss(model_B, ids, V_pls))
        losses["lrr"].append(compute_loss(model_B, ids, V_lrr))
        losses["zero"].append(compute_loss(model_B, ids, V_zero))

        if (i + 1) % 20 == 0:
            print(f"  已处理 {i+1}/{len(test_ids)}")

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*60}")
print(f"Next-token loss（Layer {LAYER}, rank {RANK}）")
print(f"{'='*60}")
print(f"{'Method':>12}  {'Mean Loss':>10}  {'Δ vs baseline':>15}")
print("-" * 60)

baseline_mean = np.mean(losses["baseline"])
for m in ["baseline", "cca", "pls", "lrr", "zero"]:
    mean = np.mean(losses[m])
    delta = mean - baseline_mean
    print(f"{m:>12}  {mean:>10.4f}  {delta:>+15.4f}")

print(f"\nPerplexity:")
for m in ["baseline", "cca", "pls", "lrr", "zero"]:
    ppl = np.exp(np.mean(losses[m]))
    print(f"  {m:>12}: {ppl:.4f}")