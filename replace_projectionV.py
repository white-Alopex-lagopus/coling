import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置
# ============================================================
MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"
TRAIN_PATH = "../dataset/wikitext-02/train.jsonl"
TEST_PATH  = "../dataset/wikitext-02/test.jsonl"

N_TRAIN = 300
N_TEST  = 50
MAX_LEN = 128
RANK = 64
DEVICE = "cuda"
SEED = 42
BATCH_SIZE = 8
STREAM_BATCH = 20000

torch.manual_seed(SEED)

# ============================================================
# 加载
# ============================================================
print("加载模型...")
tok = AutoTokenizer.from_pretrained(MODEL_A)
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()

L_A = model_A.config.num_hidden_layers
L_B = model_B.config.num_hidden_layers
print(f"Layers: A={L_A}, B={L_B}")

# ============================================================
# 数据
# ============================================================
def load_ids(path, n, max_len=MAX_LEN):
    df = pd.read_json(path, lines=True)
    out = []
    for t in df["text"].iloc[:n].tolist():
        ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if len(ids) >= 32:
            out.append(ids[:max_len].unsqueeze(0).to(DEVICE))
    return out

train_ids = load_ids(TRAIN_PATH, N_TRAIN)
test_ids  = load_ids(TEST_PATH,  N_TEST)
print(f"Train: {len(train_ids)}, Test: {len(test_ids)}")

# ============================================================
# 提取 K、V
# ============================================================
@torch.no_grad()
def extract_v(model, layer_idx, ids):
    attn = model.model.layers[layer_idx].self_attn
    out = model.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    normed = model.model.layers[layer_idx].input_layernorm(out.hidden_states[layer_idx])
    return attn.v_proj(normed)   # [1, T, d]

@torch.no_grad()
def extract_v_flatten(model, layer_idx, ids_list):
    """返回 [N_total, d] float32 CPU"""
    outs = []
    for ids in ids_list:
        v = extract_v(model, layer_idx, ids)              # [1, T, d]
        outs.append(v.reshape(-1, v.shape[-1]).float().cpu())
    return torch.cat(outs, dim=0)

# ============================================================
# 协方差 + 投影（每层独立学）
# ============================================================
def compute_cov(X, Y, batch_size=STREAM_BATCH):
    N = X.shape[0]
    dX, dY = X.shape[1], Y.shape[1]
    mX = X.mean(dim=0)
    mY = Y.mean(dim=0)
    Cxx = torch.zeros(dX, dX)
    Cyy = torch.zeros(dY, dY)
    Cxy = torch.zeros(dX, dY)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        x = X[s:e] - mX
        y = Y[s:e] - mY
        Cxx += x.T @ x
        Cyy += y.T @ y
        Cxy += x.T @ y
        del x, y
    return Cxx / N, Cyy / N, Cxy / N, mX, mY

def cca_proj(Cxx, Cyy, Cxy, r, reg=1e-3):
    dX, dY = Cxx.shape[0], Cyy.shape[0]
    Cxx = Cxx + reg * torch.eye(dX)
    Cyy = Cyy + reg * torch.eye(dY)
    Lx = torch.linalg.cholesky(Cxx)
    Ly = torch.linalg.cholesky(Cyy)
    M = torch.linalg.solve_triangular(Lx, Cxy, upper=False)
    M = torch.linalg.solve_triangular(Ly, M.T, upper=False).T
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    W_S = torch.linalg.solve_triangular(Lx.T, U[:, :r], upper=True)
    return W_S / W_S.norm(dim=0, keepdim=True).clamp_min(1e-6)

def pls_proj(Cxy, r):
    U, S, Vh = torch.linalg.svd(Cxy, full_matrices=False)
    W_S = U[:, :r]
    return W_S / W_S.norm(dim=0, keepdim=True).clamp_min(1e-6)

def lrr_proj(Cxx, Cxy, r, lam=1.0):
    dX = Cxx.shape[0]
    W = torch.linalg.solve(Cxx + lam * torch.eye(dX), Cxy)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    return U[:, :r] @ torch.diag(S[:r]) @ Vh[:r]

def fit_M(A, Y, mean_Y, lam=1.0, batch_size=STREAM_BATCH):
    N = A.shape[0]
    r = A.shape[1]
    dY = Y.shape[1]
    AtA = torch.zeros(r, r)
    AtY = torch.zeros(r, dY)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        a = A[s:e]
        y = Y[s:e] - mean_Y
        AtA += a.T @ a
        AtY += a.T @ y
    return torch.linalg.solve(AtA + lam * torch.eye(r), AtY)

# ============================================================
# 每层学投影
# ============================================================
print(f"\n学每层投影（{L_A} 层, rank={RANK}）...")
projections = {}   # projections[layer] = {"W_S": ..., "M": ..., "mS":..., "mT":..., "W_lrr":...}

for l in range(L_A):
    V_A_tr = extract_v_flatten(model_A, l, train_ids)
    V_B_tr = extract_v_flatten(model_B, l, train_ids)
    print(f"  Layer {l}: V_A={tuple(V_A_tr.shape)}, V_B={tuple(V_B_tr.shape)}")

    Cxx, Cyy, Cxy, mS, mT = compute_cov(V_A_tr, V_B_tr)

    # CCA
    W_S_cca = cca_proj(Cxx, Cyy, Cxy, RANK)
    A_cca = (V_A_tr - mS) @ W_S_cca
    M_cca = fit_M(A_cca, V_B_tr, mT)

    # PLS
    W_S_pls = pls_proj(Cxy, RANK)
    A_pls = (V_A_tr - mS) @ W_S_pls
    M_pls = fit_M(A_pls, V_B_tr, mT)

    # LRR（直接映射 V_A -> V_B，不需要 M）
    W_lrr = lrr_proj(Cxx, Cxy, RANK)

    projections[l] = {
        "cca": {"W_S": W_S_cca, "M": M_cca, "mS": mS, "mT": mT},
        "pls": {"W_S": W_S_pls, "M": M_pls, "mS": mS, "mT": mT},
        "lrr": {"W": W_lrr, "mS": mS, "mT": mT},
    }

    del V_A_tr, V_B_tr, Cxx, Cyy, Cxy
    gc.collect()
    torch.cuda.empty_cache()

print("投影学习完成。")

# ============================================================
# 重建 V（单个样本）
# ============================================================
@torch.no_grad()
def reconstruct_v_sharer(ids, method):
    """
    返回 {layer: V_hat [1, T, d]}
    用 Sharer 的 V 重建 Receiver 的 V。
    """
    out = {}
    for l in range(L_A):
        v_A = extract_v(model_A, l, ids)                  # [1, T, d_A]
        d = v_A.shape[-1]
        # v_A_flat = v_A.reshape(-1, d)
        v_A_flat = v_A.reshape(-1, d).float().cpu()

        if method == "sharer":
            v_hat = v_A_flat
        elif method == "cca":
            p = projections[l]["cca"]
            A = (v_A_flat - p["mS"]) @ p["W_S"]
            v_hat = A @ p["M"] + p["mT"]
        elif method == "pls":
            p = projections[l]["pls"]
            A = (v_A_flat - p["mS"]) @ p["W_S"]
            v_hat = A @ p["M"] + p["mT"]
        elif method == "lrr":
            p = projections[l]["lrr"]
            v_hat = (v_A_flat - p["mS"]) @ p["W"] + p["mT"]
        elif method == "zero":
            v_hat = torch.zeros_like(v_A_flat)
        elif method == "random":
            v_hat = torch.randn_like(v_A_flat)
        else:
            raise ValueError(method)

        out[l] = v_hat.reshape(1, -1, d).to(torch.float16)
    return out

# ============================================================
# 多层替换 forward
# ============================================================
def forward_with_replacement(model, ids, replacements):
    hooks = []
    for layer, v_new in replacements.items():
        vt = v_new.to(DEVICE).to(torch.float16)
        def make_hook(vt):
            def hook(module, inp, out):
                if out.shape != vt.shape:
                    return out
                return vt
            return hook
        hooks.append(model.model.layers[layer].self_attn.v_proj.register_forward_hook(make_hook(vt)))
    try:
        with torch.no_grad():
            logits = model(input_ids=ids).logits
    finally:
        for h in hooks:
            h.remove()
    return logits

# ============================================================
# 指标
# ============================================================
def compute_metrics(logits_base, logits_repl, ids):
    lb = logits_base[:, :-1, :].float()
    lr = logits_repl[:, :-1, :].float()
    top1_b = lb.argmax(dim=-1)
    top1_r = lr.argmax(dim=-1)
    agree = (top1_b == top1_r).float().mean().item()

    logp_base = F.log_softmax(lb, dim=-1)
    p_base = logp_base.exp()
    logp_repl = F.log_softmax(lr, dim=-1)
    kl = F.kl_div(logp_repl, p_base, reduction='batchmean').item()

    ce = F.cross_entropy(
        lr.reshape(-1, lr.size(-1)),
        ids[:, 1:].reshape(-1)
    ).item()
    return ce, kl, agree

# ============================================================
# 主循环
# ============================================================
print(f"\n开始多层重建替换（{L_B} 层, rank={RANK}, {len(test_ids)} 样本）...")

methods = ["baseline", "sharer", "cca", "pls", "lrr", "zero", "random"]
results = {m: {"ce": [], "kl": [], "agree": []} for m in methods}

for i, ids in enumerate(test_ids):
    with torch.no_grad():
        logits_base = model_B(input_ids=ids).logits
    ce_b = F.cross_entropy(
        logits_base[:, :-1, :].float().reshape(-1, logits_base.size(-1)),
        ids[:, 1:].reshape(-1)
    ).item()
    results["baseline"]["ce"].append(ce_b)
    results["baseline"]["kl"].append(0.0)
    results["baseline"]["agree"].append(1.0)

    for method in ["sharer", "cca", "pls", "lrr", "zero", "random"]:
        replacements = reconstruct_v_sharer(ids, method)
        logits = forward_with_replacement(model_B, ids, replacements)
        ce, kl, agree = compute_metrics(logits_base, logits, ids)
        results[method]["ce"].append(ce)
        results[method]["kl"].append(kl)
        results[method]["agree"].append(agree)

    if (i + 1) % 10 == 0:
        print(f"  已处理 {i+1}/{len(test_ids)}")

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*80}")
print(f"多层重建 V 替换（{L_B} 层, rank={RANK}）")
print(f"{'='*80}")
print(f"{'Method':>12}  {'CE':>8}  {'ΔCE':>8}  {'KL':>8}  {'Top1-AGREE':>12}")
print("-" * 80)

base_ce = np.mean(results["baseline"]["ce"])
for m in methods:
    ce = np.mean(results[m]["ce"])
    kl = np.mean(results[m]["kl"])
    agree = np.mean(results[m]["agree"])
    print(f"{m:>12}  {ce:>8.4f}  {ce - base_ce:>+8.4f}  {kl:>8.4f}  {agree:>12.4f}")