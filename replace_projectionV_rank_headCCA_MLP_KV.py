import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
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

N_TRAIN = 200
N_TEST  = 30
MAX_LEN = 128

RANKS = [32, 64, 128, 256]        # 方向 A：扫 rank
PER_HEAD_RANK = 8                  # 方向 B：每 head rank（8 head × 8 = 64 总）
MLP_RANK = 64                      # 方向 C：用 r=64 的 CCA 做基础 + MLP
MLP_HIDDEN = 256
MLP_EPOCHS = 10
MLP_LR = 1e-3

DEVICE = "cuda"
SEED = 42
STREAM_BATCH = 10000

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
N_KV = model_B.config.num_key_value_heads
HEAD_DIM = getattr(model_B.config, "head_dim",
                   model_B.config.hidden_size // model_B.config.num_attention_heads)
D_KV = N_KV * HEAD_DIM
print(f"Layers: A={L_A}, B={L_B}, n_kv={N_KV}, head_dim={HEAD_DIM}, d_kv={D_KV}")

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
# 提取 K/V（单层）
# ============================================================
@torch.no_grad()
def extract_kv_one(model, layer_idx, ids):
    """返回 k, v，形状 [1, T, d_kv]"""
    attn = model.model.layers[layer_idx].self_attn
    out = model.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    normed = model.model.layers[layer_idx].input_layernorm(out.hidden_states[layer_idx])
    k = attn.k_proj(normed)
    v = attn.v_proj(normed)
    return k, v

@torch.no_grad()
def extract_kv_flat(model, layer_idx, ids_list):
    """返回 K_flat, V_flat，[N_total, d_kv] float32 CPU"""
    Ks, Vs = [], []
    for ids in ids_list:
        k, v = extract_kv_one(model, layer_idx, ids)
        Ks.append(k.reshape(-1, k.shape[-1]).float().cpu())
        Vs.append(v.reshape(-1, v.shape[-1]).float().cpu())
    return torch.cat(Ks, 0), torch.cat(Vs, 0)

# ============================================================
# 协方差 + CCA 工具
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
    return Cxx / N, Cyy / N, Cxy / N, mX, mY

def cca_projections(Cxx, Cyy, Cxy, r, reg=1e-3):
    dX, dY = Cxx.shape[0], Cyy.shape[0]
    Cxx_r = Cxx + reg * torch.eye(dX)
    Cyy_r = Cyy + reg * torch.eye(dY)
    Lx = torch.linalg.cholesky(Cxx_r)
    Ly = torch.linalg.cholesky(Cyy_r)
    M = torch.linalg.solve_triangular(Lx, Cxy, upper=False)
    M = torch.linalg.solve_triangular(Ly, M.T, upper=False).T
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    W_S = torch.linalg.solve_triangular(Lx.T, U[:, :r], upper=True)
    W_S = W_S / W_S.norm(dim=0, keepdim=True).clamp_min(1e-6)
    return W_S

def fit_linear_M(A, Y, mean_Y, lam=1.0, batch_size=STREAM_BATCH):
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
# 方向 C：MLP 修正
# ============================================================
class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

def train_mlp_correction(A, V_T, mT, M, hidden=MLP_HIDDEN, epochs=MLP_EPOCHS, lr=MLP_LR):
    """
    A: [N, r] 训练侧投影坐标
    V_T: [N, d] 目标 V
    mT: [1, d] 目标均值
    M: [r, d] 线性映射
    学习 MLP(A) ≈ V_T - (A @ M + mT)
    """
    with torch.no_grad():
        base = A @ M + mT              # [N, d]
        residual = V_T - base          # [N, d]

    # 采样控制训练规模
    N = A.shape[0]
    if N > 20000:
        idx = torch.randperm(N)[:20000]
        A_tr = A[idx].to(DEVICE)
        R_tr = residual[idx].to(DEVICE)
    else:
        A_tr = A.to(DEVICE)
        R_tr = residual.to(DEVICE)

    mlp = ResidualMLP(A.shape[1], hidden, V_T.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)

    bs = 2048
    for ep in range(epochs):
        perm = torch.randperm(A_tr.shape[0], device=DEVICE)
        total = 0.0
        for s in range(0, A_tr.shape[0], bs):
            e = min(s + bs, A_tr.shape[0])
            idx_b = perm[s:e]
            pred = mlp(A_tr[idx_b])
            loss = ((pred - R_tr[idx_b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * (e - s)
        total /= A_tr.shape[0]
        if (ep + 1) % 5 == 0:
            print(f"      MLP ep{ep+1}: mse={total:.4f}")

    mlp.eval()
    return mlp

# ============================================================
# 阶段 1：每层学投影
# ============================================================
print(f"\n{'='*60}")
print("阶段 1：学每层投影")
print(f"{'='*60}")

proj = {}   # proj[l] = {...}

for l in range(min(L_A, L_B)):
    print(f"\nLayer {l}:")

    # 提取训练数据
    K_A_tr, V_A_tr = extract_kv_flat(model_A, l, train_ids)
    K_B_tr, V_B_tr = extract_kv_flat(model_B, l, train_ids)
    print(f"  train: V_A={tuple(V_A_tr.shape)}, V_B={tuple(V_B_tr.shape)}")

    # ---- 全局 CCA（V），各 rank 共享同一个 W_S 前缀 ----
    Cxx, Cyy, Cxy, mS_V, mT_V = compute_cov(V_A_tr, V_B_tr)
    W_S_max = cca_projections(Cxx, Cyy, Cxy, max(RANKS))

    entry = {
        "mS_V": mS_V, "mT_V": mT_V,
        "W_S_V": W_S_max,                # [d_S, max(RANKS)]
        "M_V": {},                       # rank -> [rank, d_T]
        "per_head": None,
        "mlp": None,
        "mS_K": None, "mT_K": None, "W_S_K": None, "M_K": None,  # 方向 D 用
    }
    for r in RANKS:
        W_S_r = W_S_max[:, :r]
        A_tr = (V_A_tr - mS_V) @ W_S_r
        M = fit_linear_M(A_tr, V_B_tr, mT_V)
        entry["M_V"][r] = M

    # ---- 方向 B：逐头 CCA ----
    per_head = []
    for h in range(N_KV):
        s = h * HEAD_DIM
        e = (h + 1) * HEAD_DIM
        V_A_h = V_A_tr[:, s:e]
        V_B_h = V_B_tr[:, s:e]
        Cxx_h, Cyy_h, Cxy_h, mS_h, mT_h = compute_cov(V_A_h, V_B_h)
        W_S_h = cca_projections(Cxx_h, Cyy_h, Cxy_h, PER_HEAD_RANK)
        A_h = (V_A_h - mS_h) @ W_S_h
        M_h = fit_linear_M(A_h, V_B_h, mT_h)
        per_head.append({
            "W_S": W_S_h, "M": M_h, "mS": mS_h, "mT": mT_h,
            "slice": (s, e),
        })
    entry["per_head"] = per_head

    # ---- 方向 C：MLP 修正（在 r=MLP_RANK 的 CCA 上） ----
    W_S_mlp = W_S_max[:, :MLP_RANK]
    A_mlp = (V_A_tr - mS_V) @ W_S_mlp
    M_mlp = entry["M_V"][MLP_RANK]
    print(f"  训练 MLP 修正（rank={MLP_RANK}）...")
    mlp = train_mlp_correction(A_mlp, V_B_tr, mT_V, M_mlp)
    entry["mlp"] = {
        "W_S": W_S_mlp, "M": M_mlp, "mS": mS_V, "mT": mT_V,
        "net": mlp,
    }

    # ---- 方向 D：K 也重建（用 r=64） ----
    Cxx_K, Cyy_K, Cxy_K, mS_K, mT_K = compute_cov(K_A_tr, K_B_tr)
    W_S_K = cca_projections(Cxx_K, Cyy_K, Cxy_K, 64)
    A_K = (K_A_tr - mS_K) @ W_S_K
    M_K = fit_linear_M(A_K, K_B_tr, mT_K)
    entry.update({"mS_K": mS_K, "mT_K": mT_K, "W_S_K": W_S_K, "M_K": M_K})

    proj[l] = entry

    del K_A_tr, V_A_tr, K_B_tr, V_B_tr
    gc.collect()
    torch.cuda.empty_cache()

print("\n阶段 1 完成。")

# ============================================================
# 阶段 2：重建函数
# ============================================================
@torch.no_grad()
def reconstruct_V(ids, method, rank=None):
    """
    返回 {layer: V_hat [1, T, d]}（CPU float16）
    method: "cca", "per_head", "cca_mlp", "cca_kv"(V部分), "sharer", "zero", "random"
    """
    out = {}
    for l in range(min(L_A, L_B)):
        p = proj[l]
        _, v_A = extract_kv_one(model_A, l, ids)     # [1, T, d]
        d = v_A.shape[-1]
        v_flat = v_A.reshape(-1, d).float().cpu()

        if method == "sharer":
            v_hat = v_flat
        elif method == "cca":
            W = p["W_S_V"][:, :rank]
            A = (v_flat - p["mS_V"]) @ W
            v_hat = A @ p["M_V"][rank] + p["mT_V"]
        elif method == "per_head":
            parts = []
            for h_info in p["per_head"]:
                s, e = h_info["slice"]
                v_h = v_flat[:, s:e]
                A_h = (v_h - h_info["mS"]) @ h_info["W_S"]
                vh_hat = A_h @ h_info["M"] + h_info["mT"]
                parts.append(vh_hat)
            v_hat = torch.cat(parts, dim=1)
        elif method == "cca_mlp":
            info = p["mlp"]
            A = (v_flat - info["mS"]) @ info["W_S"]
            base = A @ info["M"] + info["mT"]
            with torch.no_grad():
                corr = info["net"](A.to(DEVICE)).cpu()
            v_hat = base + corr
        elif method == "cca_kv":
            # 只重建 V（K 单独处理）
            W = p["W_S_V"][:, :64]
            A = (v_flat - p["mS_V"]) @ W
            v_hat = A @ p["M_V"][64] + p["mT_V"]
        elif method == "zero":
            v_hat = torch.zeros_like(v_flat)
        elif method == "random":
            v_hat = torch.randn_like(v_flat)
        else:
            raise ValueError(method)

        out[l] = v_hat.reshape(1, -1, d).to(torch.float16)
    return out

@torch.no_grad()
def reconstruct_K(ids):
    """方向 D：重建 K"""
    out = {}
    for l in range(min(L_A, L_B)):
        p = proj[l]
        k_A, _ = extract_kv_one(model_A, l, ids)
        d = k_A.shape[-1]
        k_flat = k_A.reshape(-1, d).float().cpu()
        A = (k_flat - p["mS_K"]) @ p["W_S_K"]
        k_hat = A @ p["M_K"] + p["mT_K"]
        out[l] = k_hat.reshape(1, -1, d).to(torch.float16)
    return out

# ============================================================
# 替换 forward
# ============================================================
def forward_with_replacement(ids, v_replace=None, k_replace=None):
    hooks = []
    if v_replace is not None:
        for l, vt in v_replace.items():
            vt_d = vt.to(DEVICE).to(torch.float16)
            def make_hook(vt_d):
                def hook(module, inp, out):
                    if out.shape != vt_d.shape:
                        return out
                    return vt_d
                return hook
            hooks.append(model_B.model.layers[l].self_attn.v_proj.register_forward_hook(make_hook(vt_d)))
    if k_replace is not None:
        for l, kt in k_replace.items():
            kt_d = kt.to(DEVICE).to(torch.float16)
            def make_hook(kt_d):
                def hook(module, inp, out):
                    if out.shape != kt_d.shape:
                        return out
                    return kt_d
                return hook
            hooks.append(model_B.model.layers[l].self_attn.k_proj.register_forward_hook(make_hook(kt_d)))
    try:
        with torch.no_grad():
            logits = model_B(input_ids=ids).logits
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
    agree = (lb.argmax(-1) == lr.argmax(-1)).float().mean().item()
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
# 阶段 2：评估
# ============================================================
print(f"\n{'='*60}")
print("阶段 2：评估")
print(f"{'='*60}")

methods = (
    [("cca_r{}".format(r), "cca", r) for r in RANKS]
    + [("per_head", "per_head", None)]
    + [("cca_mlp", "cca_mlp", None)]
    + [("cca_kv", "cca_kv", None)]
    + [("sharer", "sharer", None), ("zero", "zero", None), ("random", "random", None)]
)

results = {name: {"ce": [], "kl": [], "agree": []} for name, _, _ in methods}
results["baseline"] = {"ce": [], "kl": [], "agree": []}

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

    # 常规方法（只换 V）
    for name, mtype, rank in methods:
        if name == "cca_kv":
            continue   # 单独处理
        v_repl = reconstruct_V(ids, mtype, rank)
        logits = forward_with_replacement(ids, v_replace=v_repl)
        ce, kl, agree = compute_metrics(logits_base, logits, ids)
        results[name]["ce"].append(ce)
        results[name]["kl"].append(kl)
        results[name]["agree"].append(agree)

    # 方向 D：同时换 K 和 V
    v_repl = reconstruct_V(ids, "cca_kv")
    k_repl = reconstruct_K(ids)
    logits = forward_with_replacement(ids, v_replace=v_repl, k_replace=k_repl)
    ce, kl, agree = compute_metrics(logits_base, logits, ids)
    results["cca_kv"]["ce"].append(ce)
    results["cca_kv"]["kl"].append(kl)
    results["cca_kv"]["agree"].append(agree)

    if (i + 1) % 10 == 0:
        print(f"  已处理 {i+1}/{len(test_ids)}")

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*90}")
print(f"多层重建 V 替换汇总（{L_B} 层）")
print(f"{'='*90}")
print(f"{'Method':>14}  {'CE':>8}  {'ΔCE':>8}  {'KL':>10}  {'Top1-AGREE':>12}")
print("-" * 90)

base_ce = np.mean(results["baseline"]["ce"])
order = ["baseline"] + [n for n, _, _ in methods]
for m in order:
    ce = np.mean(results[m]["ce"])
    kl = np.mean(results[m]["kl"])
    agree = np.mean(results[m]["agree"])
    print(f"{m:>14}  {ce:>8.4f}  {ce - base_ce:>+8.4f}  {kl:>10.4f}  {agree:>12.4f}")

# ============================================================
# 结论速览
# ============================================================
print(f"\n{'='*60}")
print("关键结论")
print(f"{'='*60}")
for m in [n for n, _, _ in methods]:
    ce = np.mean(results[m]["ce"])
    delta = ce - base_ce
    if delta < 1.0:
        verdict = "接近 baseline，方案可行"
    elif delta < 3.0:
        verdict = "有损失，可优化"
    elif delta < 8.0:
        verdict = "损失较大"
    else:
        verdict = "基本崩溃"
    print(f"  {m:>14}: ΔCE={delta:+.4f}  →  {verdict}")