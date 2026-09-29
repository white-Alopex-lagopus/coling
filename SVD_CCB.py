import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
R_VALUES = [8, 16, 32, 64, 128, 256, 512]
CCA_COMPONENTS = [1, 2, 4, 8, 16, 32, 64]

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
# RoPE 工具
# ============================================================
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def strip_rope(x, position_ids, base=1000000.0):
    """
    x: [B, H, T, D]
    position_ids: [B, T]
    应用逆 RoPE：sin 取负
    """
    D = x.shape[-1]
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float32, device=x.device) / D))
    # [B, T, D/2]
    freqs = torch.einsum('bt,d->btd', position_ids.float().to(x.device), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)          # [B, T, D]
    cos = emb.cos()[:, None, :, :]                   # [B, 1, T, D]
    sin = emb.sin()[:, None, :, :]
    # 逆旋转：x*cos - rotate_half(x)*sin
    return x * cos - rotate_half(x) * sin

# ============================================================
# 提取 K/V
# ============================================================
@torch.no_grad()
def extract_kv(model, layer_idx, padded, strip_rope_flag=False):
    """
    返回 K 或 V，形状 [N, D]
    strip_rope_flag=True 时返回剥离 RoPE 的 K。
    """
    attn = model.model.layers[layer_idx].self_attn
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)
    rope_base = getattr(model.config, "rope_theta", 1000000.0)

    all_feat = []
    for start in range(0, len(padded), BATCH_SIZE):
        batch = padded[start:start+BATCH_SIZE].to(DEVICE)
        B_b, T_b = batch.shape
        position_ids = torch.arange(T_b, device=DEVICE).unsqueeze(0).expand(B_b, -1)

        out = model.model(input_ids=batch, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[layer_idx]
        normed = model.model.layers[layer_idx].input_layernorm(hidden)

        k = attn.k_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)  # [B, H, T, D]
        v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)

        if strip_rope_flag:
            k = strip_rope(k, position_ids, base=rope_base)

        k = k.transpose(1, 2).reshape(B_b, T_b, -1).float().cpu()
        v = v.transpose(1, 2).reshape(B_b, T_b, -1).float().cpu()
        all_feat.append((k, v))

        del out, hidden, normed, k, v
        torch.cuda.empty_cache()

    K = torch.cat([x[0] for x in all_feat], dim=0).reshape(-1, all_feat[0][0].shape[-1])
    V = torch.cat([x[1] for x in all_feat], dim=0).reshape(-1, all_feat[0][1].shape[-1])
    return K, V

# ============================================================
# 加载模型
# ============================================================
print("\n加载模型...")
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()

# ============================================================
# A: 联合 SVD 共享基
# ============================================================
def find_shared_basis(X_S_train, X_T_train, r):
    mean_S = X_S_train.mean(dim=0, keepdim=True)
    mean_T = X_T_train.mean(dim=0, keepdim=True)
    X_S = X_S_train - mean_S
    X_T = X_T_train - mean_T
    X_cat = torch.cat([X_S, X_T], dim=1)
    _, _, Vh = torch.linalg.svd(X_cat, full_matrices=False)
    V = Vh[:r].T
    d_S = X_S.shape[1]
    return V[:d_S], V[d_S:], mean_S, mean_T

def eval_basis(X_S_test, X_T_test, V_S, V_T, mean_S, mean_T):
    A = (X_S_test - mean_S) @ V_S
    X_hat = A @ V_T.T + mean_T
    residual = ((X_hat - X_T_test) ** 2).sum()
    total = ((X_T_test - mean_T) ** 2).sum()
    return (1 - residual / total).item()

# ============================================================
# B: CCA
# ============================================================
def compute_covariances(X, Y, reg=1e-3):
    """
    分批计算 Cxx, Cyy, Cxy。
    """
    N = X.shape[0]
    dX = X.shape[1]
    dY = Y.shape[1]
    X_mean = X.mean(dim=0)
    Y_mean = Y.mean(dim=0)

    Cxx = torch.zeros(dX, dX, dtype=torch.float32)
    Cyy = torch.zeros(dY, dY, dtype=torch.float32)
    Cxy = torch.zeros(dX, dY, dtype=torch.float32)

    for s in range(0, N, COV_BATCH):
        e = min(s + COV_BATCH, N)
        x = (X[s:e].float() - X_mean).to(DEVICE)
        y = (Y[s:e].float() - Y_mean).to(DEVICE)
        Cxx += (x.T @ x).cpu()
        Cyy += (y.T @ y).cpu()
        Cxy += (x.T @ y).cpu()
        del x, y
        torch.cuda.empty_cache()

    Cxx /= N
    Cyy /= N
    Cxy /= N
    return Cxx, Cyy, Cxy, X_mean, Y_mean

def cca_spectrum(Cxx, Cyy, Cxy, reg=1e-3):
    """
    返回典型相关系数（降序）。
    """
    dX = Cxx.shape[0]
    dY = Cyy.shape[0]
    Cxx = Cxx + reg * torch.eye(dX)
    Cyy = Cyy + reg * torch.eye(dY)

    Lx = torch.linalg.cholesky(Cxx)
    Ly = torch.linalg.cholesky(Cyy)

    # M = Lx^{-1} @ Cxy @ Ly^{-T}
    M = torch.linalg.solve_triangular(Lx, Cxy, upper=False)
    M = torch.linalg.solve_triangular(Ly, M.T, upper=False).T

    S = torch.linalg.svdvals(M)
    return S

# ============================================================
# 主循环
# ============================================================
results_A = {}   # results_A[layer][r] = (train_r2, test_r2)  对 K_stripped
results_B = {}   # results_B[layer] = 典型相关系数列表        对 K_stripped 和 V

for layer in LAYERS_TO_TEST:
    print(f"\n{'='*60}")
    print(f"Layer {layer}")
    print(f"{'='*60}")

    # 提取 KV（不剥离用于 B 的 V；剥离用于 A 和 B 的 K）
    KA_train, VA_train = extract_kv(model_A, layer, padded_train, strip_rope_flag=False)
    KB_train, VB_train = extract_kv(model_B, layer, padded_train, strip_rope_flag=False)
    KA_test,  VA_test  = extract_kv(model_A, layer, padded_test,  strip_rope_flag=False)
    KB_test,  VB_test  = extract_kv(model_B, layer, padded_test,  strip_rope_flag=False)

    # 剥离 RoPE
    KA_train_s = extract_kv(model_A, layer, padded_train, strip_rope_flag=True)[0]
    KB_train_s = extract_kv(model_B, layer, padded_train, strip_rope_flag=True)[0]
    KA_test_s  = extract_kv(model_A, layer, padded_test,  strip_rope_flag=True)[0]
    KB_test_s  = extract_kv(model_B, layer, padded_test,  strip_rope_flag=True)[0]

    # 用 mask 筛选真实 token
    mtr = mask_train.reshape(-1)
    mte = mask_test.reshape(-1)
    KA_train_s, KB_train_s = KA_train_s[mtr], KB_train_s[mtr]
    KA_test_s,  KB_test_s  = KA_test_s[mte],  KB_test_s[mte]
    VA_train, VB_train = VA_train[mtr], VB_train[mtr]
    VA_test,  VB_test  = VA_test[mte],  VB_test[mte]

    # ---------- A: K_stripped 的联合 SVD ----------
    print(f"\n[A] K_stripped 联合 SVD")
    results_A[layer] = {}
    for r in R_VALUES:
        V_S, V_T, mS, mT = find_shared_basis(KA_train_s, KB_train_s, r)
        r2_tr = eval_basis(KA_train_s, KB_train_s, V_S, V_T, mS, mT)
        r2_te = eval_basis(KA_test_s,  KB_test_s,  V_S, V_T, mS, mT)
        results_A[layer][r] = (r2_tr, r2_te)
        print(f"  r={r:>4}: train R²={r2_tr:.4f}, test R²={r2_te:.4f}")

    # ---------- B: CCA ----------
    print(f"\n[B] CCA 典型相关系数")
    results_B[layer] = {}

    # K_stripped
    Cxx, Cyy, Cxy, _, _ = compute_covariances(KA_train_s, KB_train_s)
    spec_K = cca_spectrum(Cxx, Cyy, Cxy)
    results_B[layer]["K_stripped"] = spec_K[:max(CCA_COMPONENTS)].tolist()

    # V
    Cxx, Cyy, Cxy, _, _ = compute_covariances(VA_train, VB_train)
    spec_V = cca_spectrum(Cxx, Cyy, Cxy)
    results_B[layer]["V"] = spec_V[:max(CCA_COMPONENTS)].tolist()

    print("  K_stripped 前 8 个典型相关系数:")
    print("    " + "  ".join(f"{v:.4f}" for v in spec_K[:8]))
    print("  V 前 8 个典型相关系数:")
    print("    " + "  ".join(f"{v:.4f}" for v in spec_V[:8]))

    # 释放
    del KA_train, VA_train, KB_train, VB_train
    del KA_test, VA_test, KB_test, VB_test
    del KA_train_s, KB_train_s, KA_test_s, KB_test_s
    torch.cuda.empty_cache()

# ============================================================
# 汇总 A
# ============================================================
print(f"\n{'='*80}")
print("[A] K_stripped 联合 SVD: test R²（train 拟合，test 评估）")
print(f"{'='*80}")
header = f"{'Layer':>6}  " + "  ".join(f"r={r:<5}" for r in R_VALUES)
print(header)
print("-" * len(header))
for layer in LAYERS_TO_TEST:
    row = f"{layer:>6}  "
    for r in R_VALUES:
        _, r2 = results_A[layer][r]
        row += f"{r2:>7.4f}  "
    print(row)

# ============================================================
# 汇总 B
# ============================================================
print(f"\n{'='*80}")
print("[B] CCA 典型相关系数（train 拟合）")
print(f"{'='*80}")

for feat in ["K_stripped", "V"]:
    print(f"\nFeature: {feat}")
    header = f"{'Layer':>6}  " + "  ".join(f"cc{i+1:<3}" for i in range(8))
    print(header)
    print("-" * len(header))
    for layer in LAYERS_TO_TEST:
        spec = results_B[layer][feat]
        row = f"{layer:>6}  " + "  ".join(f"{v:>5.3f}" for v in spec[:8])
        print(row)