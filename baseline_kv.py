import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
DATA_PATH = "../dataset/wikitext-02/train.jsonl"

MAX_ARTICLES = 9007
MAX_TOKENS_PER_ART = 256
LAYER = 14
DEVICE = "cuda"
SEED = 42

RIDGE_BATCH = 50000

torch.manual_seed(SEED)


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

max_len = max(len(x) for x in article_ids)
padded = torch.zeros(len(article_ids), max_len, dtype=torch.long)
for i, ids in enumerate(article_ids):
    padded[i, :len(ids)] = ids

B = len(padded)
T = max_len
print(f"文章数: {B}, 每篇最多 {T} token, 总 token: {B * T}")


# ============================================================
# 工具
# ============================================================
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x, cos, sin):
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin


def apply_rotary_batched(K, cos, sin, batch_size=16):
    outs = []
    B_ = K.shape[0]
    for s in range(0, B_, batch_size):
        e = min(s + batch_size, B_)
        k_b = K[s:e].to(DEVICE)
        c_b = cos[s:e].to(DEVICE)
        s_b = sin[s:e].to(DEVICE)
        outs.append(apply_rotary(k_b, c_b, s_b).cpu())
        del k_b, c_b, s_b
        torch.cuda.empty_cache()
    return torch.cat(outs, 0)


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B_, H, T_, D = x.shape
    return x[:, :, None, :, :].expand(B_, H, n_rep, T_, D).reshape(B_, H * n_rep, T_, D)


def cos_sim_batched(A, B_, batch_size=64):
    A = A.reshape(A.shape[0], -1)
    B_ = B_.reshape(B_.shape[0], -1)
    N = A.shape[0]
    dot = 0.0; nA = 0.0; nB = 0.0
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        a = A[s:e].double(); b = B_[s:e].double()
        dot += (a * b).sum().item()
        nA += (a ** 2).sum().item()
        nB += (b ** 2).sum().item()
        del a, b
    return dot / (np.sqrt(nA) * np.sqrt(nB) + 1e-12)


def r2_score(pred, target):
    pred = pred.float()
    target = target.float()
    return (1 - ((pred - target)**2).sum() / (target**2).sum()).item()


# ============================================================
# Ridge
# ============================================================
def fit_ridge(X, Y, lam=100.0, batch_size=RIDGE_BATCH):
    D_X = X.shape[-1]
    D_Y = Y.shape[-1]
    XtX = torch.zeros(D_X, D_X, dtype=torch.float32)
    XtY = torch.zeros(D_X, D_Y, dtype=torch.float32)
    N = X.shape[0]
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        x = X[s:e].float(); y = Y[s:e].float()
        XtX += x.T @ x
        XtY += x.T @ y
    return torch.linalg.solve(XtX + lam * torch.eye(D_X), XtY)


def apply_W(X, W, batch_size=RIDGE_BATCH):
    N = X.shape[0]
    outs = []
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        outs.append((X[s:e].float() @ W).cpu())
    return torch.cat(outs, 0)


# ============================================================
# 提取 Q/K/V（单层）
# ============================================================
@torch.no_grad()
def extract_qkv(model, layer_idx, padded, need_q=True):
    attn = model.model.layers[layer_idx].self_attn
    n_q = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // n_q)

    all_q, all_k, all_v = [], [], []
    batch_size = 64

    for start in range(0, len(padded), batch_size):
        batch = padded[start:start+batch_size].to(DEVICE)
        B_b, T_b = batch.shape

        out = model.model(input_ids=batch, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[layer_idx]
        normed = model.model.layers[layer_idx].input_layernorm(hidden)

        k = attn.k_proj(normed).view(B_b, T_b, n_kv, head_dim)
        k = attn.k_norm(k).transpose(1, 2)
        all_k.append(k.transpose(1, 2).reshape(B_b, T_b, -1).float().cpu())

        v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
        all_v.append(v.transpose(1, 2).reshape(B_b, T_b, -1).float().cpu())

        if need_q:
            q = attn.q_proj(normed).view(B_b, T_b, n_q, head_dim)
            q = attn.q_norm(q).transpose(1, 2)
            pos_ids = torch.arange(T_b).unsqueeze(0).expand(B_b, -1).cpu()
            cos, sin = model.model.rotary_emb(q.cpu(), pos_ids)
            cos, sin = cos.to(DEVICE), sin.to(DEVICE)
            q = apply_rotary(q, cos, sin)
            all_q.append(q.transpose(1, 2).reshape(B_b, T_b, -1).float().cpu())

        del out, hidden, normed
        torch.cuda.empty_cache()

    def cat(x):
        if not x:
            return None
        return torch.cat(x, dim=0).reshape(-1, x[0].shape[-1])

    return cat(all_q), cat(all_k), cat(all_v)


# ============================================================
# Attention
# ============================================================
def attention_single(Q, K, V):
    n_rep = Q.shape[1] // K.shape[1]
    K = repeat_kv(K, n_rep)
    V = repeat_kv(V, n_rep)
    scale = 1.0 / (Q.shape[-1] ** 0.5)
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
    Tq, Tk = scores.shape[-2], scores.shape[-1]
    mask = torch.triu(torch.ones(Tq, Tk, device=Q.device, dtype=torch.bool), 1)
    scores = scores.masked_fill(mask[None, None], torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, V), probs


@torch.no_grad()
def attention_batched(Q, K, V, batch_size=16):
    B_ = Q.shape[0]
    outs, probs_list = [], []
    for s in range(0, B_, batch_size):
        e = min(s + batch_size, B_)
        q = Q[s:e].to(DEVICE).float()
        k = K[s:e].to(DEVICE).float()
        v = V[s:e].to(DEVICE).float()
        o, p = attention_single(q, k, v)
        outs.append(o.cpu()); probs_list.append(p.cpu())
        del q, k, v, o, p
        torch.cuda.empty_cache()
    return torch.cat(outs, 0), torch.cat(probs_list, 0)


# ============================================================
# 加载模型
# ============================================================
print("\n加载模型...")
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()

model_A.model.rotary_emb.inv_freq = model_A.model.rotary_emb.inv_freq.to(DEVICE)
model_B.model.rotary_emb.inv_freq = model_B.model.rotary_emb.inv_freq.to(DEVICE)

n_q = model_B.config.num_attention_heads
n_kv = model_B.config.num_key_value_heads
head_dim = getattr(model_B.config, "head_dim",
                   model_B.config.hidden_size // n_q)


# ============================================================
# 提取
# ============================================================
print(f"\n提取第 {LAYER} 层 Q/K/V...")
Q_B, K_B, V_B = extract_qkv(model_B, LAYER, padded, need_q=True)
_,  K_A, V_A = extract_qkv(model_A, LAYER, padded, need_q=False)

print(f"  Q_B: {Q_B.shape}")
print(f"  K_A: {K_A.shape}, V_A: {V_A.shape}")

N = V_A.shape[0]
D_K = K_A.shape[1]
D_V = V_A.shape[1]


# ============================================================
# 训练所有映射
# ============================================================
print("\n训练映射 W...")

# Global W
W_K_global = fit_ridge(K_A, K_B, lam=100.0)
W_V_global = fit_ridge(V_A, V_B, lam=100.0)

# Random W (orthogonal)
def random_orthogonal(d):
    A = torch.randn(d, d)
    Q_r, _ = torch.linalg.qr(A)
    return Q_r

W_K_rand = random_orthogonal(D_K)
W_V_rand = random_orthogonal(D_V)

# Layer-matched W
# 假设层号相同，用同一层的 W。但我们现在只有 Layer 14 的数据，
# 所以 layer-matched = 用 Layer 14 训练的 W。
# 严格来说需要每层单独训练，这里用同一个 W（因为只测了一层）
W_K_layer = W_K_global   # 单层测试下和 global 相同
W_V_layer = W_V_global

print(f"  W_K_global: {W_K_global.shape}")
print(f"  W_V_global: {W_V_global.shape}")


# ============================================================
# 端到端评估四路 baseline
# ============================================================
print("\n" + "=" * 70)
print(f"Baseline Comparison (Layer {LAYER})")
print("=" * 70)

# reshape
Q_B_ = Q_B.view(B, T, n_q, head_dim).transpose(1, 2)
K_B_ = K_B.view(B, T, n_kv, head_dim).transpose(1, 2)
V_B_ = V_B.view(B, T, n_kv, head_dim).transpose(1, 2)

# RoPE for real K
pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1).cpu()
cos, sin = model_B.model.rotary_emb(K_B_.cpu(), pos_ids)
cos = cos.to(DEVICE); sin = sin.to(DEVICE)
K_B_rope = apply_rotary_batched(K_B_, cos, sin)

# Real attention
print("\n计算真实 attention...")
O_real, A_real = attention_batched(Q_B_, K_B_rope, V_B_)
O_real_flat = O_real.transpose(1, 2).reshape(B, T, -1)


# ---- 四种方法 ----
methods = {}

# 1. Identity（不做映射）
K_id = K_A.view(B, T, n_kv, head_dim).transpose(1, 2)
V_id = V_A.view(B, T, n_kv, head_dim).transpose(1, 2)
K_id_rope = apply_rotary_batched(K_id, cos, sin)
methods["Identity"] = (K_id_rope, V_id.cpu(), K_id, V_id)

# 2. Random W
K_rand = apply_W(K_A, W_K_rand).view(B, T, n_kv, head_dim).transpose(1, 2)
V_rand = apply_W(V_A, W_V_rand).view(B, T, n_kv, head_dim).transpose(1, 2)
K_rand_rope = apply_rotary_batched(K_rand, cos, sin)
methods["Random W"] = (K_rand_rope, V_rand.cpu(), K_rand, V_rand)

# 3. Layer-matched W（单层下等同 Global，但保留标签）
K_layer = apply_W(K_A, W_K_layer).view(B, T, n_kv, head_dim).transpose(1, 2)
V_layer = apply_W(V_A, W_V_layer).view(B, T, n_kv, head_dim).transpose(1, 2)
K_layer_rope = apply_rotary_batched(K_layer, cos, sin)
methods["Layer-matched W"] = (K_layer_rope, V_layer.cpu(), K_layer, V_layer)

# 4. Global W
K_glob = apply_W(K_A, W_K_global).view(B, T, n_kv, head_dim).transpose(1, 2)
V_glob = apply_W(V_A, W_V_global).view(B, T, n_kv, head_dim).transpose(1, 2)
K_glob_rope = apply_rotary_batched(K_glob, cos, sin)
methods["Global W (ours)"] = (K_glob_rope, V_glob.cpu(), K_glob, V_glob)


# ---- 计算 ----
print("\n" + "=" * 70)
print(f"{'Method':>22}  {'K R²':>8}  {'V R²':>8}  {'AttnCos':>9}  {'OutCos':>9}")
print("-" * 70)

results = []
for name, (K_rope, V_cpu, K_pre, V_pre) in methods.items():
    print(f"\n计算 {name}...")

    # R²
    if name == "Identity":
        K_r2 = r2_score(K_pre.reshape(-1, D_K), K_B.float())
        V_r2 = r2_score(V_pre.reshape(-1, D_V), V_B.float())
    else:
        K_r2 = r2_score(K_pre.reshape(-1, D_K), K_B.float())
        V_r2 = r2_score(V_pre.reshape(-1, D_V), V_B.float())

    # attention
    O_t, A_t = attention_batched(Q_B_, K_rope, V_cpu)
    O_t_flat = O_t.transpose(1, 2).reshape(B, T, -1)

    ac = cos_sim_batched(A_real, A_t)
    oc = cos_sim_batched(O_real_flat, O_t_flat)

    results.append({
        "method": name,
        "K_r2": K_r2,
        "V_r2": V_r2,
        "attn_cos": ac,
        "out_cos": oc,
    })

    # 清理
    del O_t, A_t, O_t_flat
    torch.cuda.empty_cache()


# ---- 汇总 ----
print()
print("=" * 70)
print(f"Baseline Comparison Summary (Layer {LAYER})")
print("=" * 70)
print(f"{'Method':>22}  {'K R²':>8}  {'V R²':>8}  {'AttnCos':>9}  {'OutCos':>9}")
print("-" * 70)
for r in results:
    print(f"{r['method']:>22}  {r['K_r2']:8.4f}  {r['V_r2']:8.4f}  "
          f"{r['attn_cos']:9.4f}  {r['out_cos']:9.4f}")
print("=" * 70)