import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
ATTN_BATCH = 16
MLP_EPOCHS = 50
MLP_LR = 1e-3
MLP_BATCH = 256
TRAIN_RATIO = 0.8

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


def r2_score(pred, target):
    pred = pred.float()
    target = target.float()
    return (1 - ((pred - target)**2).sum() / (target**2).sum()).item()


# ============================================================
# Ridge
# ============================================================
def fit_ridge_batched(X, Y, idx=None, lam=100.0, batch_size=RIDGE_BATCH):
    D_X = X.shape[-1]
    D_Y = Y.shape[-1]
    XtX = torch.zeros(D_X, D_X, dtype=torch.float32)
    XtY = torch.zeros(D_X, D_Y, dtype=torch.float32)

    if idx is None:
        N = X.shape[0]
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            x = X[s:e].float(); y = Y[s:e].float()
            XtX += x.T @ x
            XtY += x.T @ y
    else:
        N = len(idx)
        for s in range(0, N, batch_size):
            e = min(s + batch_size, N)
            batch_idx = idx[s:e]
            x = X[batch_idx].float(); y = Y[batch_idx].float()
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
# MLP
# ============================================================
class VMapper(torch.nn.Module):
    def __init__(self, d_in=1024, d_out=1024, hidden=2048):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, d_out),
        )
    def forward(self, x):
        return self.net(x)


# ============================================================
# 提取 Q/K/V
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
            q = q_norm_safe(attn, q).transpose(1, 2)
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


def q_norm_safe(attn, q):
    if hasattr(attn, "q_norm") and attn.q_norm is not None:
        return attn.q_norm(q)
    return q


# ============================================================
# Attention（单 batch）
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


# ============================================================
# 流式 cosine：不保留全量 A/O
# ============================================================
@torch.no_grad()
def attention_streaming_cos(Q, K_ref, K_tgt, V_ref, V_tgt, batch_size=ATTN_BATCH):
    """
    对每个 batch 算两路 attention，累积 cos 的中间量。
    不保留任何全量 O 或 A。
    返回 (cos_output, cos_attn)。
    """
    B_ = Q.shape[0]

    dot_O = 0.0; nO_r = 0.0; nO_t = 0.0
    dot_A = 0.0; nA_r = 0.0; nA_t = 0.0

    for s in range(0, B_, batch_size):
        e = min(s + batch_size, B_)
        q = Q[s:e].to(DEVICE).float()
        k_r = K_ref[s:e].to(DEVICE).float()
        k_t = K_tgt[s:e].to(DEVICE).float()
        v_r = V_ref[s:e].to(DEVICE).float()
        v_t = V_tgt[s:e].to(DEVICE).float()

        O_r, A_r = attention_single(q, k_r, v_r)
        O_t, A_t = attention_single(q, k_t, v_t)

        # 展平 + double
        O_r_f = O_r.reshape(O_r.shape[0], -1).double()
        O_t_f = O_t.reshape(O_t.shape[0], -1).double()
        A_r_f = A_r.reshape(A_r.shape[0], -1).double()
        A_t_f = A_t.reshape(A_t.shape[0], -1).double()

        # 累积
        dot_O += (O_r_f * O_t_f).sum().item()
        nO_r += (O_r_f ** 2).sum().item()
        nO_t += (O_t_f ** 2).sum().item()

        dot_A += (A_r_f * A_t_f).sum().item()
        nA_r += (A_r_f ** 2).sum().item()
        nA_t += (A_t_f ** 2).sum().item()

        del q, k_r, k_t, v_r, v_t, O_r, A_r, O_t, A_t
        del O_r_f, O_t_f, A_r_f, A_t_f
        torch.cuda.empty_cache()

    cos_O = dot_O / (np.sqrt(nO_r) * np.sqrt(nO_t) + 1e-12)
    cos_A = dot_A / (np.sqrt(nA_r) * np.sqrt(nA_t) + 1e-12)
    return cos_O, cos_A


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

# train/test 划分
perm = torch.randperm(N)
n_train = int(N * TRAIN_RATIO)
train_idx = perm[:n_train]
test_idx = perm[n_train:]
print(f"  train: {len(train_idx)}, test: {len(test_idx)}")


# ============================================================
# 训练映射
# ============================================================
print("\n训练 W_K (global linear)...")
W_K_global = fit_ridge_batched(K_A, K_B, idx=train_idx, lam=100.0)

print("训练 W_V (global linear)...")
W_V_global = fit_ridge_batched(V_A, V_B, idx=train_idx, lam=100.0)

V_linear_test = apply_W(V_A[test_idx], W_V_global)
V_linear_r2 = r2_score(V_linear_test, V_B[test_idx])
print(f"  Linear W_V test R² = {V_linear_r2:.4f}")


# ============================================================
# 训练 V MLP
# ============================================================
print(f"\n训练 V MLP ({MLP_EPOCHS} epochs)...")

mapper = VMapper(D_V, D_V, hidden=2048).to(DEVICE)
opt = torch.optim.Adam(mapper.parameters(), lr=MLP_LR)
loss_fn = torch.nn.MSELoss()

n_train_n = len(train_idx)

for epoch in range(MLP_EPOCHS):
    mapper.train()
    perm_e = torch.randperm(n_train_n)
    total_loss = 0.0
    n_steps = 0

    for s in range(0, n_train_n, MLP_BATCH):
        e = min(s + MLP_BATCH, n_train_n)
        batch_orig_idx = train_idx[perm_e[s:e]]

        x = V_A[batch_orig_idx].to(DEVICE)
        y = V_B[batch_orig_idx].to(DEVICE)

        pred = mapper(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        n_steps += 1

        del x, y, pred, loss
        if n_steps % 100 == 0:
            torch.cuda.empty_cache()

    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1:3d}  loss = {total_loss/n_steps:.6f}")


# 评估 MLP
mapper.eval()
with torch.no_grad():
    V_mlp_test = []
    for s in range(0, len(test_idx), 4096):
        e = min(s + 4096, len(test_idx))
        idx = test_idx[s:e]
        x = V_A[idx].to(DEVICE)
        V_mlp_test.append(mapper(x).cpu())
        del x
        torch.cuda.empty_cache()
    V_mlp_test = torch.cat(V_mlp_test, 0)

V_mlp_r2 = r2_score(V_mlp_test, V_B[test_idx])
print(f"  MLP V test R² = {V_mlp_r2:.4f}")


# ============================================================
# 准备映射后的 K/V（全量）
# ============================================================
print("\n准备映射后的 K/V...")

# K 用 global linear
K_lin = apply_W(K_A, W_K_global)

# V linear
V_lin = apply_W(V_A, W_V_global)

# V MLP（全量）
mapper.eval()
with torch.no_grad():
    V_mlp_full = []
    for s in range(0, N, 4096):
        e = min(s + 4096, N)
        x = V_A[s:e].to(DEVICE)
        V_mlp_full.append(mapper(x).cpu())
        del x
        torch.cuda.empty_cache()
    V_mlp_full = torch.cat(V_mlp_full, 0)


# ============================================================
# reshape + RoPE
# ============================================================
Q_B_ = Q_B.view(B, T, n_q, head_dim).transpose(1, 2)
K_B_ = K_B.view(B, T, n_kv, head_dim).transpose(1, 2)
V_B_ = V_B.view(B, T, n_kv, head_dim).transpose(1, 2)

K_lin_ = K_lin.view(B, T, n_kv, head_dim).transpose(1, 2)
V_lin_ = V_lin.view(B, T, n_kv, head_dim).transpose(1, 2)
V_mlp_ = V_mlp_full.view(B, T, n_kv, head_dim).transpose(1, 2)

K_id_ = K_A.view(B, T, n_kv, head_dim).transpose(1, 2)
V_id_ = V_A.view(B, T, n_kv, head_dim).transpose(1, 2)

# RoPE
pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1).cpu()
cos, sin = model_B.model.rotary_emb(K_B_.cpu(), pos_ids)
cos = cos.to(DEVICE); sin = sin.to(DEVICE)

K_B_rope = apply_rotary_batched(K_B_, cos, sin)
K_lin_rope = apply_rotary_batched(K_lin_, cos, sin)
K_id_rope = apply_rotary_batched(K_id_, cos, sin)

del K_B_, K_lin_, K_id_
gc.collect()
torch.cuda.empty_cache()


# ============================================================
# 流式端到端评估
# ============================================================
print("\n" + "=" * 75)
print(f"End-to-End Summary (Layer {LAYER})")
print("=" * 75)

results = []

# 1. Identity
print("\n[1/3] Identity...")
cos_O, cos_A = attention_streaming_cos(
    Q_B_, K_B_rope, K_id_rope, V_B_, V_id_
)
results.append({
    "method": "Identity",
    "K_r2": r2_score(K_A, K_B.float()),
    "V_r2": r2_score(V_A, V_B.float()),
    "attn_cos": cos_A,
    "out_cos": cos_O,
})
print(f"  AttnCos = {cos_A:.4f}, OutCos = {cos_O:.4f}")

del K_id_rope, V_id_
gc.collect()
torch.cuda.empty_cache()

# 2. Global W_K + Linear V
print("\n[2/3] Global W_K + Linear V...")
cos_O, cos_A = attention_streaming_cos(
    Q_B_, K_B_rope, K_lin_rope, V_B_, V_lin_
)
results.append({
    "method": "Global W_K + Linear V",
    "K_r2": r2_score(K_lin, K_B.float()),
    "V_r2": r2_score(V_lin, V_B.float()),
    "attn_cos": cos_A,
    "out_cos": cos_O,
})
print(f"  AttnCos = {cos_A:.4f}, OutCos = {cos_O:.4f}")

# 3. Global W_K + MLP V
print("\n[3/3] Global W_K + MLP V...")
cos_O, cos_A = attention_streaming_cos(
    Q_B_, K_B_rope, K_lin_rope, V_B_, V_mlp_
)
results.append({
    "method": "Global W_K + MLP V",
    "K_r2": r2_score(K_lin, K_B.float()),
    "V_r2": r2_score(V_mlp_full, V_B.float()),
    "attn_cos": cos_A,
    "out_cos": cos_O,
})
print(f"  AttnCos = {cos_A:.4f}, OutCos = {cos_O:.4f}")


# ============================================================
# 汇总
# ============================================================
print()
print("=" * 75)
print(f"End-to-End Summary (Layer {LAYER})")
print("=" * 75)
print(f"{'Method':>28}  {'K R²':>8}  {'V R²':>8}  {'AttnCos':>9}  {'OutCos':>9}")
print("-" * 75)
for r in results:
    print(f"{r['method']:>28}  {r['K_r2']:8.4f}  {r['V_r2']:8.4f}  "
          f"{r['attn_cos']:9.4f}  {r['out_cos']:9.4f}")
print("=" * 75)