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
def fit_ridge_batched(X, Y, idx=None, lam=100.0, batch_size=RIDGE_BATCH):
    """用 idx 指定的样本拟合 Ridge。idx=None 则用全部。"""
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
def attention_batched(Q, K, V, batch_size=ATTN_BATCH):
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

# 评估线性 V
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


# 评估 MLP V (test)
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
# 端到端评估
# ============================================================
print("\n" + "=" * 70)
print(f"End-to-End Comparison (Layer {LAYER})")
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

del O_real, K_B_rope
gc.collect()
torch.cuda.empty_cache()


# ============================================================
# 三种映射组合
# ============================================================
print("\n计算映射后的 attention...")

# 1. Global W_K + Linear W_V
K_lin = apply_W(K_A, W_K_global)
V_lin = apply_W(V_A, W_V_global)

# 2. Global W_K + MLP V
K_mlp = K_lin   # K 用同一份
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

# 3. Identity（对照）
K_id = K_A
V_id = V_A


# 逐个评估
def eval_transfer(name, K_pre, V_pre):
    print(f"\n评估 {name}...")

    K_r2 = r2_score(K_pre, K_B.float())
    V_r2 = r2_score(V_pre, V_B.float())

    K_pre_ = K_pre.view(B, T, n_kv, head_dim).transpose(1, 2)
    V_pre_ = V_pre.view(B, T, n_kv, head_dim).transpose(1, 2)
    K_pre_rope = apply_rotary_batched(K_pre_, cos, sin)

    O_t, A_t = attention_batched(Q_B_, K_pre_rope, V_pre_.cpu())
    O_t_flat = O_t.transpose(1, 2).reshape(B, T, -1)

    ac = cos_sim_batched(A_real, A_t)
    oc = cos_sim_batched(O_real_flat, O_t_flat)

    del K_pre_, V_pre_, K_pre_rope, O_t, A_t, O_t_flat
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "method": name,
        "K_r2": K_r2,
        "V_r2": V_r2,
        "attn_cos": ac,
        "out_cos": oc,
    }


results = []
results.append(eval_transfer("Identity", K_id, V_id))
results.append(eval_transfer("Global W_K + Linear V", K_lin, V_lin))
results.append(eval_transfer("Global W_K + MLP V", K_mlp, V_mlp_full))


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