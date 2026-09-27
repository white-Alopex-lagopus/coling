# import os
# import time
# import sys
# sys.path.insert(0, '/home/baimingyu/workspace/gpu_monitor')

# from gpu_detector import GPUDetector, wait_for_gpu

# gpu_id = wait_for_gpu(gpu_ids=[0, 1])  # 无限等待
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


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

MAX_ARTICLES = 1981
MAX_TOKENS_PER_ART = 256
LAYER = 14
DEVICE = "cuda"
SEED = 42

RIDGE_BATCH = 50000
TRAIN_RATIO = 0.8

MLP_EPOCHS = 200
MLP_LR = 1e-3
MLP_BATCH = 256
MLP_HIDDEN = 512      # 2048 → 512
PATIENCE = 10

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
N = B * T
print(f"文章数: {B}, 每篇最多 {T} token, 总 token: {N}")


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
def fit_ridge(X, Y, lam=100.0, batch_size=RIDGE_BATCH):
    D_X = X.shape[-1]
    D_Y = Y.shape[-1]
    XtX = torch.zeros(D_X, D_X, dtype=torch.float32)
    XtY = torch.zeros(D_X, D_Y, dtype=torch.float32)
    N_ = X.shape[0]
    for s in range(0, N_, batch_size):
        e = min(s + batch_size, N_)
        x = X[s:e].float(); y = Y[s:e].float()
        XtX += x.T @ x
        XtY += x.T @ y
    return torch.linalg.solve(XtX + lam * torch.eye(D_X), XtY)


def apply_W(X, W, batch_size=RIDGE_BATCH):
    N_ = X.shape[0]
    outs = []
    for s in range(0, N_, batch_size):
        e = min(s + batch_size, N_)
        outs.append((X[s:e].float() @ W).cpu())
    return torch.cat(outs, 0)


# ============================================================
# MLP
# ============================================================
class VMapper(torch.nn.Module):
    def __init__(self, d_in, d_out, hidden):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, d_out),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
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
        all_k.append(k.transpose(1, 2).reshape(B_b, T_b, -1).to(torch.float16).cpu())

        v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
        all_v.append(v.transpose(1, 2).reshape(B_b, T_b, -1).to(torch.float16).cpu())

        if need_q:
            q = attn.q_proj(normed).view(B_b, T_b, n_q, head_dim)
            q = attn.q_norm(q).transpose(1, 2)
            pos_ids = torch.arange(T_b).unsqueeze(0).expand(B_b, -1).cpu()
            cos, sin = model.model.rotary_emb(q.cpu(), pos_ids)
            cos, sin = cos.to(DEVICE), sin.to(DEVICE)
            q = apply_rotary(q, cos, sin)
            all_q.append(q.transpose(1, 2).reshape(B_b, T_b, -1).to(torch.float16).cpu())

        del out, hidden, normed
        torch.cuda.empty_cache()

    def cat(x):
        if not x:
            return None
        return torch.cat(x, dim=0).reshape(-1, x[0].shape[-1])

    return cat(all_q), cat(all_k), cat(all_v)


# ============================================================
# Attention（流式 cos）
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


def _accumulate(a, O_r, A_r, O_t, A_t):
    O_r_f = O_r.reshape(O_r.shape[0], -1).double()
    O_t_f = O_t.reshape(O_t.shape[0], -1).double()
    A_r_f = A_r.reshape(A_r.shape[0], -1).double()
    A_t_f = A_t.reshape(A_t.shape[0], -1).double()
    a["dO"] += (O_r_f * O_t_f).sum().item()
    a["nOr"] += (O_r_f ** 2).sum().item()
    a["nOt"] += (O_t_f ** 2).sum().item()
    a["dA"] += (A_r_f * A_t_f).sum().item()
    a["nAr"] += (A_r_f ** 2).sum().item()
    a["nAt"] += (A_t_f ** 2).sum().item()


@torch.no_grad()
def streaming_cos_layer(Q_B_rope, K_A_rope, K_lin_rope,
                        V_A, V_lin, V_mlp_cos,
                        K_B_rope, V_B,
                        B, T, DEVICE, batch_articles=16):
    """
    输入都是 [B, H, T, D]。
    Q/K 已经加过 RoPE。
    """
    acc = {}
    for name in ["Identity", "Linear", "MLP_cos"]:
        acc[name] = {"dO": 0.0, "nOr": 0.0, "nOt": 0.0,
                     "dA": 0.0, "nAr": 0.0, "nAt": 0.0}

    for b_start in range(0, B, batch_articles):
        b_end = min(b_start + batch_articles, B)

        q = Q_B_rope[b_start:b_end].to(DEVICE).float()
        k_ref = K_B_rope[b_start:b_end].to(DEVICE).float()
        v_ref = V_B[b_start:b_end].to(DEVICE).float()

        O_r, A_r = attention_single(q, k_ref, v_ref)

        # Identity
        k_id = K_A_rope[b_start:b_end].to(DEVICE).float()
        v_id = V_A[b_start:b_end].to(DEVICE).float()
        O_t, A_t = attention_single(q, k_id, v_id)
        _accumulate(acc["Identity"], O_r, A_r, O_t, A_t)
        del k_id, v_id, O_t, A_t

        # Linear V
        k_lin = K_lin_rope[b_start:b_end].to(DEVICE).float()
        v_lin = V_lin[b_start:b_end].to(DEVICE).float()
        O_t, A_t = attention_single(q, k_lin, v_lin)
        _accumulate(acc["Linear"], O_r, A_r, O_t, A_t)
        del v_lin, O_t, A_t

        # MLP_cos V
        if V_mlp_cos is not None:
            v_mlp = V_mlp_cos[b_start:b_end].to(DEVICE).float()
            O_t, A_t = attention_single(q, k_lin, v_mlp)
            _accumulate(acc["MLP_cos"], O_r, A_r, O_t, A_t)
            del v_mlp, O_t, A_t

        del q, k_ref, v_ref, O_r, A_r, k_lin
        torch.cuda.empty_cache()

    out = {}
    for name, a in acc.items():
        cos_O = a["dO"] / (np.sqrt(a["nOr"]) * np.sqrt(a["nOt"]) + 1e-12)
        cos_A = a["dA"] / (np.sqrt(a["nAr"]) * np.sqrt(a["nAt"]) + 1e-12)
        out[name] = (cos_O, cos_A)
    return out


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
D = V_A.shape[1]

perm = torch.randperm(N)
n_train = int(N * TRAIN_RATIO)
train_idx = perm[:n_train]
test_idx = perm[n_train:]
print(f"  train: {len(train_idx)}, test: {len(test_idx)}")


# ============================================================
# 训练 Ridge
# ============================================================
print("\n训练 W_K, W_V (Ridge)...")
W_K = fit_ridge(K_A[train_idx], K_B[train_idx], lam=100.0)
W_V = fit_ridge(V_A[train_idx], V_B[train_idx], lam=100.0)

K_lin = apply_W(K_A, W_K)
V_lin = apply_W(V_A, W_V)

K_r2 = r2_score(K_lin, K_B.float())
V_lin_r2 = r2_score(V_lin[test_idx], V_B[test_idx].float())
print(f"  K R² = {K_r2:.4f}, Linear V R² = {V_lin_r2:.4f}")


# ============================================================
# 训练 MLP (cosine loss + val + early stop)
# ============================================================
print(f"\n训练 MLP (cosine loss, hidden={MLP_HIDDEN}, max {MLP_EPOCHS} epochs)...")

# 从 train_idx 里再划 10% 作为 val
n_val = int(len(train_idx) * 0.1)
val_idx = train_idx[:n_val]
tr_idx = train_idx[n_val:]
print(f"  MLP train: {len(tr_idx)}, MLP val: {len(val_idx)}")

mapper_cos = VMapper(D, D, hidden=MLP_HIDDEN).to(DEVICE)
opt = torch.optim.Adam(mapper_cos.parameters(), lr=MLP_LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MLP_EPOCHS)

best_val_loss = float('inf')
best_state = None
wait = 0

for epoch in range(MLP_EPOCHS):
    # ---- train ----
    mapper_cos.train()
    perm_e = torch.randperm(len(tr_idx))
    train_loss = 0.0
    n_steps = 0

    for s in range(0, len(tr_idx), MLP_BATCH):
        e = min(s + MLP_BATCH, len(tr_idx))
        batch_orig_idx = tr_idx[perm_e[s:e]]

        x = V_A[batch_orig_idx].to(DEVICE).float()
        y = V_B[batch_orig_idx].to(DEVICE).float()

        pred = mapper_cos(x)
        cos = F.cosine_similarity(pred, y, dim=-1)
        loss = F.mse_loss(pred, y) + 0.5 * (1 - cos).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss.item()
        n_steps += 1

        del x, y, pred, loss, cos
        if n_steps % 100 == 0:
            torch.cuda.empty_cache()

    train_loss /= n_steps

    # ---- val ----
    mapper_cos.eval()
    val_loss = 0.0
    n_val_steps = 0
    with torch.no_grad():
        for s in range(0, len(val_idx), MLP_BATCH):
            e = min(s + MLP_BATCH, len(val_idx))
            batch_orig_idx = val_idx[s:e]

            x = V_A[batch_orig_idx].to(DEVICE).float()
            y = V_B[batch_orig_idx].to(DEVICE).float()

            pred = mapper_cos(x)
            cos = F.cosine_similarity(pred, y, dim=-1)
            loss = F.mse_loss(pred, y) + 0.5 * (1 - cos).mean()

            val_loss += loss.item()
            n_val_steps += 1

            del x, y, pred, loss, cos
    val_loss /= n_val_steps

    scheduler.step()

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  epoch {epoch+1:3d}  train={train_loss:.6f}  val={val_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

    # ---- early stopping ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.clone() for k, v in mapper_cos.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"  Early stop at epoch {epoch+1}, best val={best_val_loss:.6f}")
            break

# 恢复最优
if best_state is not None:
    mapper_cos.load_state_dict(best_state)

print(f"  Best val cos_loss = {best_val_loss:.6f}")


# 全量预测
mapper_cos.eval()
with torch.no_grad():
    V_mlp_cos = []
    for s in range(0, N, 4096):
        e = min(s + 4096, N)
        x = V_A[s:e].to(DEVICE).float()
        V_mlp_cos.append(mapper_cos(x).cpu())
        del x
        torch.cuda.empty_cache()
    V_mlp_cos = torch.cat(V_mlp_cos, 0)

V_mlp_cos_r2 = r2_score(V_mlp_cos[test_idx], V_B[test_idx].float())
print(f"  MLP_cos V R² = {V_mlp_cos_r2:.4f}")


# ============================================================
# 端到端评估
# ============================================================
print("\n准备端到端评估...")

Q_B_ = Q_B.view(B, T, n_q, head_dim).transpose(1, 2)
K_B_ = K_B.view(B, T, n_kv, head_dim).transpose(1, 2)
V_B_ = V_B.view(B, T, n_kv, head_dim).transpose(1, 2)
K_A_ = K_A.view(B, T, n_kv, head_dim).transpose(1, 2)
V_A_ = V_A.view(B, T, n_kv, head_dim).transpose(1, 2)
K_lin_ = K_lin.view(B, T, n_kv, head_dim).transpose(1, 2)
V_lin_ = V_lin.view(B, T, n_kv, head_dim).transpose(1, 2)
V_mlp_cos_ = V_mlp_cos.view(B, T, n_kv, head_dim).transpose(1, 2)

# RoPE
pos_ids = torch.arange(T).unsqueeze(0).expand(B, -1).cpu()
_cos, _sin = model_B.model.rotary_emb(
    torch.zeros(B, T, head_dim), pos_ids
)
cos_full = _cos.float().cpu()
sin_full = _sin.float().cpu()


def apply_rotary_batched(K, cos_f, sin_f, batch_size=16):
    outs = []
    Bn = K.shape[0]
    for s in range(0, Bn, batch_size):
        e = min(s + batch_size, Bn)
        k_b = K[s:e].to(DEVICE)
        c_b = cos_f[s:e].to(DEVICE)
        s_b = sin_f[s:e].to(DEVICE)
        outs.append(apply_rotary(k_b, c_b, s_b).cpu())
        del k_b, c_b, s_b
        torch.cuda.empty_cache()
    return torch.cat(outs, 0)


K_B_rope = apply_rotary_batched(K_B_.cpu(), cos_full, sin_full)
K_A_rope = apply_rotary_batched(K_A_.cpu(), cos_full, sin_full)
K_lin_rope = apply_rotary_batched(K_lin_.cpu(), cos_full, sin_full)

print("\n流式端到端评估...")
out = streaming_cos_layer(
    Q_B_,
    K_A_rope, K_lin_rope,
    V_A_, V_lin_, V_mlp_cos_,
    K_B_rope, V_B_,
    B, T, DEVICE,
    batch_articles=16,
)


# ============================================================
# 汇总
# ============================================================
print()
print("=" * 80)
print(f"Cosine Loss + Hidden={MLP_HIDDEN} (Layer {LAYER})")
print("=" * 80)
print(f"{'Method':>20}  {'V R²':>10}  {'AttnCos':>10}  {'OutCos':>10}")
print("-" * 80)
print(f"{'Identity':>20}  {'-':>10}  {out['Identity'][1]:10.4f}  {out['Identity'][0]:10.4f}")
print(f"{'Linear W':>20}  {V_lin_r2:10.4f}  {out['Linear'][1]:10.4f}  {out['Linear'][0]:10.4f}")
print(f"{'MLP (cosine)':>20}  {V_mlp_cos_r2:10.4f}  {out['MLP_cos'][1]:10.4f}  {out['MLP_cos'][0]:10.4f}")
print("=" * 80)

print()
print("关键对比：")
print(f"  Linear V:      R²={V_lin_r2:.4f}  OutCos={out['Linear'][0]:.4f}")
print(f"  MLP (cosine):  R²={V_mlp_cos_r2:.4f}  OutCos={out['MLP_cos'][0]:.4f}")
print(f"  ΔOutCos:       {out['MLP_cos'][0] - out['Linear'][0]:+.4f}")