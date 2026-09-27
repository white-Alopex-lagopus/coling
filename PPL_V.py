import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置
# ============================================================
MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"
DATA_PATH = "../dataset/wikitext-02/train.jsonl"

# 训练 MLP 用
TRAIN_ARTICLES = 4000
# PPL 评估用
EVAL_ARTICLES = 400
MAX_TOKENS_PER_ART = 256

DEVICE = "cuda"
SEED = 42
RIDGE_BATCH = 50000
RIDGE_LAMBDA = 100.0

MLP_HIDDEN = 512
MLP_EPOCHS = 200
MLP_LR = 1e-3
MLP_BATCH = 256
MLP_PATIENCE = 10
MLP_VAL_RATIO = 0.1
MLP_LAMBDA_COS = 0.5

torch.manual_seed(SEED)


# ============================================================
# 数据
# ============================================================
df = pd.read_json(DATA_PATH, lines=True)
all_texts = df["text"].tolist()

train_texts = all_texts[:TRAIN_ARTICLES]
eval_texts = all_texts[TRAIN_ARTICLES:TRAIN_ARTICLES + EVAL_ARTICLES]

tok = AutoTokenizer.from_pretrained(MODEL_A)


def build_padded_with_mask(texts, max_len=MAX_TOKENS_PER_ART):
    ids_list = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        ids = ids[:max_len]
        if len(ids) >= 16:
            ids_list.append(ids)
    L = max(len(x) for x in ids_list)
    padded = torch.zeros(len(ids_list), L, dtype=torch.long)
    mask = torch.zeros(len(ids_list), L, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, :len(ids)] = ids
        mask[i, :len(ids)] = 1
    return padded, mask


train_padded, train_mask = build_padded_with_mask(train_texts)
eval_padded, eval_mask = build_padded_with_mask(eval_texts)

B_train, T_train = train_padded.shape
B_eval, T_eval = eval_padded.shape

print(f"Train: {B_train} 篇 × {T_train} token")
print(f"Eval : {B_eval} 篇 × {T_eval} token")


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
def fit_ridge(X, Y, lam=RIDGE_LAMBDA, batch_size=RIDGE_BATCH):
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


# ============================================================
# MLP
# ============================================================
class VMapper(torch.nn.Module):
    def __init__(self, d_in, d_out, hidden, residual=False):
        super().__init__()
        self.residual = residual
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, d_out),
        )
        if residual:
            torch.nn.init.zeros_(self.net[-1].weight)
            torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        if self.residual:
            return x + self.net(x)
        return self.net(x)


def train_mlp(mapper, X_all, Y_all, train_idx, val_idx,
              epochs=200, batch_size=256, lr=1e-3, patience=10,
              lambda_cos=0.5, device="cuda", log_every=10):
    mapper = mapper.to(device)
    opt = torch.optim.Adam(mapper.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        mapper.train()
        perm = torch.randperm(len(train_idx))
        tr_loss = 0.0; n_steps = 0

        for s in range(0, len(train_idx), batch_size):
            e = min(s + batch_size, len(train_idx))
            bidx = train_idx[perm[s:e]]
            x = X_all[bidx].to(device).float()
            y = Y_all[bidx].to(device).float()

            pred = mapper(x)
            mse = F.mse_loss(pred, y)
            cos = F.cosine_similarity(pred, y, dim=-1).mean()
            loss = mse + lambda_cos * (1 - cos)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += loss.item(); n_steps += 1
            del x, y, pred, mse, cos, loss
            if n_steps % 100 == 0:
                torch.cuda.empty_cache()
        tr_loss /= n_steps

        # val
        mapper.eval()
        val_loss = 0.0; n_val = 0
        with torch.no_grad():
            for s in range(0, len(val_idx), batch_size):
                e = min(s + batch_size, len(val_idx))
                bidx = val_idx[s:e]
                x = X_all[bidx].to(device).float()
                y = Y_all[bidx].to(device).float()
                pred = mapper(x)
                mse = F.mse_loss(pred, y)
                cos = F.cosine_similarity(pred, y, dim=-1).mean()
                loss = mse + lambda_cos * (1 - cos)
                val_loss += loss.item(); n_val += 1
                del x, y, pred, mse, cos, loss
        val_loss /= n_val
        scheduler.step()

        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"    epoch {epoch+1:3d}  train={tr_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in mapper.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop at epoch {epoch+1}, best_val={best_val:.6f}")
                break

    if best_state is not None:
        mapper.load_state_dict(best_state)
    print(f"    Best val = {best_val:.6f}")
    return mapper


@torch.no_grad()
def mlp_predict(mapper, X, batch_size=4096, device="cuda"):
    mapper.eval()
    outs = []
    for s in range(0, X.shape[0], batch_size):
        e = min(s + batch_size, X.shape[0])
        x = X[s:e].to(device).float()
        outs.append(mapper(x).cpu())
        del x
        torch.cuda.empty_cache()
    return torch.cat(outs, 0)


# ============================================================
# 提取 K/V
# ============================================================
@torch.no_grad()
def extract_kv(model, layer_idx, padded, mask):
    attn = model.model.layers[layer_idx].self_attn
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)

    all_k, all_v = [], []
    batch_size = 64

    for start in range(0, len(padded), batch_size):
        batch = padded[start:start+batch_size].to(DEVICE)
        m = mask[start:start+batch_size].to(DEVICE)
        B_b, T_b = batch.shape

        out = model.model(input_ids=batch, attention_mask=m,
                          output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[layer_idx]
        normed = model.model.layers[layer_idx].input_layernorm(hidden)

        k = attn.k_proj(normed).view(B_b, T_b, n_kv, head_dim)
        k = attn.k_norm(k).transpose(1, 2)
        all_k.append(k.transpose(1, 2).reshape(B_b, T_b, -1).to(torch.float16).cpu())

        v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
        all_v.append(v.transpose(1, 2).reshape(B_b, T_b, -1).to(torch.float16).cpu())

        del out, hidden, normed
        torch.cuda.empty_cache()

    K = torch.cat(all_k, 0).reshape(-1, all_k[0].shape[-1])
    V = torch.cat(all_v, 0).reshape(-1, all_v[0].shape[-1])
    return K, V


# ============================================================
# 加载模型
# ============================================================
print("\n加载模型...")
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()

model_A.model.rotary_emb.inv_freq = model_A.model.rotary_emb.inv_freq.to(DEVICE)
model_B.model.rotary_emb.inv_freq = model_B.model.rotary_emb.inv_freq.to(DEVICE)

n_layers = model_A.config.num_hidden_layers
n_q = model_B.config.num_attention_heads
n_kv = model_B.config.num_key_value_heads
head_dim = getattr(model_B.config, "head_dim",
                   model_B.config.hidden_size // n_q)

D = n_kv * head_dim
print(f"Layers: {n_layers}, D={D}")


# ============================================================
# 1. 训练每层的 W_K, W_V 和 MLP
# ============================================================
print("\n[1] 训练每层映射...")

W_K_list = []
W_V_list = []
mapper_mlp_list = []      # 普通 MLP + 混合 loss
mapper_res_list = []      # 残差 MLP + 混合 loss

# train 的非 padding 索引
train_flat_mask = train_mask.reshape(-1).bool()
train_valid_idx = torch.where(train_flat_mask)[0]

# 划 train/val
perm = torch.randperm(len(train_valid_idx))
n_val = int(len(train_valid_idx) * MLP_VAL_RATIO)
val_idx_global = train_valid_idx[perm[:n_val]]
tr_idx_global = train_valid_idx[perm[n_val:]]

print(f"  Train valid: {len(tr_idx_global)}, Val: {len(val_idx_global)}")

# 只训前 3 层测试（快速验证）
TEST_LAYERS = [0, 14, 27] if False else list(range(n_layers))

for l in TEST_LAYERS:
    print(f"\n  === Layer {l} ===")
    K_A, V_A = extract_kv(model_A, l, train_padded, train_mask)
    K_B, V_B = extract_kv(model_B, l, train_padded, train_mask)

    # 只用非 padding
    K_A_v = K_A[train_valid_idx]
    V_A_v = V_A[train_valid_idx]
    K_B_v = K_B[train_valid_idx]
    V_B_v = V_B[train_valid_idx]

    # Ridge W_K
    W_K = fit_ridge(K_A_v, K_B_v)
    W_V = fit_ridge(V_A_v, V_B_v)
    W_K_list.append(W_K)
    W_V_list.append(W_V)

    K_r2 = r2_score(K_A_v.float() @ W_K, K_B_v.float())
    V_r2 = r2_score(V_A_v.float() @ W_V, V_B_v.float())
    print(f"  Linear: K R²={K_r2:.4f}, V R²={V_r2:.4f}")

    # MLP (普通 + 混合)
    mapper_mlp = VMapper(D, D, MLP_HIDDEN, residual=False)
    mapper_mlp = train_mlp(
        mapper_mlp, V_A, V_B,
        tr_idx_global, val_idx_global,
        epochs=MLP_EPOCHS, batch_size=MLP_BATCH,
        lr=MLP_LR, patience=MLP_PATIENCE,
        lambda_cos=MLP_LAMBDA_COS, device=DEVICE, log_every=20,
    )
    mapper_mlp_list.append(mapper_mlp)

    # 残差 MLP
    mapper_res = VMapper(D, D, MLP_HIDDEN, residual=True)
    mapper_res = train_mlp(
        mapper_res, V_A, V_B,
        tr_idx_global, val_idx_global,
        epochs=MLP_EPOCHS, batch_size=MLP_BATCH,
        lr=MLP_LR, patience=MLP_PATIENCE,
        lambda_cos=MLP_LAMBDA_COS, device=DEVICE, log_every=20,
    )
    mapper_res_list.append(mapper_res)

    # 释放
    del K_A, V_A, K_B, V_B, K_A_v, V_A_v, K_B_v, V_B_v
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# 2. 提取 eval 的 0.6B K/V
# ============================================================
print("\n[2] 提取 eval 的 0.6B K/V...")

K_A_eval = []
V_A_eval = []
for l in range(n_layers):
    K, V = extract_kv(model_A, l, eval_padded, eval_mask)
    K_A_eval.append(K)
    V_A_eval.append(V)
    if (l + 1) % 7 == 0:
        print(f"  Layer {l+1}/{n_layers}")
print("  Done.")


# ============================================================
# 3. 对 eval V 做三种映射
# ============================================================
print("\n[3] 对 eval V 做映射...")

# V 的三种版本
V_linear_eval = []
V_mlp_eval = []
V_res_eval = []

for l in range(n_layers):
    v_a = V_A_eval[l]

    # Linear
    v_lin = (v_a.float() @ W_V_list[l]).to(torch.float16)
    V_linear_eval.append(v_lin)

    # MLP
    v_mlp = mlp_predict(mapper_mlp_list[l], v_a).to(torch.float16)
    V_mlp_eval.append(v_mlp)

    # Residual MLP
    v_res = mlp_predict(mapper_res_list[l], v_a).to(torch.float16)
    V_res_eval.append(v_res)

    if (l + 1) % 7 == 0:
        print(f"  Layer {l+1}/{n_layers}")

print("  Done.")


# ============================================================
# 4. forward_with_transferred_kv
# ============================================================
@torch.no_grad()
def forward_with_transferred_kv(model, input_ids, attention_mask,
                                 K_A_all, V_mapped_all, W_K_list,
                                 n_q, n_kv, head_dim):
    """
    K: 从 K_A 通过 W_K 映射
    V: 直接用 V_mapped_all（已经映射过的，三种版本传不同的进来）
    """
    B_b, T_b = input_ids.shape
    hidden = model.model.embed_tokens(input_ids)
    device = hidden.device

    position_ids = torch.arange(T_b, device=device).unsqueeze(0).expand(B_b, -1)
    cos, sin = model.model.rotary_emb(hidden, position_ids)

    causal = torch.triu(
        torch.full((T_b, T_b), torch.finfo(hidden.dtype).min,
                   device=device, dtype=hidden.dtype),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    pad_mask = attention_mask[:, None, None, :].to(hidden.dtype)
    pad_mask = (1.0 - pad_mask) * torch.finfo(hidden.dtype).min
    combined_mask = causal + pad_mask

    n_rep = n_q // n_kv
    scale = 1.0 / math.sqrt(head_dim)

    for l, layer in enumerate(model.model.layers):
        K_a = K_A_all[l].to(device).float()
        V_a = V_mapped_all[l].to(device).float()
        W_K = W_K_list[l].to(device)

        K_map = (K_a @ W_K).to(hidden.dtype)
        V_map = V_a.to(hidden.dtype)

        K = K_map.view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
        V = V_map.view(B_b, T_b, n_kv, head_dim).transpose(1, 2)

        K = apply_rotary(K, cos, sin)

        residual = hidden
        x = layer.input_layernorm(hidden)
        attn = layer.self_attn
        q = attn.q_proj(x).view(B_b, T_b, n_q, head_dim)
        q = attn.q_norm(q).transpose(1, 2)
        q = apply_rotary(q, cos, sin)

        K_exp = repeat_kv(K, n_rep)
        V_exp = repeat_kv(V, n_rep)

        scores = torch.matmul(q.float(), K_exp.float().transpose(-1, -2)) * scale
        scores = scores + combined_mask
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        attn_out = torch.matmul(probs, V_exp)

        attn_out = attn_out.transpose(1, 2).reshape(B_b, T_b, -1)
        attn_out = attn.o_proj(attn_out)
        hidden = residual + attn_out

        residual = hidden
        x = layer.post_attention_layernorm(hidden)
        hidden = residual + layer.mlp(x)

    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)
    return logits


# ============================================================
# 5. PPL
# ============================================================
@torch.no_grad()
def compute_ppl_native(model, input_ids, attention_mask, batch_size=4):
    total_loss = 0.0
    total_tokens = 0
    for s in range(0, len(input_ids), batch_size):
        e = min(s + batch_size, len(input_ids))
        batch = input_ids[s:e].to(DEVICE)
        mask = attention_mask[s:e].to(DEVICE)
        outputs = model(input_ids=batch, attention_mask=mask)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = batch[:, 1:].reshape(-1)
        shift_mask = mask[:, 1:].reshape(-1).float()
        loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")
        total_loss += (loss * shift_mask).sum().item()
        total_tokens += shift_mask.sum().item()
        del outputs, logits, loss
        torch.cuda.empty_cache()
    return math.exp(total_loss / total_tokens)


@torch.no_grad()
def compute_ppl_transfer(model, input_ids, attention_mask,
                         K_A_all, V_mapped_all, W_K_list,
                         n_q, n_kv, head_dim, batch_size=2):
    total_loss = 0.0
    total_tokens = 0
    for s in range(0, len(input_ids), batch_size):
        e = min(s + batch_size, len(input_ids))
        batch = input_ids[s:e].to(DEVICE)
        mask = attention_mask[s:e].to(DEVICE)
        b = batch.shape[0]
        T_b = batch.shape[1]

        K_slice = [K_A_all[l][s*T_b:e*T_b] for l in range(n_layers)]
        V_slice = [V_mapped_all[l][s*T_b:e*T_b] for l in range(n_layers)]

        logits = forward_with_transferred_kv(
            model, batch, mask, K_slice, V_slice, W_K_list,
            n_q, n_kv, head_dim
        )

        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = batch[:, 1:].reshape(-1)
        shift_mask = mask[:, 1:].reshape(-1).float()
        loss = F.cross_entropy(shift_logits, shift_labels, reduction="none")
        total_loss += (loss * shift_mask).sum().item()
        total_tokens += shift_mask.sum().item()
        del logits, loss
        torch.cuda.empty_cache()
    return math.exp(total_loss / total_tokens)


# ============================================================
# 6. 跑 4 种 PPL
# ============================================================
print("\n" + "=" * 70)
print("PPL 评估")
print("=" * 70)

print("\n[1/4] Native (1.7B 自己)...")
ppl_native = compute_ppl_native(model_B, eval_padded, eval_mask)
print(f"  PPL_native = {ppl_native:.4f}")

print("\n[2/4] Transfer - Linear V...")
ppl_linear = compute_ppl_transfer(
    model_B, eval_padded, eval_mask,
    K_A_eval, V_linear_eval, W_K_list,
    n_q, n_kv, head_dim, batch_size=2
)
print(f"  PPL_linear = {ppl_linear:.4f}")

print("\n[3/4] Transfer - MLP V (混合 loss)...")
ppl_mlp = compute_ppl_transfer(
    model_B, eval_padded, eval_mask,
    K_A_eval, V_mlp_eval, W_K_list,
    n_q, n_kv, head_dim, batch_size=2
)
print(f"  PPL_mlp = {ppl_mlp:.4f}")

print("\n[4/4] Transfer - Residual MLP V...")
ppl_res = compute_ppl_transfer(
    model_B, eval_padded, eval_mask,
    K_A_eval, V_res_eval, W_K_list,
    n_q, n_kv, head_dim, batch_size=2
)
print(f"  PPL_res = {ppl_res:.4f}")


# ============================================================
# 7. 汇总
# ============================================================
print()
print("=" * 75)
print("Final Summary")
print("=" * 75)
print(f"{'Method':>30}  {'PPL':>10}  {'Ratio':>10}")
print("-" * 75)
print(f"{'Native (1.7B)':>30}  {ppl_native:10.4f}  {1.0:10.4f}")
print(f"{'Transfer - Linear V':>30}  {ppl_linear:10.4f}  {ppl_linear/ppl_native:10.4f}")
print(f"{'Transfer - MLP V':>30}  {ppl_mlp:10.4f}  {ppl_mlp/ppl_native:10.4f}")
print(f"{'Transfer - Residual MLP V':>30}  {ppl_res:10.4f}  {ppl_res/ppl_native:10.4f}")
print("=" * 75)

print()
print("关键对比：")
print(f"  Linear V:       PPL={ppl_linear:.4f}  Ratio={ppl_linear/ppl_native:.4f}")
print(f"  MLP V:          PPL={ppl_mlp:.4f}  Ratio={ppl_mlp/ppl_native:.4f}")
print(f"  Residual MLP V: PPL={ppl_res:.4f}  Ratio={ppl_res/ppl_native:.4f}")