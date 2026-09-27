import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

# 训练 W 用
TRAIN_ARTICLES = 2000
# PPL 评估用
EVAL_ARTICLES = 500
MAX_TOKENS_PER_ART = 256

DEVICE = "cuda"
SEED = 42
RIDGE_BATCH = 50000
RIDGE_LAMBDA = 100.0

torch.manual_seed(SEED)


# ============================================================
# 数据
# ============================================================
df = pd.read_json(DATA_PATH, lines=True)
all_texts = df["text"].tolist()

train_texts = all_texts[:TRAIN_ARTICLES]
eval_texts = all_texts[TRAIN_ARTICLES:TRAIN_ARTICLES + EVAL_ARTICLES]

tok = AutoTokenizer.from_pretrained(MODEL_A)

def build_padded(texts, max_len=MAX_TOKENS_PER_ART):
    ids_list = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        ids = ids[:max_len]
        if len(ids) >= 16:
            ids_list.append(ids)
    L = max(len(x) for x in ids_list)
    padded = torch.zeros(len(ids_list), L, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, :len(ids)] = ids
    return padded

train_padded = build_padded(train_texts)
eval_padded = build_padded(eval_texts)

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


# ============================================================
# 提取 K/V（pre-RoPE，fp16 存）
# ============================================================
@torch.no_grad()
def extract_kv(model, layer_idx, padded):
    attn = model.model.layers[layer_idx].self_attn
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)

    all_k, all_v = [], []
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

        del out, hidden, normed
        torch.cuda.empty_cache()

    K = torch.cat(all_k, 0).reshape(-1, all_k[0].shape[-1])
    V = torch.cat(all_v, 0).reshape(-1, all_v[0].shape[-1])
    return K, V


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

print(f"Layers: {n_layers}, n_q={n_q}, n_kv={n_kv}, head_dim={head_dim}")


# ============================================================
# 1. 训练每层的 W_K, W_V
# ============================================================
print("\n[1] 训练每层映射 W_K, W_V...")

W_K_list = []
W_V_list = []

for l in range(n_layers):
    K_A = extract_kv(model_A, l, train_padded)
    V_A = extract_kv(model_A, l, train_padded)[1] if False else None

    # 需要重新调用两次，避免写复杂
    K_A, V_A = extract_kv(model_A, l, train_padded)
    K_B, V_B = extract_kv(model_B, l, train_padded)

    W_K = fit_ridge(K_A, K_B)
    W_V = fit_ridge(V_A, V_B)

    W_K_list.append(W_K)
    W_V_list.append(W_V)

    K_r2 = 1 - ((K_A.float() @ W_K - K_B.float())**2).sum() / (K_B.float()**2).sum()
    V_r2 = 1 - ((V_A.float() @ W_V - V_B.float())**2).sum() / (V_B.float()**2).sum()
    print(f"  Layer {l:2d}: K R²={K_r2.item():.4f}  V R²={V_r2.item():.4f}")

    del K_A, V_A, K_B, V_B
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# 2. 提取 eval 集的 0.6B K/V（pre-RoPE）
# ============================================================
print("\n[2] 提取 eval 集的 0.6B K/V...")

K_A_eval = []
V_A_eval = []
for l in range(n_layers):
    K, V = extract_kv(model_A, l, eval_padded)
    K_A_eval.append(K)
    V_A_eval.append(V)
    if (l + 1) % 7 == 0:
        print(f"  Layer {l+1}/{n_layers}")
print("  Done.")


# ============================================================
# 3. 自定义 forward：1.7B 的 Q + 映射后的 K/V
# ============================================================
@torch.no_grad()
def forward_with_transferred_kv(model, input_ids, K_A_all, V_A_all,
                                 W_K_list, W_V_list, n_q, n_kv, head_dim):
    """
    input_ids: [B, T]
    K_A_all, V_A_all: list of [B*T, n_kv*D]（pre-RoPE）
    W_K_list, W_V_list: list of [n_kv*D, n_kv*D]
    返回 logits [B, T, vocab]
    """
    B_b, T_b = input_ids.shape
    hidden = model.model.embed_tokens(input_ids)
    device = hidden.device

    # RoPE
    position_ids = torch.arange(T_b, device=device).unsqueeze(0).expand(B_b, -1)
    cos, sin = model.model.rotary_emb(hidden, position_ids)

    # 因果 mask
    causal_mask = torch.triu(
        torch.full((T_b, T_b), torch.finfo(hidden.dtype).min,
                   device=device, dtype=hidden.dtype),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    n_rep = n_q // n_kv
    scale = 1.0 / math.sqrt(head_dim)

    for l, layer in enumerate(model.model.layers):
        # ---- K/V：从 0.6B 映射 ----
        K_a = K_A_all[l].to(device).float()       # [B*T, n_kv*D]
        V_a = V_A_all[l].to(device).float()
        W_K = W_K_list[l].to(device)
        W_V = W_V_list[l].to(device)

        K_map = (K_a @ W_K).to(hidden.dtype)      # [B*T, n_kv*D]
        V_map = (V_a @ W_V).to(hidden.dtype)

        K = K_map.view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
        V = V_map.view(B_b, T_b, n_kv, head_dim).transpose(1, 2)

        # RoPE on K
        K = apply_rotary(K, cos, sin)

        # ---- Q：1.7B 自己的 ----
        residual = hidden
        x = layer.input_layernorm(hidden)

        attn = layer.self_attn
        q = attn.q_proj(x).view(B_b, T_b, n_q, head_dim)
        q = attn.q_norm(q).transpose(1, 2)
        q = apply_rotary(q, cos, sin)

        # ---- Attention ----
        K_exp = repeat_kv(K, n_rep)
        V_exp = repeat_kv(V, n_rep)

        scores = torch.matmul(q.float(), K_exp.float().transpose(-1, -2)) * scale
        scores = scores + causal_mask
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        attn_out = torch.matmul(probs, V_exp)

        # ---- o_proj + residual ----
        attn_out = attn_out.transpose(1, 2).reshape(B_b, T_b, -1)
        attn_out = attn.o_proj(attn_out)
        hidden = residual + attn_out

        # ---- MLP ----
        residual = hidden
        x = layer.post_attention_layernorm(hidden)
        hidden = residual + layer.mlp(x)

    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)
    return logits


# ============================================================
# 4. 三个 PPL 计算
# ============================================================
@torch.no_grad()
def compute_ppl_native(model, input_ids, batch_size=4):
    """1.7B 自己的 KV"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for s in range(0, len(input_ids), batch_size):
        batch = input_ids[s:s+batch_size].to(DEVICE)
        outputs = model(input_ids=batch)
        logits = outputs.logits  # [B, T, V]

        # shift
        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = batch[:, 1:].reshape(-1)

        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        total_loss += loss.item()
        total_tokens += shift_labels.numel()

        del outputs, logits, loss
        torch.cuda.empty_cache()

    return math.exp(total_loss / total_tokens)


@torch.no_grad()
def compute_ppl_transfer(model, input_ids, K_A_all, V_A_all, W_K_list, W_V_list,
                         n_q, n_kv, head_dim, use_mapping=True, batch_size=2):
    """
    use_mapping=True: 用映射后的 K/V
    use_mapping=False: Identity（不映射，直接用 0.6B 的 K/V）
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for s in range(0, len(input_ids), batch_size):
        e = min(s + batch_size, len(input_ids))
        batch = input_ids[s:e].to(DEVICE)
        b = batch.shape[0]
        T_b = batch.shape[1]

        # 取当前 batch 的 K_A / V_A
        K_A_slice = []
        V_A_slice = []
        for l in range(n_layers):
            # K_A_all[l] 是 [B_eval*T_eval, D]
            K_A_slice.append(K_A_all[l][s*T_b:e*T_b])
            V_A_slice.append(V_A_all[l][s*T_b:e*T_b])

        # 如果 use_mapping=False，用 identity W（单位阵）
        if use_mapping:
            W_K_use = W_K_list
            W_V_use = W_V_list
        else:
            # Identity：跳过映射
            W_K_use = [torch.eye(W.shape[0]) for W in W_K_list]
            W_V_use = [torch.eye(W.shape[0]) for W in W_V_list]

        logits = forward_with_transferred_kv(
            model, batch, K_A_slice, V_A_slice,
            W_K_use, W_V_use, n_q, n_kv, head_dim
        )

        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = batch[:, 1:].reshape(-1)

        loss = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        total_loss += loss.item()
        total_tokens += shift_labels.numel()

        del logits, loss
        torch.cuda.empty_cache()

    return math.exp(total_loss / total_tokens)


# ============================================================
# 5. 三种 PPL
# ============================================================
print("\n" + "=" * 70)
print("PPL 评估")
print("=" * 70)

print("\n[PPL 1/3] Native (1.7B 自己的 KV)...")
ppl_native = compute_ppl_native(model_B, eval_padded)
print(f"  PPL_native = {ppl_native:.4f}")

print("\n[PPL 2/3] Identity (0.6B 的 KV，不映射)...")
ppl_identity = compute_ppl_transfer(
    model_B, eval_padded, K_A_eval, V_A_eval,
    W_K_list, W_V_list, n_q, n_kv, head_dim,
    use_mapping=False, batch_size=2
)
print(f"  PPL_identity = {ppl_identity:.4f}")

print("\n[PPL 3/3] Transfer (映射后的 0.6B KV)...")
ppl_transfer = compute_ppl_transfer(
    model_B, eval_padded, K_A_eval, V_A_eval,
    W_K_list, W_V_list, n_q, n_kv, head_dim,
    use_mapping=True, batch_size=2
)
print(f"  PPL_transfer = {ppl_transfer:.4f}")


# ============================================================
# 6. 汇总
# ============================================================
print()
print("=" * 70)
print("Final Summary")
print("=" * 70)
print(f"PPL_native    (1.7B 自己)        : {ppl_native:.4f}")
print(f"PPL_identity  (0.6B KV 不映射)   : {ppl_identity:.4f}")
print(f"PPL_transfer  (0.6B KV 映射后)   : {ppl_transfer:.4f}")
print()
print(f"Ratio (transfer / native)  : {ppl_transfer / ppl_native:.4f}")
print(f"Ratio (identity / native)  : {ppl_identity / ppl_native:.4f}")
print("=" * 70)