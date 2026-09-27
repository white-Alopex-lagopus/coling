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

TRAIN_PATH = "../dataset/wikitext-02/train.jsonl"
VAL_PATH   = "../dataset/wikitext-02/validation.jsonl"
TEST_PATH  = "../dataset/wikitext-02/test.jsonl"

N_TRAIN = 3000
N_VAL   = 500
N_TEST  = 500

MAX_TOKENS_PER_ART = 128

DEVICE = "cuda"
SEED = 42
RIDGE_BATCH = 50000
RIDGE_LAMBDA = 100.0

torch.manual_seed(SEED)


# ============================================================
# 数据
# ============================================================
tok = AutoTokenizer.from_pretrained(MODEL_A)


def load_data(path, n):
    df = pd.read_json(path, lines=True)
    texts = df["text"].iloc[:n].tolist()
    ids_list = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        ids = ids[:MAX_TOKENS_PER_ART]
        if len(ids) >= 16:
            ids_list.append(ids)
    L = max(len(x) for x in ids_list)
    padded = torch.zeros(len(ids_list), L, dtype=torch.long)
    mask = torch.zeros(len(ids_list), L, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, :len(ids)] = ids
        mask[i, :len(ids)] = 1
    return padded, mask


train_padded, train_mask = load_data(TRAIN_PATH, N_TRAIN)
val_padded,   val_mask   = load_data(VAL_PATH,   N_VAL)
test_padded,  test_mask  = load_data(TEST_PATH,  N_TEST)

print(f"Train: {train_padded.shape}")
print(f"Val  : {val_padded.shape}")
print(f"Test : {test_padded.shape}")


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
# 流式提取 K/V
# ============================================================
@torch.no_grad()
def extract_kv_for_batch(model, padded, mask, article_indices, layer_idx,
                          n_kv, head_dim):
    ids = padded[article_indices].to(DEVICE)
    m = mask[article_indices].to(DEVICE)
    B_b, T_b = ids.shape

    out = model.model(input_ids=ids, attention_mask=m,
                      output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states[layer_idx]
    normed = model.model.layers[layer_idx].input_layernorm(hidden)
    attn = model.model.layers[layer_idx].self_attn

    k = attn.k_proj(normed).view(B_b, T_b, n_kv, head_dim)
    k = attn.k_norm(k).transpose(1, 2)
    k = k.transpose(1, 2).reshape(B_b, T_b, -1)

    v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
    v = v.transpose(1, 2).reshape(B_b, T_b, -1)

    del out, hidden, normed
    return k, v


# ============================================================
# forward
# ============================================================
def forward_transfer(model, input_ids, attention_mask,
                     K_A_all, V_A_all, K_B_all_unused,
                     W_K_list, W_V_linear_list,
                     W_V_concat_1_list,   # 方案 1: [2D, D]
                     W_V_concat_5_list,   # 方案 5: [2D, D]
                     mode,                 # "linear" / "concat1" / "concat5"
                     n_q, n_kv, head_dim, n_layers):
    B_b, T_b = input_ids.shape
    device = input_ids.device
    hidden = model.model.embed_tokens(input_ids)

    position_ids = torch.arange(T_b, device=device).unsqueeze(0).expand(B_b, -1)
    cos, sin = model.model.rotary_emb(hidden, position_ids)

    causal = torch.triu(
        torch.full((T_b, T_b), torch.finfo(hidden.dtype).min,
                   device=device, dtype=hidden.dtype),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    pad_mask = (1.0 - attention_mask[:, None, None, :].to(hidden.dtype)) \
               * torch.finfo(hidden.dtype).min
    combined_mask = causal + pad_mask

    n_rep = n_q // n_kv
    scale = 1.0 / math.sqrt(head_dim)

    for l, layer in enumerate(model.model.layers):
        K_a = K_A_all[l].to(device).float()
        V_a = V_A_all[l].to(device).float()

        W_K = W_K_list[l].to(device)
        K_map = (K_a @ W_K).to(hidden.dtype)

        # V 映射
        if mode == "linear":
            W_V = W_V_linear_list[l].to(device)
            V_map = (V_a @ W_V).to(hidden.dtype)
        elif mode == "concat1":
            # 方案 1: concat[V_A, K_A]
            X = torch.cat([V_a, K_a], dim=-1)      # [B, T, 2D]
            W_V = W_V_concat_1_list[l].to(device)
            V_map = (X @ W_V).to(hidden.dtype)
        elif mode == "concat5":
            # 方案 5: concat[V_A, K_A @ W_K]
            K_a_mapped = K_a @ W_K                  # [B, T, D]
            X = torch.cat([V_a, K_a_mapped], dim=-1)
            W_V = W_V_concat_5_list[l].to(device)
            V_map = (X @ W_V).to(hidden.dtype)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        K = K_map.reshape(B_b, T_b, n_kv, head_dim).transpose(1, 2)
        V = V_map.reshape(B_b, T_b, n_kv, head_dim).transpose(1, 2)

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
# 加载模型
# ============================================================
print("\n加载模型...")
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()

for p in model_B.parameters():
    p.requires_grad = False

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
# 1. 训练所有层的 W_K, W_V（三套）
# ============================================================
print("\n[1] 训练所有层的映射...")

W_K_list = [None] * n_layers
W_V_linear_list = [None] * n_layers
W_V_concat_1_list = [None] * n_layers
W_V_concat_5_list = [None] * n_layers

# 流式累加矩阵
XtX_K = [torch.zeros(D, D) for _ in range(n_layers)]
XtY_K = [torch.zeros(D, D) for _ in range(n_layers)]

XtX_V = [torch.zeros(D, D) for _ in range(n_layers)]
XtY_V = [torch.zeros(D, D) for _ in range(n_layers)]

# 方案 1: 输入 [V_A, K_A]，X 维度 2D
XtX_c1 = [torch.zeros(2*D, 2*D) for _ in range(n_layers)]
XtY_c1 = [torch.zeros(2*D, D) for _ in range(n_layers)]

# 方案 5: 输入 [V_A, K_A@W_K]，但 W_K 还没训练，需要先训练 W_K
# 先只累加 K 的 XtX
print("  第一阶段：训练 W_K...")

BATCH_ART = 64
for s in range(0, len(train_padded), BATCH_ART):
    e = min(s + BATCH_ART, len(train_padded))
    ba = torch.arange(s, e)

    # A 的 K/V
    K_A_b, V_A_b = None, None
    with torch.no_grad():
        ids = train_padded[ba].to(DEVICE)
        m = train_mask[ba].to(DEVICE)
        out = model_A.model(input_ids=ids, attention_mask=m,
                            output_hidden_states=True, use_cache=False)
        K_A_b_list = []
        V_A_b_list = []
        B_b, T_b = ids.shape
        for l in range(n_layers):
            layer = model_A.model.layers[l]
            attn = layer.self_attn
            hidden = out.hidden_states[l]
            normed = layer.input_layernorm(hidden)
            k = attn.k_proj(normed).view(B_b, T_b, n_kv, head_dim)
            k = attn.k_norm(k).transpose(1, 2).transpose(1, 2).reshape(B_b, T_b, -1)
            K_A_b_list.append(k)
            v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
            v = v.transpose(1, 2).reshape(B_b, T_b, -1)
            V_A_b_list.append(v)
        del out

    # B 的 K/V
    with torch.no_grad():
        out = model_B.model(input_ids=ids, attention_mask=m,
                            output_hidden_states=True, use_cache=False)
        K_B_b_list = []
        V_B_b_list = []
        for l in range(n_layers):
            layer = model_B.model.layers[l]
            attn = layer.self_attn
            hidden = out.hidden_states[l]
            normed = layer.input_layernorm(hidden)
            k = attn.k_proj(normed).view(B_b, T_b, n_kv, head_dim)
            k = attn.k_norm(k).transpose(1, 2).transpose(1, 2).reshape(B_b, T_b, -1)
            K_B_b_list.append(k)
            v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
            v = v.transpose(1, 2).reshape(B_b, T_b, -1)
            V_B_b_list.append(v)
        del out

    # 累加
    for l in range(n_layers):
        Ka = K_A_b_list[l].reshape(-1, D).float().cpu()
        Va = V_A_b_list[l].reshape(-1, D).float().cpu()
        Kb = K_B_b_list[l].reshape(-1, D).float().cpu()
        Vb = V_B_b_list[l].reshape(-1, D).float().cpu()

        # W_K
        XtX_K[l] += Ka.T @ Ka
        XtY_K[l] += Ka.T @ Kb

        # W_V linear
        XtX_V[l] += Va.T @ Va
        XtY_V[l] += Va.T @ Vb

        # 方案 1: [V_A, K_A]
        Xc1 = torch.cat([Va, Ka], dim=-1)     # [N, 2D]
        XtX_c1[l] += Xc1.T @ Xc1
        XtY_c1[l] += Xc1.T @ Vb

    del K_A_b_list, V_A_b_list, K_B_b_list, V_B_b_list
    torch.cuda.empty_cache()
    if (s // BATCH_ART + 1) % 5 == 0:
        print(f"  batch {s//BATCH_ART + 1}/{(len(train_padded)+BATCH_ART-1)//BATCH_ART}")

# 解 W_K, W_V_linear, W_V_concat_1
for l in range(n_layers):
    W_K_list[l] = torch.linalg.solve(XtX_K[l] + RIDGE_LAMBDA * torch.eye(D), XtY_K[l])
    W_V_linear_list[l] = torch.linalg.solve(XtX_V[l] + RIDGE_LAMBDA * torch.eye(D), XtY_V[l])
    W_V_concat_1_list[l] = torch.linalg.solve(XtX_c1[l] + RIDGE_LAMBDA * torch.eye(2*D), XtY_c1[l])

print("  Done W_K, W_V_linear, W_V_concat_1")

# 方案 5 需要 W_K 已知，再累加
print("  第二阶段：训练方案 5 的 W_V...")

XtX_c5 = [torch.zeros(2*D, 2*D) for _ in range(n_layers)]
XtY_c5 = [torch.zeros(2*D, D) for _ in range(n_layers)]

for s in range(0, len(train_padded), BATCH_ART):
    e = min(s + BATCH_ART, len(train_padded))
    ba = torch.arange(s, e)

    with torch.no_grad():
        ids = train_padded[ba].to(DEVICE)
        m = train_mask[ba].to(DEVICE)
        out = model_A.model(input_ids=ids, attention_mask=m,
                            output_hidden_states=True, use_cache=False)
        K_A_b_list = []
        V_A_b_list = []
        B_b, T_b = ids.shape
        for l in range(n_layers):
            layer = model_A.model.layers[l]
            attn = layer.self_attn
            hidden = out.hidden_states[l]
            normed = layer.input_layernorm(hidden)
            k = attn.k_proj(normed).view(B_b, T_b, n_kv, head_dim)
            k = attn.k_norm(k).transpose(1, 2).transpose(1, 2).reshape(B_b, T_b, -1)
            K_A_b_list.append(k)
            v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
            v = v.transpose(1, 2).reshape(B_b, T_b, -1)
            V_A_b_list.append(v)
        del out

        out = model_B.model(input_ids=ids, attention_mask=m,
                            output_hidden_states=True, use_cache=False)
        V_B_b_list = []
        for l in range(n_layers):
            layer = model_B.model.layers[l]
            attn = layer.self_attn
            hidden = out.hidden_states[l]
            normed = layer.input_layernorm(hidden)
            v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
            v = v.transpose(1, 2).reshape(B_b, T_b, -1)
            V_B_b_list.append(v)
        del out

    for l in range(n_layers):
        Ka = K_A_b_list[l].reshape(-1, D).float().cpu()
        Va = V_A_b_list[l].reshape(-1, D).float().cpu()
        Vb = V_B_b_list[l].reshape(-1, D).float().cpu()

        # K 先映射
        Ka_mapped = Ka @ W_K_list[l]     # [N, D]

        # 方案 5: [V_A, K_A@W_K]
        Xc5 = torch.cat([Va, Ka_mapped], dim=-1)
        XtX_c5[l] += Xc5.T @ Xc5
        XtY_c5[l] += Xc5.T @ Vb

    del K_A_b_list, V_A_b_list, V_B_b_list
    torch.cuda.empty_cache()
    if (s // BATCH_ART + 1) % 5 == 0:
        print(f"  batch {s//BATCH_ART + 1}/{(len(train_padded)+BATCH_ART-1)//BATCH_ART}")

for l in range(n_layers):
    W_V_concat_5_list[l] = torch.linalg.solve(XtX_c5[l] + RIDGE_LAMBDA * torch.eye(2*D), XtY_c5[l])

print("  Done W_V_concat_5")


# ============================================================
# 2. PPL 评估
# ============================================================
print("\n" + "=" * 70)
print("PPL 评估（用 test 集）")
print("=" * 70)


@torch.no_grad()
def ppl_native(padded, mask, batch_size=2):
    total_loss, total_tokens = 0.0, 0
    for s in range(0, len(padded), batch_size):
        e = min(s + batch_size, len(padded))
        ids = padded[s:e].to(DEVICE)
        m = mask[s:e].to(DEVICE)
        out = model_B(input_ids=ids, attention_mask=m)
        logits = out.logits
        sl = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        slb = ids[:, 1:].reshape(-1)
        sm = m[:, 1:].reshape(-1).float()
        lp = F.cross_entropy(sl, slb, reduction="none")
        total_loss += (lp * sm).sum().item()
        total_tokens += sm.sum().item()
        del out, logits, lp
        torch.cuda.empty_cache()
    return math.exp(total_loss / total_tokens)


@torch.no_grad()
def ppl_transfer(padded, mask, mode, batch_size=2):
    total_loss, total_tokens = 0.0, 0
    for s in range(0, len(padded), batch_size):
        e = min(s + batch_size, len(padded))
        ba = torch.arange(s, e)

        ids = padded[ba].to(DEVICE)
        m_ = mask[ba].to(DEVICE)

        # 流式提 K/V
        K_b, V_b = [], []
        with torch.no_grad():
            out = model_A.model(input_ids=ids, attention_mask=m_,
                                output_hidden_states=True, use_cache=False)
            B_b, T_b = ids.shape
            for l in range(n_layers):
                layer = model_A.model.layers[l]
                attn = layer.self_attn
                hidden = out.hidden_states[l]
                normed = layer.input_layernorm(hidden)
                k = attn.k_proj(normed).view(B_b, T_b, n_kv, head_dim)
                k = attn.k_norm(k).transpose(1, 2).transpose(1, 2).reshape(B_b, T_b, -1)
                K_b.append(k)
                v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
                v = v.transpose(1, 2).reshape(B_b, T_b, -1)
                V_b.append(v)
            del out

        logits = forward_transfer(
            model_B, ids, m_,
            K_b, V_b, None,
            W_K_list, W_V_linear_list,
            W_V_concat_1_list, W_V_concat_5_list,
            mode,
            n_q, n_kv, head_dim, n_layers,
        )

        sl = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        slb = ids[:, 1:].reshape(-1)
        sm = m_[:, 1:].reshape(-1).float()
        lp = F.cross_entropy(sl, slb, reduction="none")
        total_loss += (lp * sm).sum().item()
        total_tokens += sm.sum().item()

        del logits, lp, K_b, V_b
        torch.cuda.empty_cache()
    return math.exp(total_loss / total_tokens)


print("\n[1/4] PPL_native...")
ppl_nat = ppl_native(test_padded, test_mask)
print(f"  PPL_native = {ppl_nat:.4f}")

print("\n[2/4] PPL_linear (只用 V_A)...")
ppl_lin = ppl_transfer(test_padded, test_mask, mode="linear")
print(f"  PPL_linear = {ppl_lin:.4f}")

print("\n[3/4] PPL_concat1 (方案 1: [V_A, K_A])...")
ppl_c1 = ppl_transfer(test_padded, test_mask, mode="concat1")
print(f"  PPL_concat1 = {ppl_c1:.4f}")

print("\n[4/4] PPL_concat5 (方案 5: [V_A, K_A@W_K])...")
ppl_c5 = ppl_transfer(test_padded, test_mask, mode="concat5")
print(f"  PPL_concat5 = {ppl_c5:.4f}")


# ============================================================
# 3. 汇总
# ============================================================
print()
print("=" * 80)
print("K 辅助 V 映射：PPL 验证结果")
print("=" * 80)
print(f"{'方法':>30}  {'PPL':>10}  {'Ratio':>10}  {'Δ vs Linear':>15}")
print("-" * 80)
print(f"{'Native (1.7B)':>30}  {ppl_nat:10.4f}  {1.0:10.4f}  {'—':>15}")
print(f"{'Linear (V_A only)':>30}  {ppl_lin:10.4f}  {ppl_lin/ppl_nat:10.4f}  {'—':>15}")
print(f"{'Concat1 [V_A, K_A]':>30}  {ppl_c1:10.4f}  {ppl_c1/ppl_nat:10.4f}  "
      f"{(ppl_c1-ppl_lin)/ppl_nat:>+15.4f}")
print(f"{'Concat5 [V_A, K_A@W_K]':>30}  {ppl_c5:10.4f}  {ppl_c5/ppl_nat:10.4f}  "
      f"{(ppl_c5-ppl_lin)/ppl_nat:>+15.4f}")
print("=" * 80)

print()
print("解读：")
if ppl_c1 < ppl_lin and ppl_c5 < ppl_lin:
    print("  ✅ 两个方案都优于 Linear → K 辅助有效")
elif ppl_c1 < ppl_lin:
    print("  ✅ 方案 1 优于 Linear → K 辅助有效（方案 1）")
elif ppl_c5 < ppl_lin:
    print("  ✅ 方案 5 优于 Linear → K 辅助有效（方案 5）")
else:
    print("  ❌ 两个方案都不如 Linear → V R² 提升未转化为 PPL 改善")