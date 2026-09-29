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
TEST_PATH  = "../dataset/wikitext-02/test.jsonl"

N_TRAIN = 3000
N_TEST  = 500
MAX_TOKENS_PER_ART = 128

DEVICE = "cuda"
SEED = 42
RIDGE_LAMBDA = 100.0
BATCH_ART = 32

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
test_padded,  test_mask  = load_data(TEST_PATH,  N_TEST)
print(f"Train: {train_padded.shape}, Test: {test_padded.shape}")


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


def fit_ridge_from_matrix(XtX, XtY, lam=RIDGE_LAMBDA):
    d = XtX.shape[0]
    return torch.linalg.solve(XtX + lam * torch.eye(d), XtY)


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
# 提取：一次 forward 拿到所有层的 K/V（GPU tensor）
# ============================================================
@torch.no_grad()
def extract_kv_all_layers(model, padded, mask, article_indices):
    ids = padded[article_indices].to(DEVICE)
    m = mask[article_indices].to(DEVICE)
    B_b, T_b = ids.shape

    out = model.model(input_ids=ids, attention_mask=m,
                      output_hidden_states=True, use_cache=False)

    K_list, V_list = [], []
    for l in range(n_layers):
        layer = model.model.layers[l]
        attn = layer.self_attn
        hidden = out.hidden_states[l]
        normed = layer.input_layernorm(hidden)

        k = attn.k_proj(normed).view(B_b, T_b, n_kv, head_dim)
        k = attn.k_norm(k).transpose(1, 2).transpose(1, 2).reshape(B_b, T_b, -1)
        K_list.append(k)

        v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
        v = v.transpose(1, 2).reshape(B_b, T_b, -1)
        V_list.append(v)

    del out
    return K_list, V_list


@torch.no_grad()
def extract_V_only(model, padded, mask, article_indices):
    ids = padded[article_indices].to(DEVICE)
    m = mask[article_indices].to(DEVICE)
    B_b, T_b = ids.shape
    out = model.model(input_ids=ids, attention_mask=m,
                      output_hidden_states=True, use_cache=False)
    V_list = []
    for l in range(n_layers):
        layer = model.model.layers[l]
        attn = layer.self_attn
        hidden = out.hidden_states[l]
        normed = layer.input_layernorm(hidden)
        v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
        v = v.transpose(1, 2).reshape(B_b, T_b, -1)
        V_list.append(v)
    del out
    return V_list


# ============================================================
# 方案定义
# ============================================================
def get_K_indices(scheme, l):
    if scheme == "K1":
        return [l]
    elif scheme == "K3":
        out = []
        if l - 1 >= 0: out.append(l - 1)
        out.append(l)
        if l + 1 < n_layers: out.append(l + 1)
        return out
    elif scheme == "K5":
        out = []
        for d in [2, 1]:
            if l - d >= 0: out.append(l - d)
        out.append(l)
        for d in [1, 2]:
            if l + d < n_layers: out.append(l + d)
        return out
    elif scheme == "K7":
        out = []
        for d in [3, 2, 1]:
            if l - d >= 0: out.append(l - d)
        out.append(l)
        for d in [1, 2, 3]:
            if l + d < n_layers: out.append(l + d)
        return out
    elif scheme == "K_all_mean":
        return "MEAN"
    else:
        raise ValueError(scheme)


def get_D_in(scheme):
    if scheme == "K_all_mean":
        return 2 * D
    K_idx = get_K_indices(scheme, 0)
    return (1 + len(K_idx)) * D


# ============================================================
# 训练 W_K
# ============================================================
print("\n[1] 训练 W_K...")
W_K_list = [None] * n_layers
XtX_K = [torch.zeros(D, D) for _ in range(n_layers)]
XtY_K = [torch.zeros(D, D) for _ in range(n_layers)]

N_art = len(train_padded)
for s in range(0, N_art, BATCH_ART):
    e = min(s + BATCH_ART, N_art)
    ba = torch.arange(s, e)

    K_A_b, _ = extract_kv_all_layers(model_A, train_padded, train_mask, ba)
    K_B_b, _ = extract_kv_all_layers(model_B, train_padded, train_mask, ba)

    for l in range(n_layers):
        X = K_A_b[l].reshape(-1, D).cpu().float()
        Y = K_B_b[l].reshape(-1, D).cpu().float()
        XtX_K[l] += X.T @ X
        XtY_K[l] += X.T @ Y

    del K_A_b, K_B_b
    torch.cuda.empty_cache()

for l in range(n_layers):
    W_K_list[l] = fit_ridge_from_matrix(XtX_K[l], XtY_K[l])
del XtX_K, XtY_K
gc.collect()
print("  Done.")


# ============================================================
# 流式训练 W_V（按方案）
# ============================================================
def train_W_V(scheme):
    print(f"\n[Train] {scheme}...")
    W_V_list = [None] * n_layers
    
    XtX_list = []
    XtY_list = []
    D_in_list = []
    for l in range(n_layers):
        if scheme == "K_all_mean":
            D_in = 2 * D
        else:
            K_idx = get_K_indices(scheme, l)
            D_in = (1 + len(K_idx)) * D
        D_in_list.append(D_in)
        XtX_list.append(torch.zeros(D_in, D_in, dtype=torch.float32))
        XtY_list.append(torch.zeros(D_in, D, dtype=torch.float32))

    for s in range(0, N_art, BATCH_ART):
        e = min(s + BATCH_ART, N_art)
        ba = torch.arange(s, e)

        K_A_b, V_A_b = extract_kv_all_layers(model_A, train_padded, train_mask, ba)
        V_B_b = extract_V_only(model_B, train_padded, train_mask, ba)

        # 特殊：K_all_mean
        if scheme == "K_all_mean":
            K_mean = torch.stack(K_A_b, dim=0).mean(dim=0)   # [b, T, D]

        for l in range(n_layers):
            if scheme == "K_all_mean":
                parts = [V_A_b[l].reshape(-1, D), K_mean.reshape(-1, D)]
            else:
                K_idx = get_K_indices(scheme, l)
                parts = [V_A_b[l].reshape(-1, D)]
                for ki in K_idx:
                    parts.append(K_A_b[ki].reshape(-1, D))

            X = torch.cat(parts, dim=-1).cpu().float()
            Y = V_B_b[l].reshape(-1, D).cpu().float()

            XtX_list[l] += X.T @ X
            XtY_list[l] += X.T @ Y

        del K_A_b, V_A_b, V_B_b
        if scheme == "K_all_mean":
            del K_mean
        torch.cuda.empty_cache()

        if (s // BATCH_ART + 1) % 10 == 0:
            print(f"  batch {s//BATCH_ART + 1}/{(N_art+BATCH_ART-1)//BATCH_ART}")

    for l in range(n_layers):
        W_V_list[l] = fit_ridge_from_matrix(XtX_list[l], XtY_list[l])

    del XtX_list, XtY_list
    gc.collect()
    return W_V_list


W_V_dict = {}
for scheme in ["K1", "K3", "K5", "K7", "K_all_mean"]:
    W_V_dict[scheme] = train_W_V(scheme)


# ============================================================
# forward
# ============================================================
def forward_transfer(model, input_ids, attention_mask,
                     K_A_all, V_A_all,
                     W_K_list, W_V_list, scheme,
                     n_q, n_kv, head_dim, n_layers, D):
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

    if scheme == "K_all_mean":
        K_mean = torch.stack([K_A_all[i] for i in range(n_layers)], dim=0).mean(dim=0)

    for l, layer in enumerate(model.model.layers):
        K_a = K_A_all[l].to(device).float()
        W_K = W_K_list[l].to(device)
        K_map = (K_a @ W_K).to(hidden.dtype)

        if scheme == "K_all_mean":
            X_v = torch.cat([V_A_all[l].to(device).float(),
                             K_mean.to(device).float()], dim=-1)
        else:
            K_idx = get_K_indices(scheme, l)
            parts = [V_A_all[l].to(device).float()]
            for ki in K_idx:
                parts.append(K_A_all[ki].to(device).float())
            X_v = torch.cat(parts, dim=-1)

        W_V = W_V_list[l].to(device)
        V_map = (X_v @ W_V).to(hidden.dtype)

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
# PPL 评估
# ============================================================
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
def ppl_transfer(scheme, batch_size=2):
    total_loss, total_tokens = 0.0, 0
    W_V_list = W_V_dict[scheme]

    for s in range(0, len(test_padded), batch_size):
        e = min(s + batch_size, len(test_padded))
        ba = torch.arange(s, e)
        ids = test_padded[ba].to(DEVICE)
        m_ = test_mask[ba].to(DEVICE)

        K_b, V_b = extract_kv_all_layers(model_A, test_padded, test_mask, ba)

        logits = forward_transfer(
            model_B, ids, m_, K_b, V_b,
            W_K_list, W_V_list, scheme,
            n_q, n_kv, head_dim, n_layers, D,
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


print("\n" + "=" * 70)
print("PPL 评估")
print("=" * 70)

print("\n[PPL] Native...")
ppl_nat = ppl_native(test_padded, test_mask)
print(f"  PPL_native = {ppl_nat:.4f}")

results = {"Native": ppl_nat}

for scheme in ["K1", "K3", "K5", "K7", "K_all_mean"]:
    print(f"\n[PPL] {scheme}...")
    ppl = ppl_transfer(scheme)
    results[scheme] = ppl
    print(f"  PPL_{scheme} = {ppl:.4f}")


# ============================================================
# 汇总
# ============================================================
print()
print("=" * 80)
print("多层 K 辅助 V 映射：最终结果")
print("=" * 80)
print(f"{'方法':>35}  {'PPL':>10}  {'Ratio':>10}")
print("-" * 80)
for name, ppl in results.items():
    ratio = ppl / ppl_nat
    print(f"{name:>35}  {ppl:10.4f}  {ratio:10.4f}")
print("=" * 80)

print()
print("ΔRatio vs K1:")
for scheme in ["K3", "K5", "K7", "K_all_mean"]:
    d = (results[scheme] - results["K1"]) / ppl_nat
    print(f"  {scheme}: {d:+.4f}")