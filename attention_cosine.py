import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"
DATA_PATH = "../dataset/wikitext-02/train.jsonl"
CACHE_DIR = "./cache_kv"

NUM_SAMPLES = 50
SEQ_LEN = 512
MAX_TOKENS = 2000
SEED = 42
DEVICE = "cuda"

os.makedirs(CACHE_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------- 数据 ----------
df = pd.read_json(DATA_PATH, lines=True)
text = df["text"].str.cat(sep="\n")
tok = AutoTokenizer.from_pretrained(MODEL_A)
ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
need = NUM_SAMPLES * SEQ_LEN
ids = ids[:need].reshape(NUM_SAMPLES, SEQ_LEN).to(DEVICE)

# 用连续前 MAX_TOKENS 个 token，causal mask 才有意义
token_idx = torch.arange(MAX_TOKENS)

# ---------- 提取 Q、K、V ----------
@torch.no_grad()
def extract_qkv(model_path, tag):
    path = os.path.join(CACHE_DIR, f"{tag}_qkv.pt")
    if os.path.exists(path):
        print(f"Loading cached {tag} QKV...")
        return torch.load(path)

    print(f"Extracting {tag} QKV...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16, attn_implementation="eager"
    ).to(DEVICE).eval()

    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states

    n_layers = model.config.num_hidden_layers
    D = model.config.head_dim
    H_q = model.config.num_attention_heads
    H_kv = model.config.num_key_value_heads

    qkv_list = []
    for l in range(n_layers):
        layer = model.model.layers[l]
        h = hidden[l + 1]                     # [B, L, D_model]
        B, L, _ = h.shape

        # 用 q_proj / k_proj / v_proj 直接算，形状可控
        Q = layer.self_attn.q_proj(h).view(B, L, H_q,  D).transpose(1, 2)
        K = layer.self_attn.k_proj(h).view(B, L, H_kv, D).transpose(1, 2)
        V = layer.self_attn.v_proj(h).view(B, L, H_kv, D).transpose(1, 2)

        # [B, H, L, D] -> [B*L, H, D]，取前 MAX_TOKENS 个
        Q = Q.reshape(-1, H_q,  D)[:MAX_TOKENS].float().cpu()
        K = K.reshape(-1, H_kv, D)[:MAX_TOKENS].float().cpu()
        V = V.reshape(-1, H_kv, D)[:MAX_TOKENS].float().cpu()
        qkv_list.append((Q, K, V))

    del model
    torch.cuda.empty_cache()
    torch.save(qkv_list, path)
    return qkv_list

qkv_A = extract_qkv(MODEL_A, "A")
qkv_B = extract_qkv(MODEL_B, "B")

L_A = len(qkv_A)
L_B = len(qkv_B)
print(f"Layers: A={L_A}, B={L_B}")

# ---------- Attention 输出 cosine ----------
def attention_output_cosine(Q, K_src, V_src, K_tgt, V_tgt, causal=True):
    """
    Q: [N, H_q, D]
    K_src, V_src: [N, H_kv, D]
    K_tgt, V_tgt: [N, H_kv, D]
    GQA：把 KV 头 repeat 到 Q 头数。
    """
    N, H_q, D = Q.shape
    H_kv = K_src.shape[1]
    if H_q != H_kv:
        rep = H_q // H_kv
        K_src = K_src.repeat_interleave(rep, dim=1)
        V_src = V_src.repeat_interleave(rep, dim=1)
        K_tgt = K_tgt.repeat_interleave(rep, dim=1)
        V_tgt = V_tgt.repeat_interleave(rep, dim=1)

    scale = D ** 0.5
    if causal:
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        attn_mask = torch.zeros(N, N, dtype=Q.dtype)
        attn_mask.masked_fill_(mask, torch.finfo(Q.dtype).min)
    else:
        attn_mask = torch.zeros(N, N, dtype=Q.dtype)

    # 转成 [H_q, N, D]，用 bmm 代替 einsum
    Q_h     = Q.transpose(0, 1).contiguous()       # [H_q, N, D]
    K_src_h = K_src.transpose(0, 1).contiguous()   # [H_q, N, D]
    V_src_h = V_src.transpose(0, 1).contiguous()
    K_tgt_h = K_tgt.transpose(0, 1).contiguous()
    V_tgt_h = V_tgt.transpose(0, 1).contiguous()

    logits_src = torch.bmm(Q_h, K_src_h.transpose(1, 2)) / scale + attn_mask
    logits_tgt = torch.bmm(Q_h, K_tgt_h.transpose(1, 2)) / scale + attn_mask

    out_src = torch.bmm(torch.softmax(logits_src, dim=-1), V_src_h)  # [H_q, N, D]
    out_tgt = torch.bmm(torch.softmax(logits_tgt, dim=-1), V_tgt_h)

    out_src = out_src / out_src.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    out_tgt = out_tgt / out_tgt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cos = (out_src * out_tgt).sum(dim=-1).mean()   # [H_q, N] 平均
    return cos.item()

# ---------- 计算矩阵 ----------
print("Computing attention-output cosine matrix...")
attn_cos = np.zeros((L_A, L_B))
for i in range(L_B):
    Q_i, K_i, V_i = qkv_B[i]
    for j in range(L_A):
        _, K_j, V_j = qkv_A[j]
        attn_cos[j, i] = attention_output_cosine(Q_i, K_j, V_j, K_i, V_i, causal=True)
    if (i + 1) % 7 == 0:
        print(f"  col {i+1}/{L_B}")

np.save("attn_cos_real.npy", attn_cos)

# ---------- 统计 ----------
diag = np.array([attn_cos[i, i] for i in range(min(L_A, L_B))])
mask = ~np.eye(L_A, L_B, dtype=bool)
off = attn_cos[mask]
best = [int(np.argmax(attn_cos[i])) for i in range(L_A)]
on_diag = sum(1 for i, j in enumerate(best) if i == j)

print("\n" + "="*60)
print("Attention-output cosine (proxy QKV, no RoPE)")
print("="*60)
print(f"Diagonal mean   : {diag.mean():.4f}")
print(f"Off-diag mean   : {off.mean():.4f}")
print(f"Diag - offdiag  : {diag.mean() - off.mean():+.4f}")
print(f"Best match on diag: {on_diag}/{L_A}")
print(f"Mean |best_j - i| : {np.mean([abs(j-i) for i,j in enumerate(best)]):.2f}")