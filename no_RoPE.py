import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置
# ============================================================
MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"
DATA_PATH = "../dataset/wikitext-02/train.jsonl"

MAX_ARTICLES = 9007
MAX_TOKENS_PER_ART = 256
LAYER_A = 14          # 用中间层
LAYER_B = 14
DEVICE = "cuda"
SEED = 42

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

print(f"文章数: {len(article_ids)}, 每篇最多 {max_len} token")
print(f"总 token: {len(article_ids) * max_len}")


# ============================================================
# 提取 K（RoPE 之前）
# ============================================================
@torch.no_grad()
def extract_k_pre_rope(model, layer_idx, padded, use_rope=False):
    """
    返回:
        K_flat: [N_tokens, H_kv * D_kv]   RoPE 前或后的 K
        positions: [N_tokens]             每个 token 的位置
    """
    attn = model.model.layers[layer_idx].self_attn
    n_kv_heads = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)

    all_k = []
    all_pos = []
    batch_size = 64

    for start in range(0, len(padded), batch_size):
        batch = padded[start:start+batch_size].to(DEVICE)
        B, T = batch.shape

        # 提取该层输入 hidden（用 model.model 跳过 lm_head）
        out = model.model(input_ids=batch, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[layer_idx]          # [B, T, D]

        # 过 layernorm（和 attn 内部一致）
        normed = model.model.layers[layer_idx].input_layernorm(hidden)

        # k_proj
        k = attn.k_proj(normed)                        # [B, T, H_kv*D_kv]
        k = k.view(B, T, n_kv_heads, head_dim)
        k = attn.k_norm(k)                             # RMSNorm
        # [B, T, H_kv, D]  →  [B, H_kv, T, D]
        k = k.transpose(1, 2)

        if use_rope:
            pos_ids = torch.arange(T, device=DEVICE).unsqueeze(0).expand(B, -1)
            cos, sin = model.model.rotary_emb(k, pos_ids)
            k = apply_rotary(k, cos, sin)

        # 拉平成 [B, T, H_kv*D]
        k = k.transpose(1, 2).reshape(B, T, -1)

        all_k.append(k.float().cpu())
        all_pos.append(torch.arange(T).unsqueeze(0).expand(B, -1))

        del out, hidden, normed, k
        torch.cuda.empty_cache()

    K = torch.cat(all_k, dim=0)                # [N_articles, T, H_kv*D]
    P = torch.cat(all_pos, dim=0)              # [N_articles, T]

    # 展平为 [N_tokens, D]
    K = K.reshape(-1, K.shape[-1])
    P = P.reshape(-1)

    return K, P


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x, cos, sin):
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin


# ============================================================
# 加载模型
# ============================================================
print("\n加载 0.6B...")
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
print("加载 1.7B...")
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()


# ============================================================
# 提取 K（无 RoPE 版）
# ============================================================
print("\n提取 0.6B 的 K (pre-RoPE)...")
K_A, pos_A = extract_k_pre_rope(model_A, LAYER_A, padded, use_rope=False)
print(f"  K_A: {K_A.shape}")

print("提取 1.7B 的 K (pre-RoPE)...")
K_B, pos_B = extract_k_pre_rope(model_B, LAYER_B, padded, use_rope=False)
print(f"  K_B: {K_B.shape}")

# ============================================================
# 位置分组评估
# ============================================================
def ridge_r2_by_position(K_A, K_B, pos, lam=100.0, n_buckets=4):
    """
    用一部分位置训练 W，其他位置测 R²
    """
    D_A = K_A.shape[-1]
    D_B = K_B.shape[-1]

    # 计算全局 XtX, XtY（用全部数据）
    # 这里简化：为了看位置泛化，用「leave-bucket-out」
    results = {}

    max_pos = pos.max().item() + 1
    bucket_size = max_pos // n_buckets

    for b in range(n_buckets):
        lo = b * bucket_size
        hi = (b + 1) * bucket_size if b < n_buckets - 1 else max_pos

        test_mask = (pos >= lo) & (pos < hi)
        train_mask = ~test_mask

        X_tr = K_A[train_mask].float()
        Y_tr = K_B[train_mask].float()
        X_te = K_A[test_mask].float()
        Y_te = K_B[test_mask].float()

        # 标准化
        X_m, X_s = X_tr.mean(0, keepdim=True), X_tr.std(0, keepdim=True).clamp_min(1e-6)
        Y_m, Y_s = Y_tr.mean(0, keepdim=True), Y_tr.std(0, keepdim=True).clamp_min(1e-6)

        X_tr = (X_tr - X_m) / X_s
        Y_tr = (Y_tr - Y_m) / Y_s
        X_te = (X_te - X_m) / X_s
        Y_te = (Y_te - Y_m) / Y_s

        XtX = X_tr.T @ X_tr + lam * torch.eye(D_A)
        XtY = X_tr.T @ Y_tr
        W = torch.linalg.solve(XtX, XtY)

        XW = X_te @ W
        r2 = (1 - ((XW - Y_te)**2).sum() / (Y_te**2).sum()).item()

        results[f"[{lo}, {hi})"] = r2
        print(f"  位置 {lo:4d}~{hi:4d}: R² = {r2:.4f}  (n_test={test_mask.sum().item()})")

    return results


print("\n" + "=" * 70)
print("位置泛化评估（无 RoPE）")
print("=" * 70)
results_no_rope = ridge_r2_by_position(K_A, K_B, pos_A, lam=100.0, n_buckets=4)


# ============================================================
# 对比：有 RoPE 的 K
# ============================================================
print("\n提取 0.6B 的 K (with RoPE)...")
K_A_rope, _ = extract_k_pre_rope(model_A, LAYER_A, padded, use_rope=True)
print("提取 1.7B 的 K (with RoPE)...")
K_B_rope, _ = extract_k_pre_rope(model_B, LAYER_B, padded, use_rope=True)

print("\n" + "=" * 70)
print("位置泛化评估（有 RoPE）")
print("=" * 70)
results_rope = ridge_r2_by_position(K_A_rope, K_B_rope, pos_A, lam=100.0, n_buckets=4)


# ============================================================
# 汇总
# ============================================================
print()
print("=" * 70)
print("对比：位置泛化")
print("=" * 70)
print(f"{'位置段':>15}  {'无RoPE R²':>12}  {'有RoPE R²':>12}  {'Δ':>10}")
print("-" * 70)

for key in results_no_rope:
    r2_no = results_no_rope[key]
    r2_yes = results_rope[key]
    print(f"{key:>15}  {r2_no:12.4f}  {r2_yes:12.4f}  {r2_no - r2_yes:+10.4f}")

print()
print("解读：")
print("  无 RoPE：不同位置 R² 应该相近 → 映射位置无关")
print("  有 RoPE：不同位置 R² 应该差异大 → 映射学到位置信息")