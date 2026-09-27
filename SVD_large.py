import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置
# ============================================================
MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"
TRAIN_PATH = "../dataset/wikitext-02/train.jsonl"
TEST_PATH  = "../dataset/wikitext-02/test.jsonl"

MAX_ARTICLES_TRAIN = 2000
MAX_ARTICLES_TEST  = 1400
MAX_TOKENS_PER_ART = 256
LAYERS_TO_TEST = [0, 7, 14, 21, 27]
R_VALUES = [8, 16, 32, 64, 128, 256, 512]

DEVICE = "cuda"
SEED = 42
BATCH_SIZE = 32

torch.manual_seed(SEED)

# ============================================================
# 数据准备
# ============================================================
def prepare(path, max_articles, max_tokens):
    df = pd.read_json(path, lines=True)
    texts = df["text"].iloc[:max_articles].tolist()
    tok = AutoTokenizer.from_pretrained(MODEL_A)

    article_ids = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        ids = ids[:max_tokens]
        if len(ids) >= 16:
            article_ids.append(ids)

    max_len = max(len(x) for x in article_ids)
    padded = torch.zeros(len(article_ids), max_len, dtype=torch.long)
    mask = torch.zeros(len(article_ids), max_len, dtype=torch.bool)
    for i, ids in enumerate(article_ids):
        padded[i, :len(ids)] = ids
        mask[i, :len(ids)] = True

    print(f"  {path}: {len(article_ids)} 篇, max_len={max_len}, 有效 token={mask.sum().item()}")
    return padded, mask

print("准备数据...")
padded_train, mask_train = prepare(TRAIN_PATH, MAX_ARTICLES_TRAIN, MAX_TOKENS_PER_ART)
padded_test,  mask_test  = prepare(TEST_PATH,  MAX_ARTICLES_TEST,  MAX_TOKENS_PER_ART)

# ============================================================
# 提取 V cache
# ============================================================
@torch.no_grad()
def extract_v(model, layer_idx, padded):
    """
    返回 [N, D]，只含真实 token（用 mask 筛选）。
    """
    attn = model.model.layers[layer_idx].self_attn
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)

    all_v = []
    for start in range(0, len(padded), BATCH_SIZE):
        batch = padded[start:start+BATCH_SIZE].to(DEVICE)
        B_b, T_b = batch.shape

        out = model.model(input_ids=batch, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[layer_idx]
        normed = model.model.layers[layer_idx].input_layernorm(hidden)

        v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
        v = v.transpose(1, 2).reshape(B_b, T_b, -1).float().cpu()
        all_v.append(v)

        del out, hidden, normed, v
        torch.cuda.empty_cache()

    V = torch.cat(all_v, dim=0).reshape(-1, all_v[0].shape[-1])   # [B*T, D]
    return V

# ============================================================
# 加载模型
# ============================================================
print("\n加载模型...")
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()

# ============================================================
# 共享基：联合 SVD
# ============================================================
def find_shared_basis(X_S_train, X_T_train, r):
    """
    在训练集上做联合 SVD。
    返回 V_S [d_S, r], V_T [d_T, r], mean_S [1, d_S], mean_T [1, d_T]
    """
    mean_S = X_S_train.mean(dim=0, keepdim=True)
    mean_T = X_T_train.mean(dim=0, keepdim=True)
    X_S = X_S_train - mean_S
    X_T = X_T_train - mean_T

    X_cat = torch.cat([X_S, X_T], dim=1)     # [N, d_S + d_T]
    # 只算前 r 个奇异向量，省内存
    U, S, Vh = torch.linalg.svd(X_cat, full_matrices=False)
    V = Vh[:r].T                              # [d_S + d_T, r]

    d_S = X_S.shape[1]
    V_S = V[:d_S]
    V_T = V[d_S:]
    return V_S, V_T, mean_S, mean_T

def eval_basis(X_S_test, X_T_test, V_S, V_T, mean_S, mean_T):
    """
    在测试集上评估重建质量。
    Sharer 投影：A = (X_S - mean_S) @ V_S
    Receiver 重建：X_T_hat = A @ V_T^T + mean_T
    """
    X_S_c = X_S_test - mean_S
    A = X_S_c @ V_S                          # [N, r]
    X_T_hat = A @ V_T.T + mean_T

    residual = ((X_T_hat - X_T_test) ** 2).sum()
    total = ((X_T_test - mean_T) ** 2).sum()
    return (1 - residual / total).item()

# ============================================================
# 主循环：每一层、每个 rank
# ============================================================
results = {}   # results[layer][r] = (train_r2, test_r2)

for layer in LAYERS_TO_TEST:
    print(f"\n{'='*60}")
    print(f"Layer {layer}")
    print(f"{'='*60}")

    # 提取 V
    V_A_train = extract_v(model_A, layer, padded_train)
    V_B_train = extract_v(model_B, layer, padded_train)
    V_A_test  = extract_v(model_A, layer, padded_test)
    V_B_test  = extract_v(model_B, layer, padded_test)

    # mask 筛选：只保留真实 token
    mtr = mask_train.reshape(-1)
    mte = mask_test.reshape(-1)
    V_A_train = V_A_train[mtr]
    V_B_train = V_B_train[mtr]
    V_A_test  = V_A_test[mte]
    V_B_test  = V_B_test[mte]

    print(f"  Train tokens: {V_A_train.shape}, Test tokens: {V_A_test.shape}")

    results[layer] = {}

    for r in R_VALUES:
        # 在 train 上拟合
        V_S, V_T, mS, mT = find_shared_basis(V_A_train, V_B_train, r)

        # 在 train 上评估
        r2_train = eval_basis(V_A_train, V_B_train, V_S, V_T, mS, mT)
        # 在 test 上评估
        r2_test = eval_basis(V_A_test, V_B_test, V_S, V_T, mS, mT)

        results[layer][r] = (r2_train, r2_test)
        print(f"  r={r:>4}: train R²={r2_train:.4f}, test R²={r2_test:.4f}")

    # 释放显存
    del V_A_train, V_B_train, V_A_test, V_B_test
    torch.cuda.empty_cache()

# ============================================================
# 汇总表
# ============================================================
print(f"\n{'='*80}")
print("汇总：test R²（train 拟合，test 评估）")
print(f"{'='*80}")
header = f"{'Layer':>6}  " + "  ".join(f"r={r:<5}" for r in R_VALUES)
print(header)
print("-" * len(header))
for layer in LAYERS_TO_TEST:
    row = f"{layer:>6}  "
    for r in R_VALUES:
        _, r2_test = results[layer][r]
        row += f"{r2_test:>7.4f}  "
    print(row)

# ============================================================
# 关键阈值：达到 test R² > 0.9 所需最小 rank
# ============================================================
print(f"\n{'='*60}")
print("达到 test R² > 0.90 所需的最小 rank")
print(f"{'='*60}")
for layer in LAYERS_TO_TEST:
    best_r = None
    for r in R_VALUES:
        _, r2_test = results[layer][r]
        if r2_test > 0.90:
            best_r = r
            break
    if best_r is not None:
        _, r2_test = results[layer][best_r]
        print(f"  Layer {layer:>2}: r={best_r:>4}  (test R²={r2_test:.4f})")
    else:
        _, r2_max = results[layer][R_VALUES[-1]]
        print(f"  Layer {layer:>2}: 即使 r={R_VALUES[-1]}，test R²={r2_max:.4f} < 0.90")