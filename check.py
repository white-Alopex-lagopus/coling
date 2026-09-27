import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置
# ============================================================
MODEL_PATH = "../model/Qwen3-0.6B-base"
DATA_PATH = "../dataset/wikitext-02/train.jsonl"
DEVICE = "cuda"

# ============================================================
# 1. 算 token 数
# ============================================================
print("=" * 70)
print("[1] Token 统计")
print("=" * 70)

df = pd.read_json(DATA_PATH, lines=True)
print(f"条数: {len(df)}")

tok = AutoTokenizer.from_pretrained(MODEL_PATH)
lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in df["text"]]

print(f"总 token: {sum(lengths):,}")
print(f"平均    : {sum(lengths) / len(lengths):.0f}")
print(f"最小    : {min(lengths)}")
print(f"最大    : {max(lengths)}")
print(f"中位    : {sorted(lengths)[len(lengths) // 2]}")


# ============================================================
# 2. 算同文章 vs 跨文章的 hidden 相似度
# ============================================================
print()
print("=" * 70)
print("[2] Hidden 相似度")
print("=" * 70)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.float16
).to(DEVICE).eval()

# 取前 2 篇文章
texts = df["text"].iloc[:2].tolist()

with torch.no_grad():
    hiddens = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
        ids = ids[:, :1024]  # 只取前 1024 token，省显存
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        # 取中间层，比如第 14 层
        h = out.hidden_states[14][0].float()  # [T, D]
        hiddens.append(h)
        print(f"  文章 {len(hiddens)-1}: {h.shape}")

h1, h2 = hiddens  # 文章 0, 文章 1

# 同文章：文章 0 内相邻 token
cos_same_adj = F.cosine_similarity(h1[0:1], h1[1:2]).item()
# 同文章：文章 0 内间隔 100 的 token
cos_same_far = F.cosine_similarity(h1[0:1], h1[100:101]).item()
# 跨文章：文章 0 的 token 0 vs 文章 1 的 token 0
cos_diff = F.cosine_similarity(h1[0:1], h2[0:1]).item()

# 同文章内的平均相似度（随机采样 100 对）
torch.manual_seed(42)
idx_a = torch.randint(0, h1.shape[0], (100,))
idx_b = torch.randint(0, h1.shape[0], (100,))
cos_within = F.cosine_similarity(h1[idx_a], h1[idx_b], dim=-1).mean().item()

# 跨文章的平均相似度
idx_c = torch.randint(0, h2.shape[0], (100,))
cos_across = F.cosine_similarity(h1[idx_a], h2[idx_c], dim=-1).mean().item()

print()
print(f"同文章-相邻 token     : {cos_same_adj:.4f}")
print(f"同文章-间隔 100 token : {cos_same_far:.4f}")
print(f"跨文章-token 0 vs 0   : {cos_diff:.4f}")
print(f"同文章-随机 100 对平均 : {cos_within:.4f}")
print(f"跨文章-随机 100 对平均 : {cos_across:.4f}")

print()
print("=" * 70)
print("解读")
print("=" * 70)
print(f"同文章平均: {cos_within:.4f}")
print(f"跨文章平均: {cos_across:.4f}")
print(f"差值      : {cos_within - cos_across:.4f}")
print()
if cos_within - cos_across > 0.3:
    print("→ 同文章高度相关，有效样本数 ≈ 文章数")
else:
    print("→ 同文章相关性不强，token 可近似独立")