from pathlib import Path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

"""
Budget-Utility Curve Experiment
--------------------------------
目的：
1. 对 Qwen3-1.7B 的每一层测试不同 KV 保留预算；
2. 预算：[16, 32, 48, 64, 80, 96, 128]；
3. 用 attention output 与 full-KV reference 的 cosine / relative-L2 衡量质量；
4. 输出 JSON 和 CSV，方便后续画 budget-utility curve。

注意：
- 这是第一阶段的 KV 压缩敏感性实验，不是最终的跨模型迁移实验。
- 为避免位置编码/causal mask 问题，保留的是原始 token 的位置索引，而不是重新编号。
"""

import os
import json
import math
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# 配置
# =========================
MODEL_PATH = "../model/Qwen3-0.6B-base"
# MODEL_PATH = "../model/Qwen3-1.7B-base"
# MODEL_PATH = "../model/Qwen3-4B-base"
DATA_PATH = "../dataset/wikitext-02/kvtest.jsonl"

NUM_SAMPLES = 50
SEQ_LEN = 512

# 不同 KV budget
BUDGETS = [16, 32, 48, 64, 80, 96, 128]

# 如果显存/时间允许，保持 None 表示测试全部 28 层
# 例如只测试 [0, 4, 8, 12, 16, 20, 24, 27]
TEST_LAYERS = None

SEED = 42

OUTPUT_JSON = "budget_utility_curves.json"
OUTPUT_CSV = "budget_utility_curves.csv"


# =========================
# 随机种子
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 数据
# =========================
def load_text():
    import pandas as pd

    df = pd.read_json(DATA_PATH, lines=True)
    text = df["text"].str.cat(sep="\n")
    return text


def build_input(tokenizer):
    text = load_text()

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=NUM_SAMPLES * SEQ_LEN,
        padding=False,
    )

    ids = enc["input_ids"][0]

    need = NUM_SAMPLES * SEQ_LEN

    if ids.numel() < need:
        raise RuntimeError(
            f"数据不足：需要 {need} tokens，实际只有 {ids.numel()}。"
        )

    ids = ids[:need]
    ids = ids.view(NUM_SAMPLES, SEQ_LEN)

    return ids


# =========================
# 工具函数
# =========================
def cosine_similarity(a, b):
    """
    a, b: [B, Q, D]
    返回 batch 平均 cosine
    """
    a = a.float()
    b = b.float()

    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)

    return (a * b).sum(dim=-1).mean().item()


def relative_l2(a, b):
    """
    ||a-b|| / ||b||
    """
    a = a.float()
    b = b.float()

    num = torch.linalg.vector_norm(a - b)
    den = torch.linalg.vector_norm(b).clamp_min(1e-8)

    return (num / den).item()


def select_positions(seq_len, budget, device):
    """
    在原始序列位置上均匀选择 budget 个 token。
    不重新编号，因此可以继续使用原始 position ids。
    """
    if budget >= seq_len:
        return torch.arange(seq_len, device=device)

    pos = torch.linspace(
        0,
        seq_len - 1,
        steps=budget,
        device=device,
    ).round().long()

    pos = torch.unique(pos)

    # 极端情况下补齐
    if pos.numel() < budget:
        all_pos = torch.arange(seq_len, device=device)

        mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        mask[pos] = False

        extra = all_pos[mask][: budget - pos.numel()]
        pos = torch.cat([pos, extra])

        pos = torch.sort(pos).values

    return pos


def build_causal_mask_from_positions(query_positions, key_positions):
    """
    原始位置意义上的 causal mask：
    query 只能看到 key_position <= query_position。

    返回 bool mask:
    True = mask out
    False = allowed
    """
    q = query_positions[:, None]
    k = key_positions[None, :]

    return k > q


# =========================
# 单层实验
# =========================
@torch.no_grad()
def evaluate_layer(
    layer_module,
    hidden_states,
    position_ids,
    attention_mask,
    budget,
    num_heads,
    head_dim,
    rotary_emb,
    kv_heads,
):
    device = hidden_states.device
    B, T, H = hidden_states.shape

    # ========================================================
    # 第 1 步：LayerNorm（进 attention 之前）
    # ========================================================
    normed = layer_module.input_layernorm(hidden_states)

    # ========================================================
    # 第 2 步：q/k/v 线性投影
    #   输入 [B, T, H]
    #   输出 [B, T, num_heads * head_dim]
    # ========================================================
    q = layer_module.self_attn.q_proj(normed)
    k = layer_module.self_attn.k_proj(normed)
    v = layer_module.self_attn.v_proj(normed)

    # ========================================================
    # 第 3 步：reshape 成多头形状
    #   [B, T, H*D] -> [B, T, H, D]
    # ========================================================
    q = q.view(B, T, num_heads, head_dim)
    k = k.view(B, T, kv_heads, head_dim)
    v = v.view(B, T, kv_heads, head_dim)

    # ========================================================
    # 第 4 步：Q/K norm（Qwen3 特有，之前漏了！）
    #   在 transpose 之前做，和 HF 原版一致
    # ========================================================
    q = layer_module.self_attn.q_norm(q)
    k = layer_module.self_attn.k_norm(k)

    # ========================================================
    # 第 5 步：transpose 成 [B, H, T, D]
    # ========================================================
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # ========================================================
    # 第 6 步：RoPE（只作用于 q 和 k，v 不动）
    # ========================================================
    cos, sin = rotary_emb(q, position_ids)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary(x, cos, sin):
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        return (x * cos) + (rotate_half(x) * sin)

    q = apply_rotary(q, cos, sin)
    k = apply_rotary(k, cos, sin)

    # ========================================================
    # 检查：q/k/v 不能有 NaN/Inf
    # ========================================================
    if not torch.isfinite(q).all():
        raise RuntimeError(f"Q contains NaN/Inf")
    if not torch.isfinite(k).all():
        raise RuntimeError(f"K contains NaN/Inf")
    if not torch.isfinite(v).all():
        raise RuntimeError(f"V contains NaN/Inf")

    # ========================================================
    # 第 7 步：GQA 展开（kv_heads -> num_heads）
    # ========================================================
    repeat_factor = num_heads // kv_heads
    if repeat_factor > 1:
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)

    # ========================================================
    # 第 8 步：转 fp32，算 attention（避免 fp16 溢出）
    # ========================================================
    q_attn = q.float()
    k_attn = k.float()
    v_attn = v.float()

    scale = 1.0 / math.sqrt(head_dim)

    # ---------- Reference: 完整 KV ----------
    scores_full = torch.matmul(
        q_attn,
        k_attn.transpose(-1, -2),
    ) * scale

    pos = position_ids[0]
    causal_full = pos[None, :] > pos[:, None]

    scores_full = scores_full.masked_fill(
        causal_full[None, None, :, :],
        torch.finfo(torch.float32).min,
    )

    probs_full = torch.softmax(scores_full, dim=-1)
    out_full = torch.matmul(probs_full, v_attn)

    # ---------- Compressed: 只保留 budget 个 KV ----------
    selected = select_positions(T, budget, device)

    k_comp = k_attn[:, :, selected, :]
    v_comp = v_attn[:, :, selected, :]

    key_pos = pos[selected]
    query_pos = pos

    scores_comp = torch.matmul(
        q_attn,
        k_comp.transpose(-1, -2),
    ) * scale

    causal_comp = build_causal_mask_from_positions(
        query_pos,
        key_pos,
    )

    scores_comp = scores_comp.masked_fill(
        causal_comp[None, None, :, :],
        torch.finfo(torch.float32).min,
    )

    probs_comp = torch.softmax(scores_comp, dim=-1)
    out_comp = torch.matmul(probs_comp, v_comp)

    # ========================================================
    # 第 9 步：o_proj（输出投影）
    #   [B, H, T, D] -> [B, T, H*D]
    # ========================================================
    out_full = (
        out_full.transpose(1, 2)
        .contiguous()
        .view(B, T, -1)
    )
    out_comp = (
        out_comp.transpose(1, 2)
        .contiguous()
        .view(B, T, -1)
    )
    
    
    target_dtype = layer_module.self_attn.o_proj.weight.dtype

    out_full = out_full.to(target_dtype)
    out_comp = out_comp.to(target_dtype)
    
    
    out_full = layer_module.self_attn.o_proj(out_full)
    out_comp = layer_module.self_attn.o_proj(out_comp)

    # ========================================================
    # 第 10 步：算指标
    # ========================================================
    cos_sim = cosine_similarity(out_comp, out_full)
    rel = relative_l2(out_comp, out_full)

    return cos_sim, rel


# =========================
# 主程序
# =========================
def main():
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Budget-Utility Curve Experiment")
    print("=" * 70)
    print(f"Model      : {MODEL_PATH}")
    print(f"Data       : {DATA_PATH}")
    print(f"Device     : {device}")
    print(f"Samples    : {NUM_SAMPLES}")
    print(f"Seq len    : {SEQ_LEN}")
    print(f"Budgets    : {BUDGETS}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )

    model = model.to(device)
    model.eval()

    input_ids = build_input(tokenizer).to(device)

    print(f"Input shape: {tuple(input_ids.shape)}")

    # --------------------------------------------------
    # 先取得 full forward hidden states
    # --------------------------------------------------
    print("\nRunning full forward ...")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states_all = outputs.hidden_states

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # Qwen3 GQA
    kv_heads = getattr(
        model.config,
        "num_key_value_heads",
        num_heads,
    )

    # hidden_size = model.config.hidden_size
    # head_dim = hidden_size // num_heads
    
    hidden_size = model.config.hidden_size
    head_dim = getattr(
        model.config,
        "head_dim",
        hidden_size // num_heads,
    )
    
    

    if TEST_LAYERS is None:
        layers = list(range(num_layers))
    else:
        layers = TEST_LAYERS

    print(f"Layers     : {layers}")
    print(f"Q heads    : {num_heads}")
    print(f"KV heads   : {kv_heads}")
    print(f"Head dim   : {head_dim}")

    # position ids
    position_ids = torch.arange(
        SEQ_LEN,
        device=device,
    ).unsqueeze(0)

    # 结果
    results = []

    # --------------------------------------------------
    # 每层 × 每个 budget
    # --------------------------------------------------
    for layer_idx in layers:

        print()
        print("-" * 70)
        print(f"Layer {layer_idx}")
        print("-" * 70)

        layer_module = model.model.layers[layer_idx]

        # hidden_states_all[layer_idx] 是进入该 layer 的 hidden state
        hidden_states = hidden_states_all[layer_idx]

        # 逐 budget 测试
        for budget in BUDGETS:

            cos, rel = evaluate_layer(
                layer_module=layer_module,
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=None,
                budget=budget,
                num_heads=num_heads,
                head_dim=head_dim,
                rotary_emb=model.model.rotary_emb,
                kv_heads=kv_heads,
            )

            row = {
                "layer": int(layer_idx),
                "budget": int(budget),
                "cosine": float(cos),
                "rel_l2": float(rel),
            }

            results.append(row)

            print(
                f"budget={budget:3d} "
                f"cos={cos:.4f} "
                f"relL2={rel:.4f}"
            )

    # --------------------------------------------------
    # 保存 JSON
    # --------------------------------------------------
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------
    # 保存 CSV
    # --------------------------------------------------
    with open(
        OUTPUT_CSV,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "budget",
                "cosine",
                "rel_l2",
            ],
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)

    # --------------------------------------------------
    # 汇总
    # --------------------------------------------------
    print()
    print("=" * 70)
    print("Done.")
    print("=" * 70)

    print(f"Saved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_CSV}")

    # 每个 budget 的所有层平均
    print("\nGlobal mean:")
    for budget in BUDGETS:

        subset = [
            x for x in results
            if x["budget"] == budget
        ]

        mean_cos = np.mean(
            [x["cosine"] for x in subset]
        )

        mean_rel = np.mean(
            [x["rel_l2"] for x in subset]
        )

        print(
            f"budget={budget:3d} "
            f"mean_cos={mean_cos:.4f} "
            f"mean_relL2={mean_rel:.4f}"
        )


if __name__ == "__main__":
    main()
