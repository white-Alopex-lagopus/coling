from pathlib import Path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
oracle_budget_allocation.py

目的
----
在固定总 KV budget 下，比较：

1. Uniform：
   所有 layer 使用相同 KV budget。

2. Oracle：
   根据前一步 layer sensitivity 实验得到的 sensitivity score，
   把更多 KV budget 分配给敏感 layer，把更少 budget 分配给不敏感 layer。

这是一个“可行性验证”实验，不是最终 adaptive compression 方法。

如果 Oracle 在相同平均 budget 下明显优于 Uniform，
说明 layer-wise adaptive KV budget 值得继续研究。

默认：
    Qwen3-1.7B-base
    50 samples
    512 tokens
    平均 KV budget = 64

注意：
----
本实验沿用 v2 的“原始 token selection + 原始 position + causal mask”设计。
Oracle 的 token 数在不同 layer 可以不同，但选择方式保持一致：
均匀选择原始 token positions。
"""

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 1. 配置
# ============================================================

MODEL_PATH = "../model/Qwen3-1.7B-base"
DATA_PATH = "../dataset/wikitext-02/kvtest.jsonl"

NUM_SAMPLES = 50
SEQ_LEN = 512

# Uniform baseline
UNIFORM_BUDGET = 64

# ============================================================
# Oracle budget
#
# 这些 budget 来自 layer sensitivity：
#
# 高敏感层 -> 96
# 中等层   -> 64
# 低敏感层 -> 32
#
# 最终会自动检查平均 budget。
# ============================================================

HIGH_BUDGET = 96
MID_BUDGET = 64
LOW_BUDGET = 32

# 如果你想直接指定 layer，可以在下面填。
# None = 根据 sensitivity score 自动排序。

MANUAL_HIGH_LAYERS = None
MANUAL_LOW_LAYERS = None

# 高/低敏感层数量
NUM_HIGH_LAYERS = 7
NUM_LOW_LAYERS = 7

# K/V 都压缩
TEST_K = True
TEST_V = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = Path("./oracle_budget_results")
OUTPUT_JSON = OUTPUT_DIR / "oracle_budget_results.json"

SEED = 42


# ============================================================
# 2. 你刚才得到的 layer sensitivity
# ============================================================

# cosine from layer_compression_sensitivity_v2.py
SENSITIVITY_COSINE = {
    0: 0.5205,
    1: 0.5062,
    2: 0.4494,
    3: 0.5892,
    4: 0.9562,
    5: 0.9952,
    6: 0.5704,
    7: 0.9961,
    8: 0.9830,
    9: 0.9500,
    10: 0.7699,
    11: 0.7397,
    12: 0.6700,
    13: 0.6456,
    14: 0.5919,
    15: 0.6211,
    16: 0.7451,
    17: 0.6305,
    18: 0.5765,
    19: 0.8027,
    20: 0.7772,
    21: 0.8365,
    22: 0.7972,
    23: 0.6817,
    24: 0.6852,
    25: 0.7460,
    26: 0.8339,
    27: 0.8717,
}


# ============================================================
# 3. 工具函数
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_finite(name, x):
    if x is None:
        return True

    if not torch.isfinite(x).all():
        print(
            f"[WARNING] {name} contains NaN/Inf"
        )
        return False

    return True


def cosine_similarity_tokens(a, b):
    a = a.float()
    b = b.float()

    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1])

    sim = F.cosine_similarity(
        a,
        b,
        dim=-1
    )

    if not torch.isfinite(sim).all():
        return float("nan")

    return sim.mean().item()


def make_uniform_indices(
    seq_len,
    compressed_len,
    device
):
    if compressed_len >= seq_len:
        return torch.arange(
            seq_len,
            device=device
        )

    idx = torch.linspace(
        0,
        seq_len - 1,
        steps=compressed_len,
        device=device
    ).round().long()

    return torch.unique(idx)


def build_selected_causal_mask(
    query_positions,
    key_positions
):
    return (
        key_positions.unsqueeze(0)
        <= query_positions.unsqueeze(1)
    )


def repeat_kv(
    hidden_states,
    n_rep
):
    if n_rep == 1:
        return hidden_states

    b, h, t, d = hidden_states.shape

    hidden_states = hidden_states[:, :, None, :, :]

    hidden_states = hidden_states.expand(
        b,
        h,
        n_rep,
        t,
        d
    )

    return hidden_states.reshape(
        b,
        h * n_rep,
        t,
        d
    )


# ============================================================
# 4. 根据 sensitivity 构造 Oracle budget
# ============================================================

def build_oracle_budget(num_layers):
    """
    sensitivity = 1 - cosine

    cosine 越低 -> sensitivity 越高。

    默认：
        最敏感 7 层 -> 96
        中间     14 层 -> 64
        最不敏感 7 层 -> 32

    平均：
        (7*96 + 14*64 + 7*32) / 28 = 64
    """

    if MANUAL_HIGH_LAYERS is not None:
        high_layers = set(MANUAL_HIGH_LAYERS)
    else:
        sorted_layers = sorted(
            range(num_layers),
            key=lambda x: SENSITIVITY_COSINE[x]
        )

        # cosine 越低越敏感
        high_layers = set(
            sorted_layers[:NUM_HIGH_LAYERS]
        )

    if MANUAL_LOW_LAYERS is not None:
        low_layers = set(MANUAL_LOW_LAYERS)
    else:
        sorted_layers = sorted(
            range(num_layers),
            key=lambda x: SENSITIVITY_COSINE[x],
            reverse=True
        )

        # cosine 越高越不敏感
        low_layers = set(
            sorted_layers[:NUM_LOW_LAYERS]
        )

    budgets = {}

    for layer in range(num_layers):

        if layer in high_layers:
            budgets[layer] = HIGH_BUDGET

        elif layer in low_layers:
            budgets[layer] = LOW_BUDGET

        else:
            budgets[layer] = MID_BUDGET

    return budgets, high_layers, low_layers


# ============================================================
# 5. RoPE + QKV
# ============================================================

def apply_qwen_rope(
    q,
    k,
    position_ids,
    rotary_emb
):
    cos, sin = rotary_emb(
        q,
        position_ids
    )

    from transformers.models.qwen3.modeling_qwen3 import (
        apply_rotary_pos_emb
    )

    q_embed, k_embed = apply_rotary_pos_emb(
        q,
        k,
        cos,
        sin
    )

    return q_embed, k_embed


def get_layer_qkv(
    model,
    hidden_states,
    layer_idx,
    position_ids
):
    layer = model.model.layers[
        layer_idx
    ]

    attn = layer.self_attn

    bsz, seq_len, _ = hidden_states.shape

    q = attn.q_proj(
        hidden_states
    )

    k = attn.k_proj(
        hidden_states
    )

    v = attn.v_proj(
        hidden_states
    )

    head_dim = getattr(
        model.config,
        "head_dim",
        model.config.hidden_size
        // model.config.num_attention_heads
    )

    num_heads = (
        model.config.num_attention_heads
    )

    num_kv_heads = (
        model.config.num_key_value_heads
    )

    q = q.view(
        bsz,
        seq_len,
        num_heads,
        head_dim
    ).transpose(1, 2)

    k = k.view(
        bsz,
        seq_len,
        num_kv_heads,
        head_dim
    ).transpose(1, 2)

    v = v.view(
        bsz,
        seq_len,
        num_kv_heads,
        head_dim
    ).transpose(1, 2)

    if hasattr(attn, "q_norm") and attn.q_norm is not None:
        q = attn.q_norm(q)

    if hasattr(attn, "k_norm") and attn.k_norm is not None:
        k = attn.k_norm(k)

    q, k = apply_qwen_rope(
        q,
        k,
        position_ids,
        # attn.rotary_emb
        model.model.rotary_emb
    )

    return q, k, v


# ============================================================
# 6. Attention
# ============================================================

def attention_with_selected_kv(
    q,
    k,
    v,
    key_positions
):
    bsz, num_heads, q_len, head_dim = q.shape

    num_kv_heads = k.shape[1]

    n_rep = (
        num_heads // num_kv_heads
    )

    k = repeat_kv(
        k,
        n_rep
    )

    v = repeat_kv(
        v,
        n_rep
    )

    scale = 1.0 / math.sqrt(
        head_dim
    )

    scores = torch.matmul(
        q,
        k.transpose(-1, -2)
    ) * scale

    query_positions = torch.arange(
        q_len,
        device=q.device
    )

    causal = build_selected_causal_mask(
        query_positions,
        key_positions
    )

    causal = causal.unsqueeze(
        0
    ).unsqueeze(
        0
    )

    scores = scores.masked_fill(
        ~causal,
        torch.finfo(
            scores.dtype
        ).min
    )

    valid_count = (
        causal.squeeze(0)
        .squeeze(0)
        .sum(dim=-1)
    )

    # 避免 softmax(all -inf)
    if (valid_count == 0).any():

        empty_queries = (
            valid_count == 0
        )

        scores[
            :,
            :,
            empty_queries,
            0
        ] = 0.0

    attn_weights = torch.softmax(
        scores.float(),
        dim=-1
    ).to(q.dtype)

    out = torch.matmul(
        attn_weights,
        v
    )

    return out


# ============================================================
# 7. 单层比较
# ============================================================

@torch.no_grad()
def evaluate_layer(
    q,
    k,
    v,
    selected_indices,
    test_k,
    test_v
):
    device = q.device

    full_positions = torch.arange(
        k.shape[2],
        device=device
    )

    # Full KV
    full_out = attention_with_selected_kv(
        q,
        k,
        v,
        full_positions
    )

    if not check_finite(
        "full_out",
        full_out
    ):
        return None

    # Compressed KV
    k_comp = (
        k[:, :, selected_indices, :]
        if test_k
        else k
    )

    v_comp = (
        v[:, :, selected_indices, :]
        if test_v
        else v
    )

    key_positions = (
        selected_indices
        if test_k
        else full_positions
    )

    compressed_out = attention_with_selected_kv(
        q,
        k_comp,
        v_comp,
        key_positions
    )

    if not check_finite(
        "compressed_out",
        compressed_out
    ):
        return None

    cosine = cosine_similarity_tokens(
        full_out,
        compressed_out
    )

    mse = F.mse_loss(
        compressed_out.float(),
        full_out.float()
    ).item()

    diff_norm = torch.norm(
        compressed_out.float()
        - full_out.float()
    )

    full_norm = torch.norm(
        full_out.float()
    )

    relative_l2 = (
        diff_norm
        / (full_norm + 1e-8)
    ).item()

    return {
        "cosine": cosine,
        "mse": mse,
        "relative_l2": relative_l2
    }


# ============================================================
# 8. 运行一个 budget strategy
# ============================================================

@torch.no_grad()
def run_strategy(
    model,
    input_ids_all,
    budgets,
    strategy_name
):
    print("\n")
    print("=" * 70)
    print(strategy_name)
    print("=" * 70)

    tokenizer_seq_len = (
        input_ids_all.shape[1]
    )

    num_layers = len(
        model.model.layers
    )

    # embedding
    hidden_states = model.model.embed_tokens(
        input_ids_all
    )

    position_ids = torch.arange(
        tokenizer_seq_len,
        device=input_ids_all.device
    ).unsqueeze(0).expand(
        input_ids_all.shape[0],
        -1
    )

    results = []

    for layer_idx in range(num_layers):

        budget = budgets[layer_idx]

        print(
            f"[{layer_idx:02d}] "
            f"budget={budget:3d}",
            end=" "
        )

        layer_input = hidden_states

        q, k, v = get_layer_qkv(
            model,
            layer_input,
            layer_idx,
            position_ids
        )

        if not (
            check_finite("q", q)
            and check_finite("k", k)
            and check_finite("v", v)
        ):
            print("NON_FINITE_QKV")

            results.append({
                "layer": layer_idx,
                "budget": budget,
                "status": "non_finite_qkv"
            })

            continue

        selected_indices = make_uniform_indices(
            tokenizer_seq_len,
            budget,
            input_ids_all.device
        )

        result = evaluate_layer(
            q,
            k,
            v,
            selected_indices,
            TEST_K,
            TEST_V
        )

        if result is None:

            print(
                "NON_FINITE_ATTENTION"
            )

            results.append({
                "layer": layer_idx,
                "budget": budget,
                "status": "non_finite_attention"
            })

        else:

            print(
                f"cos={result['cosine']:.4f} "
                f"relL2={result['relative_l2']:.4f}"
            )

            results.append({
                "layer": layer_idx,
                "budget": budget,
                "status": "ok",
                **result
            })

        # ----------------------------------------------------
        # 真实 full forward，进入下一层
        # ----------------------------------------------------

        layer = model.model.layers[
            layer_idx
        ]

        residual = layer_input

        hidden_norm = (
            layer.input_layernorm(
                layer_input
            )
        )
        
        cos, sin = model.model.rotary_emb(hidden_norm, position_ids)

        T_len = hidden_norm.shape[1]
        causal_mask = torch.triu(
            torch.full(
                (1, 1, T_len, T_len),
                torch.finfo(hidden_norm.dtype).min,
                device=hidden_norm.device,
                dtype=hidden_norm.dtype,
            ),
            diagonal=1,
        )

        attn_output = layer.self_attn(
            hidden_norm,
            # position_ids=position_ids
            position_embeddings=(cos, sin),
            attention_mask=causal_mask,
        )

        if isinstance(
            attn_output,
            tuple
        ):
            attn_output = attn_output[0]

        hidden_states = (
            residual
            + attn_output
        )

        residual = hidden_states

        hidden_norm = (
            layer.post_attention_layernorm(
                hidden_states
            )
        )

        mlp_output = layer.mlp(
            hidden_norm
        )

        hidden_states = (
            residual
            + mlp_output
        )

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return results


# ============================================================
# 9. 主函数
# ============================================================

@torch.no_grad()
def main():

    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("Oracle vs Uniform KV Budget Allocation")
    print("=" * 70)

    print(
        f"Model       : {MODEL_PATH}"
    )

    print(
        f"Dataset     : {DATA_PATH}"
    )

    print(
        f"Samples     : {NUM_SAMPLES}"
    )

    print(
        f"Seq len     : {SEQ_LEN}"
    )

    print(
        f"Uniform     : {UNIFORM_BUDGET}"
    )

    print(
        f"Oracle      : {HIGH_BUDGET}/{MID_BUDGET}/{LOW_BUDGET}"
    )

    print(
        f"Device      : {DEVICE}"
    )

    # --------------------------------------------------------
    # model
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    dtype = (
        torch.bfloat16
        if DEVICE == "cuda"
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        trust_remote_code=True
    )

    model.to(DEVICE)
    model.eval()

    # --------------------------------------------------------
    # dataset
    # --------------------------------------------------------

    df = pd.read_json(
        DATA_PATH,
        lines=True
    )

    text = df["text"].str.cat(
        sep="\n"
    )

    tokens = tokenizer(
        text,
        return_tensors="pt"
    ).input_ids[0]

    required = (
        NUM_SAMPLES * SEQ_LEN
    )

    if len(tokens) < required:
        raise ValueError(
            f"Need {required} tokens, "
            f"but only got {len(tokens)}"
        )

    input_ids_all = tokens[
        :required
    ].reshape(
        NUM_SAMPLES,
        SEQ_LEN
    ).to(DEVICE)

    # --------------------------------------------------------
    # budget
    # --------------------------------------------------------

    num_layers = len(
        model.model.layers
    )

    oracle_budgets, high_layers, low_layers = (
        build_oracle_budget(
            num_layers
        )
    )

    uniform_budgets = {
        layer: UNIFORM_BUDGET
        for layer in range(num_layers)
    }

    print("\n")
    print("=" * 70)
    print("Oracle budget assignment")
    print("=" * 70)

    print(
        "High sensitivity layers:",
        sorted(high_layers)
    )

    print(
        "Low sensitivity layers :",
        sorted(low_layers)
    )

    print(
        "\nLayer  SensitivityCos  OracleBudget"
    )

    for layer in range(num_layers):

        print(
            f"{layer:5d} "
            f"{SENSITIVITY_COSINE[layer]:15.4f} "
            f"{oracle_budgets[layer]:12d}"
        )

    avg_oracle = (
        sum(oracle_budgets.values())
        / num_layers
    )

    avg_uniform = (
        sum(uniform_budgets.values())
        / num_layers
    )

    print(
        f"\nAverage Uniform budget: "
        f"{avg_uniform:.2f}"
    )

    print(
        f"Average Oracle budget : "
        f"{avg_oracle:.2f}"
    )

    # --------------------------------------------------------
    # Uniform
    # --------------------------------------------------------

    uniform_results = run_strategy(
        model,
        input_ids_all,
        uniform_budgets,
        "UNIFORM"
    )

    # --------------------------------------------------------
    # Oracle
    # --------------------------------------------------------

    oracle_results = run_strategy(
        model,
        input_ids_all,
        oracle_budgets,
        "ORACLE"
    )

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    uniform_ok = [
        r for r in uniform_results
        if r["status"] == "ok"
    ]

    oracle_ok = [
        r for r in oracle_results
        if r["status"] == "ok"
    ]

    uniform_cos = np.mean([
        r["cosine"]
        for r in uniform_ok
    ])

    oracle_cos = np.mean([
        r["cosine"]
        for r in oracle_ok
    ])

    uniform_rel_l2 = np.mean([
        r["relative_l2"]
        for r in uniform_ok
    ])

    oracle_rel_l2 = np.mean([
        r["relative_l2"]
        for r in oracle_ok
    ])

    print("\n")
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        f"Uniform mean cosine : "
        f"{uniform_cos:.6f}"
    )

    print(
        f"Oracle mean cosine  : "
        f"{oracle_cos:.6f}"
    )

    print(
        f"Uniform mean Rel-L2 : "
        f"{uniform_rel_l2:.6f}"
    )

    print(
        f"Oracle mean Rel-L2  : "
        f"{oracle_rel_l2:.6f}"
    )

    print(
        f"\nCosine improvement "
        f"(Oracle - Uniform): "
        f"{oracle_cos - uniform_cos:.6f}"
    )

    print(
        f"Rel-L2 reduction "
        f"(Uniform - Oracle): "
        f"{uniform_rel_l2 - oracle_rel_l2:.6f}"
    )

    # --------------------------------------------------------
    # save
    # --------------------------------------------------------

    output = {
        "model": MODEL_PATH,
        "dataset": DATA_PATH,
        "num_samples": NUM_SAMPLES,
        "seq_len": SEQ_LEN,
        "uniform_budget": UNIFORM_BUDGET,
        "oracle_high_budget": HIGH_BUDGET,
        "oracle_mid_budget": MID_BUDGET,
        "oracle_low_budget": LOW_BUDGET,
        "average_uniform_budget": avg_uniform,
        "average_oracle_budget": avg_oracle,
        "high_sensitivity_layers": sorted(
            high_layers
        ),
        "low_sensitivity_layers": sorted(
            low_layers
        ),
        "sensitivity_cosine": SENSITIVITY_COSINE,
        "uniform_results": uniform_results,
        "oracle_results": oracle_results,
        "summary": {
            "uniform_mean_cosine": float(
                uniform_cos
            ),
            "oracle_mean_cosine": float(
                oracle_cos
            ),
            "uniform_mean_relative_l2": float(
                uniform_rel_l2
            ),
            "oracle_mean_relative_l2": float(
                oracle_rel_l2
            ),
            "cosine_improvement": float(
                oracle_cos - uniform_cos
            ),
            "relative_l2_reduction": float(
                uniform_rel_l2
                - oracle_rel_l2
            )
        }
    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nSaved to: {OUTPUT_JSON}"
    )


if __name__ == "__main__":
    main()
'''

path = Path("/mnt/data/oracle_budget_allocation.py")
path.write_text(code, encoding="utf-8")

# 简单语法检查
compile(code, str(path), "exec")

print(f"已生成并通过语法检查：{path}")
'''