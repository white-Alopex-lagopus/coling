import os
import json
import math
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ============================================================
# Config
# ============================================================

MODEL_PATH = "../model/Qwen3-1.7B-base"

NUM_SAMPLES = 50
SEQ_LEN = 512

# 每层 KV 从 512 token 压缩到 64 token
COMPRESSED_LEN = 64

COMPRESS_K = True
COMPRESS_V = True

OUTPUT_DIR = "./layer_sensitivity_results"


# ============================================================
# Utilities
# ============================================================

def cosine_similarity(x, y, eps=1e-8):
    x = x.float()
    y = y.float()

    x = x / (torch.norm(x, dim=-1, keepdim=True) + eps)
    y = y / (torch.norm(y, dim=-1, keepdim=True) + eps)

    return (x * y).sum(dim=-1)


def compress_tokens(x, target_len):
    """
    x: [B, H, T, D]
    例如 [1, 8, 512, 128] -> [1, 8, 64, 128]
    使用简单均匀平均池化。
    """
    B, H, T, D = x.shape

    if T % target_len != 0:
        raise ValueError(
            f"T={T} cannot be evenly compressed to {target_len}"
        )

    ratio = T // target_len

    x = x.reshape(B, H, target_len, ratio, D)
    x = x.mean(dim=3)

    return x


def repeat_kv(hidden_states, n_rep):
    """
    [B, KVH, T, D] -> [B, QH, T, D]
    """
    B, KVH, T, D = hidden_states.shape

    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :]
    hidden_states = hidden_states.expand(
        B, KVH, n_rep, T, D
    )

    return hidden_states.reshape(
        B, KVH * n_rep, T, D
    )


def causal_attention(q, k, v):
    """
    q: [B, QH, Tq, D]
    k: [B, QH, Tk, D]
    v: [B, QH, Tk, D]

    注意：
    这里只用于第一阶段的局部敏感性实验。
    """
    B, QH, Tq, D = q.shape
    Tk = k.shape[2]

    scale = 1.0 / math.sqrt(D)

    scores = torch.matmul(
        q,
        k.transpose(-2, -1)
    ) * scale

    # 完整 KV 时使用标准 causal mask。
    # 压缩 KV 后，当前实验暂时不把池化 token
    # 重新解释为原始位置，只做粗粒度敏感性测试。
    if Tq == Tk:
        mask = torch.triu(
            torch.ones(
                Tq,
                Tk,
                device=q.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        scores = scores.masked_fill(
            mask,
            torch.finfo(scores.dtype).min
        )

    attn = torch.softmax(scores, dim=-1)

    output = torch.matmul(attn, v)

    return output


# ============================================================
# Load model
# ============================================================

print("=" * 70)
print("Loading model")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()

device = next(model.parameters()).device

config = model.config

NUM_LAYERS = config.num_hidden_layers
NUM_Q_HEADS = config.num_attention_heads
NUM_KV_HEADS = config.num_key_value_heads
HEAD_DIM = config.head_dim

print(f"Model      : {MODEL_PATH}")
print(f"Layers     : {NUM_LAYERS}")
print(f"Q heads    : {NUM_Q_HEADS}")
print(f"KV heads   : {NUM_KV_HEADS}")
print(f"Head dim   : {HEAD_DIM}")
print(f"Samples    : {NUM_SAMPLES}")
print(f"Seq len    : {SEQ_LEN}")
print(f"Compressed : {COMPRESSED_LEN}")
print()


# ============================================================
# Load YOUR WikiText-2 dataset
# ============================================================

print("=" * 70)
print("Loading WikiText-2")
print("=" * 70)

df = pd.read_json(
    "../dataset/wikitext-02/kvtest.jsonl",
    lines=True
)

text = df["text"].str.cat(sep="\n")

# 一次性 tokenize 整个文本
tokens = tokenizer(
    text,
    return_tensors="pt"
).input_ids[0]

total_tokens = NUM_SAMPLES * SEQ_LEN

if tokens.numel() < total_tokens:
    raise ValueError(
        f"Not enough tokens: "
        f"{tokens.numel()} < {total_tokens}"
    )

# 只取前 50 * 512 个 token
tokens = tokens[:total_tokens]

# [50, 512]
input_ids_all = tokens.reshape(
    NUM_SAMPLES,
    SEQ_LEN
).to(device)

print("Data shape:", input_ids_all.shape)
print()


# ============================================================
# Main experiment
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

all_results = []


for sample_id in range(NUM_SAMPLES):

    print()
    print("=" * 70)
    print(f"Sample {sample_id + 1}/{NUM_SAMPLES}")
    print("=" * 70)

    # [1, 512]
    input_ids = input_ids_all[
        sample_id:sample_id + 1
    ]

    # --------------------------------------------------------
    # Get hidden states
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True
        )

    hidden_states_all = outputs.hidden_states

    # hidden_states_all[0] = embedding output
    # hidden_states_all[layer_id] = input to that layer

    # --------------------------------------------------------
    # Layer-by-layer
    # --------------------------------------------------------

    for layer_id in range(NUM_LAYERS):

        print(
            f"  Layer {layer_id + 1}/{NUM_LAYERS}",
            end=" ... ",
            flush=True
        )

        layer = model.model.layers[layer_id]

        hidden_states = hidden_states_all[layer_id]

        B, T, _ = hidden_states.shape

        # ----------------------------------------------------
        # Q / K / V
        # ----------------------------------------------------

        # q = layer.q_proj(hidden_states)
        # k = layer.k_proj(hidden_states)
        # v = layer.v_proj(hidden_states)
        
        q = layer.self_attn.q_proj(hidden_states)
        k = layer.self_attn.k_proj(hidden_states)
        v = layer.self_attn.v_proj(hidden_states)

        q = q.view(
            B,
            T,
            NUM_Q_HEADS,
            HEAD_DIM
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            NUM_KV_HEADS,
            HEAD_DIM
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            NUM_KV_HEADS,
            HEAD_DIM
        ).transpose(1, 2)

        # Qwen3 Q/K normalization
        if hasattr(layer, "q_norm"):
            q = layer.q_norm(q)

        if hasattr(layer, "k_norm"):
            k = layer.k_norm(k)

        # ----------------------------------------------------
        # Full KV attention
        # ----------------------------------------------------

        n_rep = NUM_Q_HEADS // NUM_KV_HEADS

        k_full = repeat_kv(k, n_rep)
        v_full = repeat_kv(v, n_rep)

        with torch.no_grad():

            full_output = causal_attention(
                q,
                k_full,
                v_full
            )

        # ----------------------------------------------------
        # Compress K/V
        # ----------------------------------------------------

        k_comp = k
        v_comp = v

        if COMPRESS_K:
            k_comp = compress_tokens(
                k_comp,
                COMPRESSED_LEN
            )

        if COMPRESS_V:
            v_comp = compress_tokens(
                v_comp,
                COMPRESSED_LEN
            )

        k_comp = repeat_kv(
            k_comp,
            n_rep
        )

        v_comp = repeat_kv(
            v_comp,
            n_rep
        )

        with torch.no_grad():

            compressed_output = causal_attention(
                q,
                k_comp,
                v_comp
            )

        # ----------------------------------------------------
        # Compare
        # ----------------------------------------------------

        # 当前 compressed attention 输出长度是 64，
        # 与原始 512 个 query 位置不同。
        #
        # 因此这里只比较前 COMPRESSED_LEN 个 query，
        # 保持两个 tensor 的 shape 一致。
        compare_len = min(
            full_output.shape[2],
            compressed_output.shape[2]
        )

        full_cmp = full_output[
            :, :, :compare_len, :
        ]

        comp_cmp = compressed_output[
            :, :, :compare_len, :
        ]

        cos = cosine_similarity(
            full_cmp,
            comp_cmp
        )

        relative_error = (
            torch.norm(
                full_cmp.float() - comp_cmp.float(),
                dim=-1
            )
            /
            (
                torch.norm(
                    full_cmp.float(),
                    dim=-1
                )
                + 1e-8
            )
        )

        result = {
            "sample_id": sample_id,
            "layer": layer_id,
            "seq_len": T,
            "compressed_len": COMPRESSED_LEN,
            "cosine_similarity": cos.mean().item(),
            "relative_error": relative_error.mean().item()
        }

        all_results.append(result)

        print(
            f"cos={result['cosine_similarity']:.4f}, "
            f"rel_err={result['relative_error']:.4f}"
        )


# ============================================================
# Aggregate
# ============================================================

print()
print("=" * 70)
print("Aggregating results")
print("=" * 70)

layer_results = []

for layer_id in range(NUM_LAYERS):

    results = [
        x for x in all_results
        if x["layer"] == layer_id
    ]

    if not results:
        continue

    cos_values = [
        x["cosine_similarity"]
        for x in results
    ]

    error_values = [
        x["relative_error"]
        for x in results
    ]

    layer_results.append({
        "layer": layer_id,
        "cosine_mean": float(np.mean(cos_values)),
        "cosine_std": float(np.std(cos_values)),
        "relative_error_mean": float(np.mean(error_values)),
        "relative_error_std": float(np.std(error_values)),
        "num_samples": len(results)
    })


# ============================================================
# Save
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "layer_sensitivity.json"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "config": {
                "model": MODEL_PATH,
                "num_samples": NUM_SAMPLES,
                "seq_len": SEQ_LEN,
                "compressed_len": COMPRESSED_LEN,
                "compress_k": COMPRESS_K,
                "compress_v": COMPRESS_V
            },
            "layers": layer_results
        },
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# Print summary
# ============================================================

print()
print("=" * 70)
print("Layer sensitivity")
print("=" * 70)

print(
    f"{'Layer':>8} "
    f"{'Cosine':>12} "
    f"{'RelError':>12}"
)

print("-" * 40)

for x in layer_results:

    print(
        f"{x['layer']:>8} "
        f"{x['cosine_mean']:>12.4f} "
        f"{x['relative_error_mean']:>12.4f}"
    )

print()
print(f"Saved to: {output_file}")
