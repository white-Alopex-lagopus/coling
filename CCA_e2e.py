import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import torch.nn.functional as F
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置
# ============================================================
MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"
PROJ_PATH = "cca_proj.pt"
DEVICE = "cuda"
MAX_NEW_TOKENS = 50

# ============================================================
# 加载
# ============================================================
print("加载模型...")
tok = AutoTokenizer.from_pretrained(MODEL_A)
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()
proj = torch.load(PROJ_PATH, map_location="cpu")
L = len(proj)
print(f"Layers: {L}")

# ============================================================
# 重建 V
# ============================================================
@torch.no_grad()
def extract_v(model, layer_idx, ids):
    attn = model.model.layers[layer_idx].self_attn
    out = model.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    normed = model.model.layers[layer_idx].input_layernorm(out.hidden_states[layer_idx])
    return attn.v_proj(normed)

@torch.no_grad()
def reconstruct_v_all_layers(source_ids):
    out = {}
    for l in range(L):
        v_A = extract_v(model_A, l, source_ids)
        d = v_A.shape[-1]
        v_flat = v_A.reshape(-1, d).float().cpu()
        p = proj[l]
        A = (v_flat - p["mS"]) @ p["W_S"]
        v_hat = A @ p["M"] + p["mT"]
        out[l] = v_hat.reshape(1, -1, d).to(torch.float16)
    return out

# ============================================================
# 注入 V 到 cache（兼容不同版本）
# ============================================================
def inject_v_into_cache(cache, v_replacements, n_kv, head_dim):
    for layer, v_new in v_replacements.items():
        v_new = v_new.to(DEVICE).to(torch.float16)

        if hasattr(cache, "value_cache"):
            old_v = cache.value_cache[layer]
        elif hasattr(cache, "layers"):
            old_v = cache.layers[layer].values
        else:
            raise RuntimeError("Unknown DynamicCache format")

        B = old_v.shape[0]
        T = v_new.shape[1]
        d = v_new.shape[2]

        v_reshaped = v_new.reshape(B, T, n_kv, head_dim).transpose(1, 2).contiguous()

        if v_reshaped.shape != old_v.shape:
            print(f"[WARN] shape mismatch: {v_reshaped.shape} vs {old_v.shape}, skip")
            continue

        if hasattr(cache, "value_cache"):
            cache.value_cache[layer] = v_reshaped
        elif hasattr(cache, "layers"):
            cache.layers[layer].values = v_reshaped

    return cache

# ============================================================
# 手动 decode 循环
# ============================================================
@torch.no_grad()
def manual_generate(model, first_token_id, past_kv, max_new_tokens=50, eos_token_id=None):
    """
    first_token_id: [1, 1] 第一个要生成的 token 的输入
    past_kv: 已有的 KV cache
    返回生成的 token ids: [1, n]
    """
    generated = first_token_id.clone()
    cache = past_kv

    for _ in range(max_new_tokens):
        out = model(input_ids=first_token_id, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
        generated = torch.cat([generated, next_token], dim=-1)
        first_token_id = next_token
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    return generated

# ============================================================
# 场景
# ============================================================
context = """
The discovery of penicillin by Alexander Fleming in 1928 marked a turning point in modern medicine.
Fleming noticed that a mold called Penicillium notatum had contaminated one of his petri dishes and
was killing the surrounding bacteria. This accidental observation led to the development of the first
antibiotic, which would go on to save millions of lives worldwide.
""".strip()

question = "\n\nQuestion: Who discovered penicillin and in what year?\nAnswer:"
prompt = context + question
prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
print(f"Prompt length: {prompt_ids.shape[1]} tokens")

# ============================================================
# 1. Baseline: 大模型自己
# ============================================================
print("\n" + "="*60)
print("Baseline")
print("="*60)

torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    out_base = model_B.generate(
        prompt_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        return_dict_in_generate=True,
    )
torch.cuda.synchronize()
t_baseline = time.time() - t0

answer_base = tok.decode(out_base.sequences[0], skip_special_tokens=True)
print(f"Time: {t_baseline:.3f}s")
print(f"Output:\n{answer_base}")

# ============================================================
# 2. CCA: 大模型 prefill → 注入重建 V → 手动 decode
# ============================================================
print("\n" + "="*60)
print("CCA")
print("="*60)

torch.cuda.synchronize()
t0 = time.time()

# Step 1: 大模型 prefill
with torch.no_grad():
    out_B = model_B(prompt_ids, use_cache=True)
    cache_B = out_B.past_key_values
    last_logits = out_B.logits[:, -1, :]   # [1, vocab]

# Step 2: 小模型 prefill + 重建 V
torch.cuda.synchronize()
t_small_start = time.time()
with torch.no_grad():
    v_repl = reconstruct_v_all_layers(prompt_ids)
torch.cuda.synchronize()
t_small = time.time() - t_small_start
print(f"Small model prefill + reconstruct: {t_small:.3f}s")

# Step 3: 注入
n_kv = model_B.config.num_key_value_heads
head_dim = getattr(model_B.config, "head_dim",
                   model_B.config.hidden_size // model_B.config.num_attention_heads)

cache_B = inject_v_into_cache(cache_B, v_repl, n_kv, head_dim)
print("V injected.")

# Step 4: 从 prefill 的最后一个 logits 取第一个 token
first_token = last_logits.argmax(dim=-1, keepdim=True)   # [1, 1]

# Step 5: 手动 decode
with torch.no_grad():
    gen_cca = manual_generate(
        model_B,
        first_token,
        cache_B,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tok.eos_token_id,
    )
torch.cuda.synchronize()
t_cca = time.time() - t0

# 拼接 prompt + 生成的 token 来解码
full_ids_cca = torch.cat([prompt_ids, gen_cca], dim=-1)
answer_cca = tok.decode(full_ids_cca[0], skip_special_tokens=True)
print(f"Total time: {t_cca:.3f}s")
print(f"Output:\n{answer_cca}")

# ============================================================
# 对比
# ============================================================
print("\n" + "="*60)
print("对比")
print("="*60)
print(f"Baseline time: {t_baseline:.3f}s")
print(f"CCA time     : {t_cca:.3f}s")
print(f"加速比       : {t_baseline / t_cca:.2f}×")
print(f"\nBaseline answer:\n{answer_base}")
print(f"\nCCA answer:\n{answer_cca}")