import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置
# ============================================================
MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"
TEST_PATH = "../dataset/wikitext-02/test.jsonl"
LAYER = 14
DEVICE = "cuda"
MAX_LEN = 128
SEED = 42

torch.manual_seed(SEED)

# ============================================================
# 加载
# ============================================================
print("加载...")
tok = AutoTokenizer.from_pretrained(MODEL_A)
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()

df = pd.read_json(TEST_PATH, lines=True)
text = df["text"].iloc[0]
ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0][:MAX_LEN]
ids = ids.unsqueeze(0).to(DEVICE)
print(f"input shape: {ids.shape}")

# ============================================================
# 提取 Sharer 第 L 层的 V（用 hook 顺便确认形状）
# ============================================================
v_shape_holder = {}

def capture_hook(module, inp, out):
    v_shape_holder["shape"] = tuple(out.shape)
    v_shape_holder["dtype"] = out.dtype

h = model_A.model.layers[LAYER].self_attn.v_proj.register_forward_hook(capture_hook)
with torch.no_grad():
    _ = model_A(ids)
h.remove()

print(f"v_proj output shape: {v_shape_holder['shape']}, dtype: {v_shape_holder['dtype']}")

# ============================================================
# 在 Receiver 上，验证 hook 是否替换到 V
# ============================================================
def compute_loss(model, input_ids, new_v=None, verbose=False):
    labels = input_ids.clone()

    hook_handle = None
    if new_v is not None:
        v_tensor = new_v.to(DEVICE).to(torch.float16)
        call_count = {"n": 0}

        def hook(module, inp, out):
            call_count["n"] += 1
            if verbose and call_count["n"] == 1:
                print(f"[hook] called! out={tuple(out.shape)}, new_v={tuple(v_tensor.shape)}")
                print(f"       out dtype={out.dtype}, new_v dtype={v_tensor.dtype}")
                print(f"       match: {out.shape == v_tensor.shape}")
            if out.shape != v_tensor.shape:
                return out   # 形状不匹配就不替换
            return v_tensor

        hook_handle = model.model.layers[LAYER].self_attn.v_proj.register_forward_hook(hook)

    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss.item()
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    if new_v is not None and verbose:
        print(f"[hook] total calls: {call_count['n']}")

    return loss


# ============================================================
# 提取一次 Sharer 的 V
# ============================================================
attn_A = model_A.model.layers[LAYER].self_attn
n_kv = model_A.config.num_key_value_heads
head_dim = getattr(model_A.config, "head_dim",
                   model_A.config.hidden_size // model_A.config.num_attention_heads)

with torch.no_grad():
    out_A = model_A.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hidden_A = out_A.hidden_states[LAYER]
    normed_A = model_A.model.layers[LAYER].input_layernorm(hidden_A)
    v_A = attn_A.v_proj(normed_A)                  # [1, T, n_kv*head_dim]
    print(f"Sharer V shape: {tuple(v_A.shape)}")

# ============================================================
# 四种情况对比
# ============================================================
print(f"\n{'='*60}")
print("Verification")
print(f"{'='*60}")

# 1. baseline
loss_base = compute_loss(model_B, ids)
print(f"baseline loss      : {loss_base:.4f}")

# 2. zero V（verbose，看 hook 是否被调用）
loss_zero = compute_loss(model_B, ids, torch.zeros_like(v_A), verbose=True)
print(f"zero V loss        : {loss_zero:.4f}")

# 3. Sharer V（直接用 Sharer 的 V）
loss_sharer = compute_loss(model_B, ids, v_A)
print(f"Sharer V loss      : {loss_sharer:.4f}")

# 4. 随机 V
loss_rand = compute_loss(model_B, ids, torch.randn_like(v_A))
print(f"random V loss      : {loss_rand:.4f}")

print(f"\nΔ vs baseline:")
print(f"  zero V   : {loss_zero - loss_base:+.4f}")
print(f"  Sharer V : {loss_sharer - loss_base:+.4f}")
print(f"  random V : {loss_rand - loss_base:+.4f}")