import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置
# ============================================================
MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"
TEST_PATH = "../dataset/wikitext-02/test.jsonl"

N_TEST = 100
MAX_LEN = 128
DEVICE = "cuda"
SEED = 42

torch.manual_seed(SEED)

# ============================================================
# 加载
# ============================================================
print("加载模型...")
tok = AutoTokenizer.from_pretrained(MODEL_A)
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()

L_A = model_A.config.num_hidden_layers
L_B = model_B.config.num_hidden_layers
print(f"Layers: A={L_A}, B={L_B}")

df = pd.read_json(TEST_PATH, lines=True)
test_ids = []
for t in df["text"].iloc[:N_TEST].tolist():
    ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if len(ids) >= 32:
        test_ids.append(ids[:MAX_LEN].unsqueeze(0).to(DEVICE))
print(f"Test samples: {len(test_ids)}")

# ============================================================
# 提取某一层的 K、V（Sharer 侧）
# ============================================================
@torch.no_grad()
def extract_kv(model, layer_idx, ids):
    """返回 k, v，形状 [1, T, d]"""
    attn = model.model.layers[layer_idx].self_attn
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)

    out = model.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    normed = model.model.layers[layer_idx].input_layernorm(out.hidden_states[layer_idx])
    k = attn.k_proj(normed)   # [1, T, n_kv*head_dim]
    v = attn.v_proj(normed)
    return k, v

# ============================================================
# 多层替换
# ============================================================
def forward_with_replacement(model, ids, replacements):
    """
    replacements: dict {layer: {"K": tensor, "V": tensor}}
    每个 tensor 形状 [1, T, d]
    返回 logits
    """
    hooks = []
    for layer, repl in replacements.items():
        if "K" in repl and repl["K"] is not None:
            kt = repl["K"].to(DEVICE).to(torch.float16)
            def make_k_hook(kt):
                def hook(module, inp, out):
                    if out.shape != kt.shape:
                        return out
                    return kt
                return hook
            hooks.append(model.model.layers[layer].self_attn.k_proj.register_forward_hook(make_k_hook(kt)))
        if "V" in repl and repl["V"] is not None:
            vt = repl["V"].to(DEVICE).to(torch.float16)
            def make_v_hook(vt):
                def hook(module, inp, out):
                    if out.shape != vt.shape:
                        return out
                    return vt
                return hook
            hooks.append(model.model.layers[layer].self_attn.v_proj.register_forward_hook(make_v_hook(vt)))

    try:
        with torch.no_grad():
            logits = model(input_ids=ids).logits
    finally:
        for h in hooks:
            h.remove()
    return logits

# ============================================================
# 指标
# ============================================================
def compute_metrics(logits_base, logits_repl):
    """
    返回 (CE_repl, KL, top1_agree)
    """
    # shift
    lb = logits_base[:, :-1, :].float()
    lr = logits_repl[:, :-1, :].float()

    # Top-1 一致率
    top1_b = lb.argmax(dim=-1)
    top1_r = lr.argmax(dim=-1)
    agree = (top1_b == top1_r).float().mean().item()

    # KL(base || repl)
    logp_base = F.log_softmax(lb, dim=-1)
    p_base = logp_base.exp()
    logp_repl = F.log_softmax(lr, dim=-1)
    kl = F.kl_div(logp_repl, p_base, reduction='batchmean').item()

    return kl, agree

# ============================================================
# 主循环
# ============================================================
print(f"\n开始多层替换实验（{L_B} 层）...")

results = {
    "baseline": {"ce": [], "kl": [], "agree": []},
    "zero_V":   {"ce": [], "kl": [], "agree": []},
    "rand_V":   {"ce": [], "kl": [], "agree": []},
    "sharer_V": {"ce": [], "kl": [], "agree": []},
    "sharer_KV":{"ce": [], "kl": [], "agree": []},
}

for i, ids in enumerate(test_ids):
    # ---------- baseline ----------
    with torch.no_grad():
        logits_base = model_B(input_ids=ids).logits

    # CE of baseline
    shift_logits = logits_base[:, :-1, :].float()
    shift_labels = ids[:, 1:]
    ce_base = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1)
    ).item()
    results["baseline"]["ce"].append(ce_base)
    results["baseline"]["kl"].append(0.0)
    results["baseline"]["agree"].append(1.0)

    # ---------- 提取 Sharer 所有层 K、V ----------
    sharer_kv = {}
    for l in range(min(L_A, L_B)):
        k_A, v_A = extract_kv(model_A, l, ids)
        sharer_kv[l] = {"K": k_A, "V": v_A}

    # ---------- zero V ----------
    repl = {l: {"V": torch.zeros_like(sharer_kv[l]["V"])} for l in sharer_kv}
    logits = forward_with_replacement(model_B, ids, repl)
    kl, agree = compute_metrics(logits_base, logits)
    ce = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
        ids[:, 1:].reshape(-1)
    ).item()
    results["zero_V"]["ce"].append(ce)
    results["zero_V"]["kl"].append(kl)
    results["zero_V"]["agree"].append(agree)

    # ---------- random V ----------
    repl = {l: {"V": torch.randn_like(sharer_kv[l]["V"])} for l in sharer_kv}
    logits = forward_with_replacement(model_B, ids, repl)
    kl, agree = compute_metrics(logits_base, logits)
    ce = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
        ids[:, 1:].reshape(-1)
    ).item()
    results["rand_V"]["ce"].append(ce)
    results["rand_V"]["kl"].append(kl)
    results["rand_V"]["agree"].append(agree)

    # ---------- Sharer V ----------
    repl = {l: {"V": sharer_kv[l]["V"]} for l in sharer_kv}
    logits = forward_with_replacement(model_B, ids, repl)
    kl, agree = compute_metrics(logits_base, logits)
    ce = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
        ids[:, 1:].reshape(-1)
    ).item()
    results["sharer_V"]["ce"].append(ce)
    results["sharer_V"]["kl"].append(kl)
    results["sharer_V"]["agree"].append(agree)

    # ---------- Sharer K + V ----------
    repl = {l: {"K": sharer_kv[l]["K"], "V": sharer_kv[l]["V"]} for l in sharer_kv}
    logits = forward_with_replacement(model_B, ids, repl)
    kl, agree = compute_metrics(logits_base, logits)
    ce = F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
        ids[:, 1:].reshape(-1)
    ).item()
    results["sharer_KV"]["ce"].append(ce)
    results["sharer_KV"]["kl"].append(kl)
    results["sharer_KV"]["agree"].append(agree)

    if (i + 1) % 20 == 0:
        print(f"  已处理 {i+1}/{len(test_ids)}")

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*80}")
print(f"多层替换结果（{L_B} 层同时替换，{len(test_ids)} 个样本）")
print(f"{'='*80}")
print(f"{'Method':>12}  {'CE':>8}  {'ΔCE':>8}  {'KL':>8}  {'Top1-AGREE':>12}")
print("-" * 80)

base_ce = np.mean(results["baseline"]["ce"])
for m in ["baseline", "zero_V", "rand_V", "sharer_V", "sharer_KV"]:
    ce = np.mean(results[m]["ce"])
    kl = np.mean(results[m]["kl"])
    agree = np.mean(results[m]["agree"])
    print(f"{m:>12}  {ce:>8.4f}  {ce - base_ce:>+8.4f}  {kl:>8.4f}  {agree:>12.4f}")