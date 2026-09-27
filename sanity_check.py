import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置
# ============================================================
MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"
DATA_PATH = "../dataset/wikitext-02/train.jsonl"

TRAIN_ARTICLES = 500
EVAL_ARTICLES = 200
MAX_TOKENS_PER_ART = 128
LAYERS_TO_TRAIN = [0, 2, 5, 7, 9, 11, 13, 15, 17, 20, 22, 25, 27]

DEVICE = "cuda"
SEED = 42
RIDGE_BATCH = 50000
RIDGE_LAMBDA = 100.0

E2E_LR = 1e-4
E2E_EPOCHS = 30
E2E_BATCH = 1
E2E_PATIENCE = 5
E2E_VAL_RATIO = 0.15

MLP_HIDDEN = 512

torch.manual_seed(SEED)


# ============================================================
# 数据
# ============================================================
df = pd.read_json(DATA_PATH, lines=True)
all_texts = df["text"].tolist()

train_texts = all_texts[:TRAIN_ARTICLES]
eval_texts = all_texts[TRAIN_ARTICLES:TRAIN_ARTICLES + EVAL_ARTICLES]

tok = AutoTokenizer.from_pretrained(MODEL_A)


def build_padded_with_mask(texts, max_len=MAX_TOKENS_PER_ART):
    ids_list = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        ids = ids[:max_len]
        if len(ids) >= 16:
            ids_list.append(ids)
    L = max(len(x) for x in ids_list)
    padded = torch.zeros(len(ids_list), L, dtype=torch.long)
    mask = torch.zeros(len(ids_list), L, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, :len(ids)] = ids
        mask[i, :len(ids)] = 1
    return padded, mask


train_padded, train_mask = build_padded_with_mask(train_texts)
eval_padded, eval_mask = build_padded_with_mask(eval_texts)

B_train, T_train = train_padded.shape
B_eval, T_eval = eval_padded.shape

print("=" * 70)
print("Sanity Check")
print("=" * 70)
print(f"Train: {B_train} 篇 × {T_train} token")
print(f"Eval : {B_eval} 篇 × {T_eval} token")


# ============================================================
# 检查 0：数据划分
# ============================================================
print("\n[Check 0] 数据划分...")

# train 和 eval 是否重叠
train_texts_set = set(train_texts)
eval_texts_set = set(eval_texts)
overlap = train_texts_set & eval_texts_set
print(f"  Train ∩ Eval 文本重叠: {len(overlap)} (应为 0)")

# E2E train/val 划分
n_eval = len(eval_padded)
perm = torch.randperm(n_eval)
n_val = int(n_eval * E2E_VAL_RATIO)
val_idx = perm[:n_val]
tr_idx = perm[n_val:]

tr_set = set(tr_idx.tolist())
val_set = set(val_idx.tolist())
overlap2 = tr_set & val_set
print(f"  E2E Train ∩ Val 索引重叠: {len(overlap2)} (应为 0)")

# 检查 val_idx 和 tr_idx 是否覆盖全部
all_idx = set(range(n_eval))
print(f"  E2E Train + Val = 全集: {tr_set | val_set == all_idx}")


# ============================================================
# 工具
# ============================================================
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x, cos, sin):
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B_, H, T_, D = x.shape
    return x[:, :, None, :, :].expand(B_, H, n_rep, T_, D).reshape(B_, H * n_rep, T_, D)


def r2_score(pred, target):
    pred = pred.float()
    target = target.float()
    return (1 - ((pred - target)**2).sum() / (target**2).sum()).item()


def fit_ridge(X, Y, lam=RIDGE_LAMBDA, batch_size=RIDGE_BATCH):
    D_X = X.shape[-1]
    D_Y = Y.shape[-1]
    XtX = torch.zeros(D_X, D_X, dtype=torch.float32)
    XtY = torch.zeros(D_X, D_Y, dtype=torch.float32)
    N_ = X.shape[0]
    for s in range(0, N_, batch_size):
        e = min(s + batch_size, N_)
        x = X[s:e].float(); y = Y[s:e].float()
        XtX += x.T @ x
        XtY += x.T @ y
    return torch.linalg.solve(XtX + lam * torch.eye(D_X), XtY)


class VMapper(torch.nn.Module):
    def __init__(self, d_in, d_out, hidden):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, d_out),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


@torch.no_grad()
def extract_kv_3d(model, layer_idx, padded, mask):
    attn = model.model.layers[layer_idx].self_attn
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)

    all_k, all_v = [], []
    batch_size = 64

    for start in range(0, len(padded), batch_size):
        batch = padded[start:start+batch_size].to(DEVICE)
        m = mask[start:start+batch_size].to(DEVICE)
        B_b, T_b = batch.shape

        out = model.model(input_ids=batch, attention_mask=m,
                          output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[layer_idx]
        normed = model.model.layers[layer_idx].input_layernorm(hidden)

        k = attn.k_proj(normed).view(B_b, T_b, n_kv, head_dim)
        k = attn.k_norm(k).transpose(1, 2)
        k = k.transpose(1, 2).reshape(B_b, T_b, -1)
        all_k.append(k.to(torch.float16).cpu())

        v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
        v = v.transpose(1, 2).reshape(B_b, T_b, -1)
        all_v.append(v.to(torch.float16).cpu())

        del out, hidden, normed
        torch.cuda.empty_cache()

    K = torch.cat(all_k, 0)
    V = torch.cat(all_v, 0)
    return K, V


def forward_e2e(model, input_ids, attention_mask,
                K_A_all, V_A_all, mappers,
                W_K_list, W_V_list,
                layers_to_train,
                n_q, n_kv, head_dim, n_layers):
    B_b, T_b = input_ids.shape
    device = input_ids.device
    hidden = model.model.embed_tokens(input_ids)

    position_ids = torch.arange(T_b, device=device).unsqueeze(0).expand(B_b, -1)
    cos, sin = model.model.rotary_emb(hidden, position_ids)

    causal = torch.triu(
        torch.full((T_b, T_b), torch.finfo(hidden.dtype).min,
                   device=device, dtype=hidden.dtype),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    pad_mask = (1.0 - attention_mask[:, None, None, :].to(hidden.dtype)) \
               * torch.finfo(hidden.dtype).min
    combined_mask = causal + pad_mask

    n_rep = n_q // n_kv
    scale = 1.0 / math.sqrt(head_dim)

    for l, layer in enumerate(model.model.layers):
        K_a = K_A_all[l].to(device).float()
        W_K = W_K_list[l].to(device)
        K_map = (K_a @ W_K).to(hidden.dtype)

        V_a = V_A_all[l].to(device).float()

        if l in layers_to_train:
            V_map = mappers[l](V_a.float()).to(hidden.dtype)
        else:
            W_V = W_V_list[l].to(device)
            V_map = (V_a @ W_V).to(hidden.dtype)

        K = K_map.reshape(B_b, T_b, n_kv, head_dim).transpose(1, 2)
        V = V_map.reshape(B_b, T_b, n_kv, head_dim).transpose(1, 2)

        K = apply_rotary(K, cos, sin)

        residual = hidden
        x = layer.input_layernorm(hidden)
        attn = layer.self_attn
        q = attn.q_proj(x).view(B_b, T_b, n_q, head_dim)
        q = attn.q_norm(q).transpose(1, 2)
        q = apply_rotary(q, cos, sin)

        K_exp = repeat_kv(K, n_rep)
        V_exp = repeat_kv(V, n_rep)

        scores = torch.matmul(q.float(), K_exp.float().transpose(-1, -2)) * scale
        scores = scores + combined_mask
        probs = torch.softmax(scores, dim=-1).to(q.dtype)
        attn_out = torch.matmul(probs, V_exp)

        attn_out = attn_out.transpose(1, 2).reshape(B_b, T_b, -1)
        attn_out = attn.o_proj(attn_out)
        hidden = residual + attn_out

        residual = hidden
        x = layer.post_attention_layernorm(hidden)
        hidden = residual + layer.mlp(x)

    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)
    return logits


# ============================================================
# 加载模型
# ============================================================
print("\n加载模型...")
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()

for p in model_B.parameters():
    p.requires_grad = False

model_A.model.rotary_emb.inv_freq = model_A.model.rotary_emb.inv_freq.to(DEVICE)
model_B.model.rotary_emb.inv_freq = model_B.model.rotary_emb.inv_freq.to(DEVICE)

n_layers = model_A.config.num_hidden_layers
n_q = model_B.config.num_attention_heads
n_kv = model_B.config.num_key_value_heads
head_dim = getattr(model_B.config, "head_dim",
                   model_B.config.hidden_size // n_q)
D = n_kv * head_dim


# ============================================================
# 训练 W_K, W_V
# ============================================================
print("\n训练 W_K, W_V...")
W_K_list = [None] * n_layers
W_V_list = [None] * n_layers

for l in range(n_layers):
    K_A, V_A = extract_kv_3d(model_A, l, train_padded, train_mask)
    K_B, V_B = extract_kv_3d(model_B, l, train_padded, train_mask)

    W_K_list[l] = fit_ridge(K_A.reshape(-1, D), K_B.reshape(-1, D))
    W_V_list[l] = fit_ridge(V_A.reshape(-1, D), V_B.reshape(-1, D))

print("  Done.")

del model_A
gc.collect()
torch.cuda.empty_cache()


# ============================================================
# 提取 eval K/V
# ============================================================
print("\n提取 eval K/V...")
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_A.model.rotary_emb.inv_freq = model_A.model.rotary_emb.inv_freq.to(DEVICE)

K_A_eval = []
V_A_eval = []
for l in range(n_layers):
    K, V = extract_kv_3d(model_A, l, eval_padded, eval_mask)
    K_A_eval.append(K)
    V_A_eval.append(V)

del model_A
gc.collect()
torch.cuda.empty_cache()
print("  Done.")


# ============================================================
# Check 1：训练 E2E mapper 并检查它是否在改变 V
# ============================================================
print("\n" + "=" * 70)
print("[Check 1] E2E mapper 是否在改变 V")
print("=" * 70)

mappers = {l: VMapper(D, D, MLP_HIDDEN).to(DEVICE) for l in LAYERS_TO_TRAIN}
all_params = []
for m in mappers.values():
    all_params += list(m.parameters())

opt = torch.optim.Adam(all_params, lr=E2E_LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E2E_EPOCHS)

best_val = float("inf")
best_state = None
wait = 0

for epoch in range(E2E_EPOCHS):
    for m in mappers.values():
        m.train()
    tr_loss = 0.0; n_steps = 0
    perm_e = torch.randperm(len(tr_idx))

    for s in range(0, len(tr_idx), E2E_BATCH):
        e = min(s + E2E_BATCH, len(tr_idx))
        batch_articles = tr_idx[perm_e[s:e]]
        ids = eval_padded[batch_articles].to(DEVICE)
        mask = eval_mask[batch_articles].to(DEVICE)

        K_A_batch = [K_A_eval[l][batch_articles].to(DEVICE) for l in range(n_layers)]
        V_A_batch = [V_A_eval[l][batch_articles].to(DEVICE) for l in range(n_layers)]

        logits = forward_e2e(model_B, ids, mask, K_A_batch, V_A_batch, mappers,
                             W_K_list, W_V_list, LAYERS_TO_TRAIN,
                             n_q, n_kv, head_dim, n_layers)

        sl = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        slb = ids[:, 1:].reshape(-1)
        sm = mask[:, 1:].reshape(-1).float()
        loss_per = F.cross_entropy(sl, slb, reduction="none")
        loss = (loss_per * sm).sum() / sm.sum().clamp_min(1)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()

        tr_loss += loss.item(); n_steps += 1
        del logits, loss_per, loss, K_A_batch, V_A_batch
        torch.cuda.empty_cache()

    tr_loss /= n_steps
    scheduler.step()

    for m in mappers.values():
        m.eval()
    val_loss = 0.0; n_val_steps = 0
    with torch.no_grad():
        for s in range(0, len(val_idx), E2E_BATCH):
            e = min(s + E2E_BATCH, len(val_idx))
            batch_articles = val_idx[s:e]
            ids = eval_padded[batch_articles].to(DEVICE)
            mask = eval_mask[batch_articles].to(DEVICE)

            K_A_batch = [K_A_eval[l][batch_articles].to(DEVICE) for l in range(n_layers)]
            V_A_batch = [V_A_eval[l][batch_articles].to(DEVICE) for l in range(n_layers)]

            logits = forward_e2e(model_B, ids, mask, K_A_batch, V_A_batch, mappers,
                                 W_K_list, W_V_list, LAYERS_TO_TRAIN,
                                 n_q, n_kv, head_dim, n_layers)

            sl = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            slb = ids[:, 1:].reshape(-1)
            sm = mask[:, 1:].reshape(-1).float()
            loss_per = F.cross_entropy(sl, slb, reduction="none")
            loss = (loss_per * sm).sum() / sm.sum().clamp_min(1)

            val_loss += loss.item(); n_val_steps += 1
            del logits, loss_per, loss, K_A_batch, V_A_batch
            torch.cuda.empty_cache()

    val_loss /= n_val_steps

    if val_loss < best_val:
        best_val = val_loss
        best_state = {l: {k: v.clone() for k, v in m.state_dict().items()}
                      for l, m in mappers.items()}
        wait = 0
    else:
        wait += 1
        if wait >= E2E_PATIENCE:
            break

if best_state is not None:
    for l, sd in best_state.items():
        mappers[l].load_state_dict(sd)

print(f"  Best val loss: {best_val:.4f}  (PPL = {math.exp(best_val):.4f})")

# 检查每个 mapper 改变了 V 多少
print(f"\n  每个 mapper 对 V 的改变：")
print(f"  {'Layer':>6}  {'||ΔV||/||V_in||':>18}  {'R²(ΔV vs V_B-V_A)':>20}")

for l in LAYERS_TO_TRAIN:
    v_in = V_A_eval[l][:100].float().to(DEVICE)
    v_out = mappers[l](v_in).cpu().float()
    v_in_cpu = v_in.cpu().float()

    rel_change = (v_out - v_in_cpu).norm() / v_in_cpu.norm()

    # 和「真实需要的改变量」对比：V_B - V_A
    # 但我们没有 V_B（1.7B 自己的 V），只能看映射后的 V 和 V_A 的差异
    print(f"  {l:>6}  {rel_change.item():>18.4f}")


# ============================================================
# Check 2：多种子稳定性
# ============================================================
print("\n" + "=" * 70)
print("[Check 2] 多种子稳定性（快速版：只训 3 个种子）")
print("=" * 70)

# 为了快速，只训 1 层 (Layer 14)，看 Ratio 的种子方差
SMALL_LAYERS = [14]
seed_results = []

for trial_seed in [0, 1, 2]:
    torch.manual_seed(trial_seed)

    mappers_t = {l: VMapper(D, D, MLP_HIDDEN).to(DEVICE) for l in SMALL_LAYERS}
    params_t = []
    for m in mappers_t.values():
        params_t += list(m.parameters())

    opt_t = torch.optim.Adam(params_t, lr=E2E_LR)

    # 短训 10 epoch
    for epoch in range(10):
        for m in mappers_t.values():
            m.train()
        perm_e = torch.randperm(len(tr_idx))
        for s in range(0, len(tr_idx), E2E_BATCH):
            e = min(s + E2E_BATCH, len(tr_idx))
            ba = tr_idx[perm_e[s:e]]
            ids = eval_padded[ba].to(DEVICE)
            mask = eval_mask[ba].to(DEVICE)
            K_A_b = [K_A_eval[l][ba].to(DEVICE) for l in range(n_layers)]
            V_A_b = [V_A_eval[l][ba].to(DEVICE) for l in range(n_layers)]

            logits = forward_e2e(model_B, ids, mask, K_A_b, V_A_b, mappers_t,
                                 W_K_list, W_V_list, SMALL_LAYERS,
                                 n_q, n_kv, head_dim, n_layers)
            sl = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            slb = ids[:, 1:].reshape(-1)
            sm = mask[:, 1:].reshape(-1).float()
            lp = F.cross_entropy(sl, slb, reduction="none")
            loss = (lp * sm).sum() / sm.sum().clamp_min(1)

            opt_t.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_t, 1.0)
            opt_t.step()
            del logits, lp, loss, K_A_b, V_A_b
            torch.cuda.empty_cache()

    # 评估
    for m in mappers_t.values():
        m.eval()
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(val_idx), E2E_BATCH):
            e = min(s + E2E_BATCH, len(val_idx))
            ba = val_idx[s:e]
            ids = eval_padded[ba].to(DEVICE)
            mask = eval_mask[ba].to(DEVICE)
            K_A_b = [K_A_eval[l][ba].to(DEVICE) for l in range(n_layers)]
            V_A_b = [V_A_eval[l][ba].to(DEVICE) for l in range(n_layers)]

            logits = forward_e2e(model_B, ids, mask, K_A_b, V_A_b, mappers_t,
                                 W_K_list, W_V_list, SMALL_LAYERS,
                                 n_q, n_kv, head_dim, n_layers)
            sl = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            slb = ids[:, 1:].reshape(-1)
            sm = mask[:, 1:].reshape(-1).float()
            lp = F.cross_entropy(sl, slb, reduction="none")
            total_loss += (lp * sm).sum().item()
            total_tok += sm.sum().item()
            del logits, lp, K_A_b, V_A_b
            torch.cuda.empty_cache()

    val_ppl = math.exp(total_loss / total_tok)
    seed_results.append(val_ppl)
    print(f"  Seed {trial_seed}: val_ppl = {val_ppl:.4f}")

print(f"\n  多种子结果: {[f'{x:.4f}' for x in seed_results]}")
print(f"  均值: {np.mean(seed_results):.4f}")
print(f"  标准差: {np.std(seed_results):.4f}")
print(f"  变异系数: {np.std(seed_results) / np.mean(seed_results):.4f}")

if np.std(seed_results) / np.mean(seed_results) < 0.05:
    print("  ✅ 多种子稳定（CV < 5%）")
else:
    print("  ⚠️ 多种子不稳定（CV > 5%）")


# ============================================================
# Check 3：测试集 PPL 复现
# ============================================================
print("\n" + "=" * 70)
print("[Check 3] 测试集 PPL 复现")
print("=" * 70)

# 用完整 13 层 mapper 在 eval 全集上算 PPL
@torch.no_grad()
def compute_ppl(mappers, layers_to_train, use_e2e=True, batch_size=1):
    total_loss, total_tokens = 0.0, 0
    for s in range(0, len(eval_padded), batch_size):
        e = min(s + batch_size, len(eval_padded))
        ba = torch.arange(s, e)
        ids = eval_padded[ba].to(DEVICE)
        mask = eval_mask[ba].to(DEVICE)
        K_A_b = [K_A_eval[l][ba].to(DEVICE) for l in range(n_layers)]
        V_A_b = [V_A_eval[l][ba].to(DEVICE) for l in range(n_layers)]

        if use_e2e:
            logits = forward_e2e(model_B, ids, mask, K_A_b, V_A_b, mappers,
                                 W_K_list, W_V_list, layers_to_train,
                                 n_q, n_kv, head_dim, n_layers)
        else:
            logits = forward_e2e(model_B, ids, mask, K_A_b, V_A_b, {},
                                 W_K_list, W_V_list, [],
                                 n_q, n_kv, head_dim, n_layers)

        sl = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        slb = ids[:, 1:].reshape(-1)
        sm = mask[:, 1:].reshape(-1).float()
        lp = F.cross_entropy(sl, slb, reduction="none")
        total_loss += (lp * sm).sum().item()
        total_tokens += sm.sum().item()
        del logits, lp, K_A_b, V_A_b
        torch.cuda.empty_cache()
    return math.exp(total_loss / total_tokens)


# Native
@torch.no_grad()
def ppl_native(batch_size=2):
    total_loss, total_tokens = 0.0, 0
    for s in range(0, len(eval_padded), batch_size):
        e = min(s + batch_size, len(eval_padded))
        ids = eval_padded[s:e].to(DEVICE)
        mask = eval_mask[s:e].to(DEVICE)
        out = model_B(input_ids=ids, attention_mask=mask)
        logits = out.logits
        sl = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        slb = ids[:, 1:].reshape(-1)
        sm = mask[:, 1:].reshape(-1).float()
        lp = F.cross_entropy(sl, slb, reduction="none")
        total_loss += (lp * sm).sum().item()
        total_tokens += sm.sum().item()
        del out, logits, lp
        torch.cuda.empty_cache()
    return math.exp(total_loss / total_tokens)


ppl_nat = ppl_native()
print(f"  PPL_native = {ppl_nat:.4f}")

ppl_lin = compute_ppl({}, [], use_e2e=False)
print(f"  PPL_linear = {ppl_lin:.4f}")

ppl_e2e = compute_ppl(mappers, LAYERS_TO_TRAIN, use_e2e=True)
print(f"  PPL_e2e = {ppl_e2e:.4f}")

print(f"\n  E2E ratio: {ppl_e2e / ppl_nat:.4f}")


# ============================================================
# 汇总
# ============================================================
print()
print("=" * 70)
print("Sanity Check 汇总")
print("=" * 70)
print(f"  Check 0 - 数据划分: {'✅' if len(overlap) == 0 else '❌'}")
print(f"  Check 1 - Mapper 改变 V: 见上表")
print(f"  Check 2 - 多种子 CV: {np.std(seed_results) / np.mean(seed_results):.4f}")
print(f"  Check 3 - E2E ratio: {ppl_e2e / ppl_nat:.4f}")
print("=" * 70)