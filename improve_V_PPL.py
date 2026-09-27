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

TRAIN_ARTICLES = 2000
EVAL_ARTICLES = 200
MAX_TOKENS_PER_ART = 256
LAYER = 14

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

print(f"Train: {B_train} 篇 × {T_train} token")
print(f"Eval : {B_eval} 篇 × {T_eval} token")


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


# ============================================================
# Ridge
# ============================================================
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


# ============================================================
# MLP
# ============================================================
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


# ============================================================
# 提取 K/V（返回 [B, T, D]）
# ============================================================
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
        # [B_b, T_b, n_kv, head_dim] -> [B_b, T_b, n_kv*head_dim]
        k = k.transpose(1, 2).reshape(B_b, T_b, -1)
        all_k.append(k.to(torch.float16).cpu())

        v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
        v = v.transpose(1, 2).reshape(B_b, T_b, -1)
        all_v.append(v.to(torch.float16).cpu())

        del out, hidden, normed
        torch.cuda.empty_cache()

    K = torch.cat(all_k, 0)     # [B, T, D]
    V = torch.cat(all_v, 0)
    return K, V


# ============================================================
# 可微 forward
# ============================================================
def forward_e2e(model, input_ids, attention_mask,
                K_A_all, V_A_all, mapper,
                W_K_list, layer_to_train,
                n_q, n_kv, head_dim, n_layers):
    """
    K_A_all[l]: [B, T, D]  (fp16 CPU)
    V_A_all[l]: [B, T, D]
    """
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
        # ---- K ----
        K_a = K_A_all[l].to(device).float()      # [B, T, D]
        W_K = W_K_list[l].to(device)
        K_map = (K_a @ W_K).to(hidden.dtype)     # [B, T, D]

        # ---- V ----
        V_a = V_A_all[l].to(device).float()      # [B, T, D]

        if l == layer_to_train:
            # 用 mapper（可微）
            # V_map = mapper(V_a.to(hidden.dtype))
            V_map = mapper(V_a.float()).to(hidden.dtype)
        else:
            # 用线性 W_V
            W_V = W_V_list_global[l].to(device)
            V_map = (V_a @ W_V).to(hidden.dtype)

        # reshape 到多头
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

print(f"Layers: {n_layers}, D={D}, train layer: {LAYER}")


# ============================================================
# 1. 训练所有层的 W_K, W_V
# ============================================================
print("\n[1] 训练各层 W_K, W_V...")

W_K_list = [None] * n_layers
W_V_list_global = [None] * n_layers

for l in range(n_layers):
    K_A, V_A = extract_kv_3d(model_A, l, train_padded, train_mask)
    K_B, V_B = extract_kv_3d(model_B, l, train_padded, train_mask)

    # 摊平成 [N, D] 做 Ridge
    K_A_flat = K_A.reshape(-1, D)
    V_A_flat = V_A.reshape(-1, D)
    K_B_flat = K_B.reshape(-1, D)
    V_B_flat = V_B.reshape(-1, D)

    W_K = fit_ridge(K_A_flat, K_B_flat)
    W_V = fit_ridge(V_A_flat, V_B_flat)
    W_K_list[l] = W_K
    W_V_list_global[l] = W_V

    if (l + 1) % 7 == 0:
        print(f"  Layer {l+1}/{n_layers}")

print("  Done.")

del model_A
gc.collect()
torch.cuda.empty_cache()


# ============================================================
# 2. 提取 eval 的 K/V（[B, T, D]）
# ============================================================
print("\n[2] 提取 eval K/V...")

model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_A.model.rotary_emb.inv_freq = model_A.model.rotary_emb.inv_freq.to(DEVICE)

K_A_eval = []
V_A_eval = []
for l in range(n_layers):
    K, V = extract_kv_3d(model_A, l, eval_padded, eval_mask)
    K_A_eval.append(K)       # [B_eval, T_eval, D]
    V_A_eval.append(V)

print("  Done.")

del model_A
gc.collect()
torch.cuda.empty_cache()


# ============================================================
# 3. 端到端训练 Layer 14 的 V mapper
# ============================================================
print(f"\n[3] 端到端训练 Layer {LAYER} 的 V mapper...")

n_eval = len(eval_padded)
perm = torch.randperm(n_eval)
n_val = int(n_eval * E2E_VAL_RATIO)
val_idx = perm[:n_val]
tr_idx = perm[n_val:]

print(f"  E2E train: {len(tr_idx)}, val: {len(val_idx)}")

mapper = VMapper(D, D, MLP_HIDDEN).to(DEVICE)
opt = torch.optim.Adam(mapper.parameters(), lr=E2E_LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E2E_EPOCHS)

best_val = float("inf")
best_state = None
wait = 0

for epoch in range(E2E_EPOCHS):
    # ---- train ----
    mapper.train()
    tr_loss = 0.0
    n_steps = 0
    perm_e = torch.randperm(len(tr_idx))

    for s in range(0, len(tr_idx), E2E_BATCH):
        e = min(s + E2E_BATCH, len(tr_idx))
        batch_articles = tr_idx[perm_e[s:e]]

        ids = eval_padded[batch_articles].to(DEVICE)
        mask = eval_mask[batch_articles].to(DEVICE)

        K_A_batch = [K_A_eval[l][batch_articles].to(DEVICE) for l in range(n_layers)]
        V_A_batch = [V_A_eval[l][batch_articles].to(DEVICE) for l in range(n_layers)]

        logits = forward_e2e(
            model_B, ids, mask,
            K_A_batch, V_A_batch, mapper,
            W_K_list, LAYER,
            n_q, n_kv, head_dim, n_layers,
        )

        shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        shift_labels = ids[:, 1:].reshape(-1)
        shift_mask = mask[:, 1:].reshape(-1).float()

        loss_per_token = F.cross_entropy(shift_logits, shift_labels, reduction="none")
        loss = (loss_per_token * shift_mask).sum() / shift_mask.sum().clamp_min(1)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
        opt.step()

        tr_loss += loss.item()
        n_steps += 1

        del logits, loss_per_token, loss, K_A_batch, V_A_batch
        torch.cuda.empty_cache()

    tr_loss /= n_steps
    scheduler.step()

    # ---- val ----
    mapper.eval()
    val_loss = 0.0
    n_val_steps = 0
    with torch.no_grad():
        for s in range(0, len(val_idx), E2E_BATCH):
            e = min(s + E2E_BATCH, len(val_idx))
            batch_articles = val_idx[s:e]

            ids = eval_padded[batch_articles].to(DEVICE)
            mask = eval_mask[batch_articles].to(DEVICE)

            K_A_batch = [K_A_eval[l][batch_articles].to(DEVICE) for l in range(n_layers)]
            V_A_batch = [V_A_eval[l][batch_articles].to(DEVICE) for l in range(n_layers)]

            logits = forward_e2e(
                model_B, ids, mask,
                K_A_batch, V_A_batch, mapper,
                W_K_list, LAYER,
                n_q, n_kv, head_dim, n_layers,
            )

            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            shift_labels = ids[:, 1:].reshape(-1)
            shift_mask = mask[:, 1:].reshape(-1).float()
            loss_per_token = F.cross_entropy(shift_logits, shift_labels, reduction="none")
            loss = (loss_per_token * shift_mask).sum() / shift_mask.sum().clamp_min(1)

            val_loss += loss.item()
            n_val_steps += 1

            del logits, loss_per_token, loss, K_A_batch, V_A_batch
            torch.cuda.empty_cache()

    val_loss /= n_val_steps

    print(f"  epoch {epoch+1:3d}  train={tr_loss:.4f}  val={val_loss:.4f}  "
          f"val_ppl={math.exp(val_loss):.4f}")

    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.clone() for k, v in mapper.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= E2E_PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

if best_state is not None:
    mapper.load_state_dict(best_state)

print(f"  Best val loss = {best_val:.4f}  (PPL = {math.exp(best_val):.4f})")


# ============================================================
# 4. PPL 评估
# ============================================================
print("\n" + "=" * 70)
print("PPL 评估")
print("=" * 70)


@torch.no_grad()
def ppl_native(model, input_ids, attention_mask, batch_size=2):
    total_loss, total_tokens = 0.0, 0
    for s in range(0, len(input_ids), batch_size):
        e = min(s + batch_size, len(input_ids))
        batch = input_ids[s:e].to(DEVICE)
        mask = attention_mask[s:e].to(DEVICE)
        outputs = model(input_ids=batch, attention_mask=mask)
        logits = outputs.logits
        sl = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        slb = batch[:, 1:].reshape(-1)
        sm = mask[:, 1:].reshape(-1).float()
        loss = F.cross_entropy(sl, slb, reduction="none")
        total_loss += (loss * sm).sum().item()
        total_tokens += sm.sum().item()
        del outputs, logits, loss
        torch.cuda.empty_cache()
    return math.exp(total_loss / total_tokens)


@torch.no_grad()
def ppl_transfer(model, input_ids, attention_mask, K_A_all, V_A_all,
                 mapper, W_K_list, layer_to_train, n_q, n_kv, head_dim,
                 n_layers, use_mapper=False, batch_size=1):
    total_loss, total_tokens = 0.0, 0
    for s in range(0, len(input_ids), batch_size):
        e = min(s + batch_size, len(input_ids))
        batch_articles = torch.arange(s, e)

        ids = input_ids[batch_articles].to(DEVICE)
        mask = attention_mask[batch_articles].to(DEVICE)

        K_A_batch = [K_A_all[l][batch_articles].to(DEVICE) for l in range(n_layers)]
        V_A_batch = [V_A_all[l][batch_articles].to(DEVICE) for l in range(n_layers)]

        if use_mapper:
            logits = forward_e2e(
                model, ids, mask, K_A_batch, V_A_batch, mapper,
                W_K_list, layer_to_train, n_q, n_kv, head_dim, n_layers,
            )
        else:
            # 全部线性（layer_to_train = -1）
            logits = forward_e2e(
                model, ids, mask, K_A_batch, V_A_batch, torch.nn.Identity().to(DEVICE),
                W_K_list, -1, n_q, n_kv, head_dim, n_layers,
            )

        sl = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        slb = ids[:, 1:].reshape(-1)
        sm = mask[:, 1:].reshape(-1).float()
        loss = F.cross_entropy(sl, slb, reduction="none")
        total_loss += (loss * sm).sum().item()
        total_tokens += sm.sum().item()

        del logits, loss
        torch.cuda.empty_cache()
    return math.exp(total_loss / total_tokens)


print("\n[1/3] PPL_native (1.7B 自己)...")
ppl_native_val = ppl_native(model_B, eval_padded, eval_mask)
print(f"  PPL_native = {ppl_native_val:.4f}")

print("\n[2/3] PPL_transfer (全线性)...")
ppl_linear = ppl_transfer(
    model_B, eval_padded, eval_mask,
    K_A_eval, V_A_eval, None, W_K_list, LAYER,
    n_q, n_kv, head_dim, n_layers,
    use_mapper=False, batch_size=1,
)
print(f"  PPL_linear = {ppl_linear:.4f}")

print("\n[3/3] PPL_transfer (Layer 14 用 E2E mapper)...")
ppl_e2e = ppl_transfer(
    model_B, eval_padded, eval_mask,
    K_A_eval, V_A_eval, mapper, W_K_list, LAYER,
    n_q, n_kv, head_dim, n_layers,
    use_mapper=True, batch_size=1,
)
print(f"  PPL_e2e = {ppl_e2e:.4f}")


# ============================================================
# 5. 汇总
# ============================================================
print()
print("=" * 75)
print("End-to-End Training Result")
print("=" * 75)
print(f"{'Method':>30}  {'PPL':>10}  {'Ratio':>10}")
print("-" * 75)
print(f"{'Native (1.7B)':>30}  {ppl_native_val:10.4f}  {1.0:10.4f}")
print(f"{'Transfer - All Linear':>30}  {ppl_linear:10.4f}  {ppl_linear/ppl_native_val:10.4f}")
print(f"{'Transfer - E2E (Layer 14)':>30}  {ppl_e2e:10.4f}  {ppl_e2e/ppl_native_val:10.4f}")
print("=" * 75)

print()
print("关键：")
print(f"  Linear ratio:  {ppl_linear/ppl_native_val:.4f}")
print(f"  E2E ratio:     {ppl_e2e/ppl_native_val:.4f}")
print(f"  Δratio:        {(ppl_e2e - ppl_linear) / ppl_native_val:+.4f}")