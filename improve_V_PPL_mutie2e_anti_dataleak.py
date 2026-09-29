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

TRAIN_PATH = "../dataset/wikitext-02/train.jsonl"
VAL_PATH   = "../dataset/wikitext-02/validation.jsonl"
TEST_PATH  = "../dataset/wikitext-02/test.jsonl"

N_TRAIN = 3000       # 流式后可以用更多
N_VAL   = 500
N_TEST  = 500

MAX_TOKENS_PER_ART = 128

LAYERS_TO_TRAIN = [0, 2, 5, 7, 9, 11, 14, 16, 18, 20, 23, 25, 27]
# LAYERS_TO_TRAIN = [0, 2, 5, 7, 9, 11, 13, 15, 17, 20, 22, 25, 27]

DEVICE = "cuda"
SEED = 42
RIDGE_BATCH = 50000
RIDGE_LAMBDA = 100.0

E2E_LR = 1e-4
E2E_EPOCHS = 30
E2E_BATCH = 2        # 流式后可以调大
E2E_PATIENCE = 5

MLP_HIDDEN = 512

torch.manual_seed(SEED)


# ============================================================
# 数据
# ============================================================
tok = AutoTokenizer.from_pretrained(MODEL_A)


def load_data(path, n):
    df = pd.read_json(path, lines=True)
    texts = df["text"].iloc[:n].tolist()
    ids_list = []
    for t in texts:
        ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        ids = ids[:MAX_TOKENS_PER_ART]
        if len(ids) >= 16:
            ids_list.append(ids)
    L = max(len(x) for x in ids_list)
    padded = torch.zeros(len(ids_list), L, dtype=torch.long)
    mask = torch.zeros(len(ids_list), L, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, :len(ids)] = ids
        mask[i, :len(ids)] = 1
    return padded, mask


train_padded, train_mask = load_data(TRAIN_PATH, N_TRAIN)
val_padded,   val_mask   = load_data(VAL_PATH,   N_VAL)
test_padded,  test_mask  = load_data(TEST_PATH,  N_TEST)

print(f"Train: {train_padded.shape}")
print(f"Val  : {val_padded.shape}")
print(f"Test : {test_padded.shape}")


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


# ============================================================
# 流式提取 K/V（只对当前 batch）
# ============================================================
@torch.no_grad()
def extract_kv_for_batch(model_A, padded, mask, article_indices,
                         n_layers, n_kv, head_dim):
    """
    只对指定文章提取 28 层的 K/V。
    返回 K_list, V_list，各 28 个张量 [b, T, D]（GPU）
    """
    ids = padded[article_indices].to(DEVICE)
    m = mask[article_indices].to(DEVICE)
    B_b, T_b = ids.shape

    out = model_A.model(input_ids=ids, attention_mask=m,
                        output_hidden_states=True, use_cache=False)

    K_list = []
    V_list = []
    for l in range(n_layers):
        layer = model_A.model.layers[l]
        attn = layer.self_attn
        hidden = out.hidden_states[l]
        normed = layer.input_layernorm(hidden)

        k = attn.k_proj(normed).view(B_b, T_b, n_kv, head_dim)
        k = attn.k_norm(k).transpose(1, 2).transpose(1, 2).reshape(B_b, T_b, -1)
        K_list.append(k)     # [B, T, D] GPU

        v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
        v = v.transpose(1, 2).reshape(B_b, T_b, -1)
        V_list.append(v)

    del out
    return K_list, V_list


# ============================================================
# 1. 预提取 train 的 K/V（用于训练 W_K/W_V）
#    这个必须做，因为要一次性 Ridge
#    但可以分批提取 + 累加，不保留全量
# ============================================================
print("\n[1] 训练 W_K, W_V...")

# 加载 model_A 和 model_B
print("加载模型...")
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


# 用流式累加 XtX, XtY 来训练 Ridge
def fit_ridge_streaming(model_a, model_b, padded, mask, n_layers, D,
                         batch_articles=64, lam=RIDGE_LAMBDA):
    """
    流式训练 Ridge。对每层分别累加 XtX, XtY。
    """
    W_K_list = [None] * n_layers
    W_V_list = [None] * n_layers

    XtX_K = [torch.zeros(D, D, dtype=torch.float32) for _ in range(n_layers)]
    XtY_K = [torch.zeros(D, D, dtype=torch.float32) for _ in range(n_layers)]
    XtX_V = [torch.zeros(D, D, dtype=torch.float32) for _ in range(n_layers)]
    XtY_V = [torch.zeros(D, D, dtype=torch.float32) for _ in range(n_layers)]

    N_art = len(padded)
    for s in range(0, N_art, batch_articles):
        e = min(s + batch_articles, N_art)
        ba = torch.arange(s, e)

        # 提 A 的 K/V
        K_A_b, V_A_b = extract_kv_for_batch(model_a, padded, mask, ba,
                                             n_layers, n_kv, head_dim)

        # 提 B 的 K/V
        with torch.no_grad():
            ids = padded[ba].to(DEVICE)
            m = mask[ba].to(DEVICE)
            out = model_b.model(input_ids=ids, attention_mask=m,
                                output_hidden_states=True, use_cache=False)
            K_B_b, V_B_b = [], []
            for l in range(n_layers):
                layer = model_b.model.layers[l]
                attn = layer.self_attn
                hidden = out.hidden_states[l]
                normed = layer.input_layernorm(hidden)
                B_b, T_b = ids.shape
                k = attn.k_proj(normed).view(B_b, T_b, n_kv, head_dim)
                k = attn.k_norm(k).transpose(1, 2).transpose(1, 2).reshape(B_b, T_b, -1)
                K_B_b.append(k)
                v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
                v = v.transpose(1, 2).reshape(B_b, T_b, -1)
                V_B_b.append(v)
            del out

        # 累加
        for l in range(n_layers):
            Ka = K_A_b[l].reshape(-1, D).float().cpu()
            Va = V_A_b[l].reshape(-1, D).float().cpu()
            Kb = K_B_b[l].reshape(-1, D).float().cpu()
            Vb = V_B_b[l].reshape(-1, D).float().cpu()

            XtX_K[l] += Ka.T @ Ka
            XtY_K[l] += Ka.T @ Kb
            XtX_V[l] += Va.T @ Va
            XtY_V[l] += Va.T @ Vb

        del K_A_b, V_A_b, K_B_b, V_B_b
        torch.cuda.empty_cache()
        if (s // batch_articles + 1) % 5 == 0:
            print(f"  Ridge batch {s//batch_articles + 1}/{(N_art + batch_articles - 1)//batch_articles}")

    # 解 W
    for l in range(n_layers):
        W_K_list[l] = torch.linalg.solve(XtX_K[l] + lam * torch.eye(D), XtY_K[l])
        W_V_list[l] = torch.linalg.solve(XtX_V[l] + lam * torch.eye(D), XtY_V[l])

    return W_K_list, W_V_list


W_K_list, W_V_list = fit_ridge_streaming(
    model_A, model_B, train_padded, train_mask,
    n_layers, D, batch_articles=64
)

print("  Done.")


# ============================================================
# 2. E2E 训练（流式）
# ============================================================
print(f"\n[2] E2E 训练 {len(LAYERS_TO_TRAIN)} 层（流式）...")
print(f"  Layers: {LAYERS_TO_TRAIN}")

mappers = {l: VMapper(D, D, MLP_HIDDEN).to(DEVICE) for l in LAYERS_TO_TRAIN}
all_params = []
for m in mappers.values():
    all_params += list(m.parameters())

opt = torch.optim.Adam(all_params, lr=E2E_LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E2E_EPOCHS)

best_val = float("inf")
best_state = None
wait = 0


def forward_e2e(model, input_ids, attention_mask,
                K_A_all, V_A_all, mappers,
                W_K_list, W_V_list, layers_to_train,
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


def run_epoch(padded, mask, train_mode):
    if train_mode:
        for m in mappers.values():
            m.train()
    else:
        for m in mappers.values():
            m.eval()

    total_loss, total_tokens = 0.0, 0
    order = torch.randperm(len(padded)) if train_mode else torch.arange(len(padded))

    with torch.set_grad_enabled(train_mode):
        for s in range(0, len(padded), E2E_BATCH):
            e = min(s + E2E_BATCH, len(padded))
            ba = order[s:e]

            ids = padded[ba].to(DEVICE)
            m_ = mask[ba].to(DEVICE)

            # ★ 流式提取当前 batch 的 K/V
            K_b, V_b = extract_kv_for_batch(
                model_A, padded, mask, ba, n_layers, n_kv, head_dim
            )

            logits = forward_e2e(model_B, ids, m_, K_b, V_b, mappers,
                                 W_K_list, W_V_list, LAYERS_TO_TRAIN,
                                 n_q, n_kv, head_dim, n_layers)

            sl = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            slb = ids[:, 1:].reshape(-1)
            sm = m_[:, 1:].reshape(-1).float()
            lp = F.cross_entropy(sl, slb, reduction="none")
            loss = (lp * sm).sum() / sm.sum().clamp_min(1)

            if train_mode:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                opt.step()

            total_loss += loss.item() * sm.sum().item()
            total_tokens += sm.sum().item()

            del logits, lp, loss, K_b, V_b
            torch.cuda.empty_cache()

    return total_loss / total_tokens


for epoch in range(E2E_EPOCHS):
    tr_loss = run_epoch(train_padded, train_mask, True)
    scheduler.step()
    val_loss = run_epoch(val_padded, val_mask, False)

    print(f"  epoch {epoch+1:3d}  train={tr_loss:.4f}  val={val_loss:.4f}  "
          f"val_ppl={math.exp(val_loss):.4f}")

    if val_loss < best_val:
        best_val = val_loss
        best_state = {l: {k: v.clone() for k, v in m.state_dict().items()}
                      for l, m in mappers.items()}
        wait = 0
    else:
        wait += 1
        if wait >= E2E_PATIENCE:
            print(f"  Early stop at epoch {epoch+1}")
            break

if best_state is not None:
    for l, sd in best_state.items():
        mappers[l].load_state_dict(sd)

print(f"  Best val PPL = {math.exp(best_val):.4f}")


# ============================================================
# 3. Test PPL（流式）
# ============================================================
print("\n" + "=" * 70)
print("Test PPL（test 从未参与训练）")
print("=" * 70)


@torch.no_grad()
def ppl_native(padded, mask, batch_size=2):
    total_loss, total_tokens = 0.0, 0
    for s in range(0, len(padded), batch_size):
        e = min(s + batch_size, len(padded))
        ids = padded[s:e].to(DEVICE)
        m = mask[s:e].to(DEVICE)
        out = model_B(input_ids=ids, attention_mask=m)
        logits = out.logits
        sl = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        slb = ids[:, 1:].reshape(-1)
        sm = m[:, 1:].reshape(-1).float()
        lp = F.cross_entropy(sl, slb, reduction="none")
        total_loss += (lp * sm).sum().item()
        total_tokens += sm.sum().item()
        del out, logits, lp
        torch.cuda.empty_cache()
    return math.exp(total_loss / total_tokens)


@torch.no_grad()
def ppl_transfer_stream(padded, mask, use_e2e=False, batch_size=2):
    for m in mappers.values():
        m.eval()
    total_loss, total_tokens = 0.0, 0
    for s in range(0, len(padded), batch_size):
        e = min(s + batch_size, len(padded))
        ba = torch.arange(s, e)

        ids = padded[ba].to(DEVICE)
        m_ = mask[ba].to(DEVICE)

        K_b, V_b = extract_kv_for_batch(
            model_A, padded, mask, ba, n_layers, n_kv, head_dim
        )

        if use_e2e:
            logits = forward_e2e(model_B, ids, m_, K_b, V_b, mappers,
                                 W_K_list, W_V_list, LAYERS_TO_TRAIN,
                                 n_q, n_kv, head_dim, n_layers)
        else:
            logits = forward_e2e(model_B, ids, m_, K_b, V_b, {},
                                 W_K_list, W_V_list, [],
                                 n_q, n_kv, head_dim, n_layers)

        sl = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        slb = ids[:, 1:].reshape(-1)
        sm = m_[:, 1:].reshape(-1).float()
        lp = F.cross_entropy(sl, slb, reduction="none")
        total_loss += (lp * sm).sum().item()
        total_tokens += sm.sum().item()

        del logits, lp, K_b, V_b
        torch.cuda.empty_cache()
    return math.exp(total_loss / total_tokens)


ppl_nat = ppl_native(test_padded, test_mask)
print(f"PPL_native = {ppl_nat:.4f}")

ppl_lin = ppl_transfer_stream(test_padded, test_mask, use_e2e=False, batch_size=2)
print(f"PPL_linear = {ppl_lin:.4f}")

ppl_e2e = ppl_transfer_stream(test_padded, test_mask, use_e2e=True, batch_size=2)
print(f"PPL_e2e    = {ppl_e2e:.4f}")


# ============================================================
# 4. 汇总
# ============================================================
print()
print("=" * 75)
print("Final Result (streaming, no leakage)")
print("=" * 75)
print(f"Layers trained: {LAYERS_TO_TRAIN}")
print(f"Data: Train={len(train_padded)}, Val={len(val_padded)}, Test={len(test_padded)}")
print()
print(f"{'Method':>20}  {'PPL':>10}  {'Ratio':>10}")
print("-" * 75)
print(f"{'Native (1.7B)':>20}  {ppl_nat:10.4f}  {1.0:10.4f}")
print(f"{'All Linear':>20}  {ppl_lin:10.4f}  {ppl_lin/ppl_nat:10.4f}")
print(f"{'E2E':>20}  {ppl_e2e:10.4f}  {ppl_e2e/ppl_nat:10.4f}")
print("=" * 75)
print()
print(f"ΔRatio (E2E - Linear): {(ppl_e2e - ppl_lin) / ppl_nat:+.4f}")