import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 配置
# ============================================================
MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"
DATA_PATH = "../dataset/wikitext-02/train.jsonl"

MAX_ARTICLES = 9007         # 先 2000 跑通，再加大
MAX_TOKENS_PER_ART = 256
LAYER = 14
DEVICE = "cuda"
SEED = 42

RIDGE_BATCH = 50000
MLP_EPOCHS = 50
MLP_LR = 1e-3
MLP_BATCH = 256
TRAIN_RATIO = 0.8

torch.manual_seed(SEED)


# ============================================================
# 数据
# ============================================================
df = pd.read_json(DATA_PATH, lines=True)
texts = df["text"].iloc[:MAX_ARTICLES].tolist()
tok = AutoTokenizer.from_pretrained(MODEL_A)

article_ids = []
for t in texts:
    ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    ids = ids[:MAX_TOKENS_PER_ART]
    if len(ids) >= 16:
        article_ids.append(ids)

max_len = max(len(x) for x in article_ids)
padded = torch.zeros(len(article_ids), max_len, dtype=torch.long)
for i, ids in enumerate(article_ids):
    padded[i, :len(ids)] = ids

B = len(padded)
T = max_len
print(f"文章数: {B}, 每篇最多 {T} token, 总 token: {B * T}")


# ============================================================
# 工具
# ============================================================
def r2_score(pred, target):
    pred = pred.float()
    target = target.float()
    return (1 - ((pred - target)**2).sum() / (target**2).sum()).item()


def fit_ridge(X, Y, lam=100.0, batch_size=RIDGE_BATCH):
    D_X = X.shape[-1]
    D_Y = Y.shape[-1]
    XtX = torch.zeros(D_X, D_X, dtype=torch.float32)
    XtY = torch.zeros(D_X, D_Y, dtype=torch.float32)
    N = X.shape[0]
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        x = X[s:e].float(); y = Y[s:e].float()
        XtX += x.T @ x
        XtY += x.T @ y
    return torch.linalg.solve(XtX + lam * torch.eye(D_X), XtY)


def apply_W(X, W, batch_size=RIDGE_BATCH):
    N = X.shape[0]
    outs = []
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        outs.append((X[s:e].float() @ W).cpu())
    return torch.cat(outs, 0)


# ============================================================
# 提取 V
# ============================================================
@torch.no_grad()
def extract_v(model, layer_idx, padded):
    attn = model.model.layers[layer_idx].self_attn
    n_kv = model.config.num_key_value_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)

    all_v = []
    batch_size = 64

    for start in range(0, len(padded), batch_size):
        batch = padded[start:start+batch_size].to(DEVICE)
        B_b, T_b = batch.shape

        out = model.model(input_ids=batch, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states[layer_idx]
        normed = model.model.layers[layer_idx].input_layernorm(hidden)

        v = attn.v_proj(normed).view(B_b, T_b, n_kv, head_dim).transpose(1, 2)
        all_v.append(v.transpose(1, 2).reshape(B_b, T_b, -1).float().cpu())

        del out, hidden, normed, v
        torch.cuda.empty_cache()

    return torch.cat(all_v, dim=0).reshape(-1, all_v[0].shape[-1])


# ============================================================
# MLP
# ============================================================
class VMapper(torch.nn.Module):
    def __init__(self, d_in=1024, d_out=1024, hidden=2048):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, d_out),
        )
    def forward(self, x):
        return self.net(x)


# ============================================================
# 加载模型
# ============================================================
print("\n加载模型...")
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()


# ============================================================
# 提取 V
# ============================================================
print(f"\n提取第 {LAYER} 层 V...")
V_A = extract_v(model_A, LAYER, padded)
V_B = extract_v(model_B, LAYER, padded)
N = V_A.shape[0]
D = V_A.shape[1]
print(f"  V_A: {V_A.shape}, V_B: {V_B.shape}")

# train/test 划分（只存索引，不复制数据）
perm = torch.randperm(N)
n_train = int(N * TRAIN_RATIO)
train_idx = perm[:n_train]
test_idx = perm[n_train:]
print(f"  train: {len(train_idx)}, test: {len(test_idx)}")


# ============================================================
# 方法 1：Identity
# ============================================================
print("\n[方法 1] Identity...")
r2_identity = r2_score(V_A[test_idx], V_B[test_idx])
print(f"  Test R² = {r2_identity:.4f}")


# ============================================================
# 方法 2：Random W
# ============================================================
print("\n[方法 2] Random W...")
W_rand = torch.randn(D, D)
Q_r, _ = torch.linalg.qr(W_rand)
V_rand = V_A[test_idx] @ Q_r
r2_rand = r2_score(V_rand, V_B[test_idx])
print(f"  Test R² = {r2_rand:.4f}")


# ============================================================
# 方法 3：Global Linear W
# ============================================================
print("\n[方法 3] Global Linear W (Ridge)...")

# 只在训练集上拟合
XtX = torch.zeros(D, D, dtype=torch.float32)
XtY = torch.zeros(D, D, dtype=torch.float32)
for s in range(0, len(train_idx), RIDGE_BATCH):
    e = min(s + RIDGE_BATCH, len(train_idx))
    idx = train_idx[s:e]
    x = V_A[idx].float()
    y = V_B[idx].float()
    XtX += x.T @ x
    XtY += x.T @ y

W_global = torch.linalg.solve(XtX + 100.0 * torch.eye(D), XtY)

# 只在测试集上评估
V_global_test = V_A[test_idx].float() @ W_global
r2_global = r2_score(V_global_test, V_B[test_idx])
print(f"  Test R² = {r2_global:.4f}")


# ============================================================
# 方法 4：MLP（逐 batch 搬 GPU）
# ============================================================
print(f"\n[方法 4] V MLP ({MLP_EPOCHS} epochs)...")

mapper = VMapper(D, D, hidden=2048).to(DEVICE)
opt = torch.optim.Adam(mapper.parameters(), lr=MLP_LR)
loss_fn = torch.nn.MSELoss()

n_train_n = len(train_idx)

for epoch in range(MLP_EPOCHS):
    mapper.train()
    perm_e = torch.randperm(n_train_n)
    total_loss = 0.0
    n_steps = 0

    for s in range(0, n_train_n, MLP_BATCH):
        e = min(s + MLP_BATCH, n_train_n)
        # 从 train_idx 里取出当前 batch 的原始索引
        batch_orig_idx = train_idx[perm_e[s:e]]

        # 只搬当前 batch 到 GPU
        x = V_A[batch_orig_idx].to(DEVICE)
        y = V_B[batch_orig_idx].to(DEVICE)

        pred = mapper(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        n_steps += 1

        del x, y, pred, loss
        if n_steps % 100 == 0:
            torch.cuda.empty_cache()

    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1:3d}  loss = {total_loss/n_steps:.6f}")


# 评估（逐 batch）
mapper.eval()
with torch.no_grad():
    V_mlp_test = []
    for s in range(0, len(test_idx), 4096):
        e = min(s + 4096, len(test_idx))
        idx = test_idx[s:e]
        x = V_A[idx].to(DEVICE)
        V_mlp_test.append(mapper(x).cpu())
        del x
        torch.cuda.empty_cache()
    V_mlp_test = torch.cat(V_mlp_test, 0)

r2_mlp = r2_score(V_mlp_test, V_B[test_idx])
print(f"  Test R² = {r2_mlp:.4f}")


# ============================================================
# 汇总
# ============================================================
print()
print("=" * 70)
print(f"V Mapping Comparison (Layer {LAYER})")
print("=" * 70)
print(f"{'Method':>30}  {'Test R²':>10}")
print("-" * 70)
print(f"{'Identity (no map)':>30}  {r2_identity:10.4f}")
print(f"{'Random orthogonal W':>30}  {r2_rand:10.4f}")
print(f"{'Global Linear W (Ridge)':>30}  {r2_global:10.4f}")
print(f"{'MLP (2-layer)':>30}  {r2_mlp:10.4f}")
print("=" * 70)