import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import numpy as np
import pandas as pd
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"
TRAIN_PATH = "../dataset/wikitext-02/train.jsonl"

N_TRAIN = 200
MAX_LEN = 128
RANK = 256
DEVICE = "cuda"
SEED = 42
STREAM_BATCH = 10000

torch.manual_seed(SEED)

print("加载模型...")
tok = AutoTokenizer.from_pretrained(MODEL_A)
model_A = AutoModelForCausalLM.from_pretrained(MODEL_A, dtype=torch.float16).to(DEVICE).eval()
model_B = AutoModelForCausalLM.from_pretrained(MODEL_B, dtype=torch.float16).to(DEVICE).eval()

L = min(model_A.config.num_hidden_layers, model_B.config.num_hidden_layers)
print(f"Layers: {L}")

df = pd.read_json(TRAIN_PATH, lines=True)
train_ids = []
for t in df["text"].iloc[:N_TRAIN].tolist():
    ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if len(ids) >= 32:
        train_ids.append(ids[:MAX_LEN].unsqueeze(0).to(DEVICE))
print(f"Train: {len(train_ids)}")

@torch.no_grad()
def extract_v(model, layer_idx, ids):
    attn = model.model.layers[layer_idx].self_attn
    out = model.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    normed = model.model.layers[layer_idx].input_layernorm(out.hidden_states[layer_idx])
    return attn.v_proj(normed)

@torch.no_grad()
def extract_v_flat(model, layer_idx, ids_list):
    outs = []
    for ids in ids_list:
        v = extract_v(model, layer_idx, ids)
        outs.append(v.reshape(-1, v.shape[-1]).float().cpu())
    return torch.cat(outs, 0)

def compute_cov(X, Y, batch_size=STREAM_BATCH):
    N = X.shape[0]
    mX = X.mean(dim=0)
    mY = Y.mean(dim=0)
    dX, dY = X.shape[1], Y.shape[1]
    Cxx = torch.zeros(dX, dX); Cyy = torch.zeros(dY, dY); Cxy = torch.zeros(dX, dY)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        x = X[s:e] - mX; y = Y[s:e] - mY
        Cxx += x.T @ x; Cyy += y.T @ y; Cxy += x.T @ y
    return Cxx/N, Cyy/N, Cxy/N, mX, mY

def cca_proj(Cxx, Cyy, Cxy, r, reg=1e-3):
    dX, dY = Cxx.shape[0], Cyy.shape[0]
    Cxx = Cxx + reg*torch.eye(dX); Cyy = Cyy + reg*torch.eye(dY)
    Lx = torch.linalg.cholesky(Cxx); Ly = torch.linalg.cholesky(Cyy)
    M = torch.linalg.solve_triangular(Lx, Cxy, upper=False)
    M = torch.linalg.solve_triangular(Ly, M.T, upper=False).T
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    W_S = torch.linalg.solve_triangular(Lx.T, U[:, :r], upper=True)
    return W_S / W_S.norm(dim=0, keepdim=True).clamp_min(1e-6)

def fit_M(A, Y, mean_Y, lam=1.0, batch_size=STREAM_BATCH):
    N, r = A.shape
    dY = Y.shape[1]
    AtA = torch.zeros(r, r); AtY = torch.zeros(r, dY)
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        a = A[s:e]; y = Y[s:e] - mean_Y
        AtA += a.T @ a; AtY += a.T @ y
    return torch.linalg.solve(AtA + lam*torch.eye(r), AtY)

print(f"\n学 CCA 投影（rank={RANK}）...")
proj = {}
for l in range(L):
    V_A = extract_v_flat(model_A, l, train_ids)
    V_B = extract_v_flat(model_B, l, train_ids)
    Cxx, Cyy, Cxy, mS, mT = compute_cov(V_A, V_B)
    W_S = cca_proj(Cxx, Cyy, Cxy, RANK)
    A = (V_A - mS) @ W_S
    M = fit_M(A, V_B, mT)
    proj[l] = {"W_S": W_S, "M": M, "mS": mS, "mT": mT}
    print(f"  Layer {l}: done")
    del V_A, V_B, Cxx, Cyy, Cxy
    gc.collect(); torch.cuda.empty_cache()

torch.save(proj, "cca_proj.pt")
print("\n保存到 cca_proj.pt")