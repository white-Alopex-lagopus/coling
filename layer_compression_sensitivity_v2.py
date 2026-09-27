#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_PATH = "../model/Qwen3-1.7B-base"
DATA_PATH = "../dataset/wikitext-02/kvtest.jsonl"
NUM_SAMPLES = 50
SEQ_LEN = 512
COMPRESSED_LEN = 64
TEST_K = True
TEST_V = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("./layer_sensitivity_results_v2")
OUTPUT_JSON = OUTPUT_DIR / "layer_sensitivity_v2.json"
SEED = 42

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def finite(name, x):
    ok = torch.isfinite(x).all().item()
    if not ok:
        bad = (~torch.isfinite(x)).sum().item()
        print(f"[WARNING] {name}: non-finite {bad}/{x.numel()}")
    return ok

def make_indices(n, k, device):
    if k >= n: return torch.arange(n, device=device)
    return torch.linspace(0, n-1, k, device=device).round().long().unique()

def repeat_kv(x, n_rep):
    if n_rep == 1: return x
    b,h,t,d = x.shape
    return x[:, :, None, :, :].expand(b,h,n_rep,t,d).reshape(b,h*n_rep,t,d)

def get_qkv(model, hidden, layer_idx, position_ids):
    attn = model.model.layers[layer_idx].self_attn
    b,t,_ = hidden.shape
    hd = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    qh = model.config.num_attention_heads
    kh = model.config.num_key_value_heads
    q = attn.q_proj(hidden).view(b,t,qh,hd).transpose(1,2)
    k = attn.k_proj(hidden).view(b,t,kh,hd).transpose(1,2)
    v = attn.v_proj(hidden).view(b,t,kh,hd).transpose(1,2)
    if hasattr(attn,"q_norm") and attn.q_norm is not None: q = attn.q_norm(q)
    if hasattr(attn,"k_norm") and attn.k_norm is not None: k = attn.k_norm(k)
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    # cos, sin = attn.rotary_emb(v, position_ids)
    cos, sin = model.model.rotary_emb(v, position_ids)
    # position_embeddings = (cos, sin)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q,k,v

def attention(q,k,v,key_positions):
    b, hq, tq, d = q.shape
    hkv = k.shape[1]
    k = repeat_kv(k, hq // hkv); v = repeat_kv(v, hq // hkv)
    scores = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(d)
    qpos = torch.arange(tq, device=q.device)
    allow = key_positions[None,:] <= qpos[:,None]
    # Prevent all-masked rows: for queries before first retained token, expose first retained token.
    empty = ~allow.any(dim=1)
    if empty.any(): allow[empty, 0] = True
    scores = scores.masked_fill(~allow[None,None,:,:], torch.finfo(scores.dtype).min)
    w = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(w,v)

def metrics(a,b):
    af=a.float().reshape(-1,a.shape[-1]); bf=b.float().reshape(-1,b.shape[-1])
    c=F.cosine_similarity(af,bf,dim=-1).mean().item()
    mse=F.mse_loss(b.float(),a.float()).item()
    rel=(torch.norm((b-a).float())/(torch.norm(a.float())+1e-8)).item()
    return c,mse,rel

@torch.no_grad()
def main():
    set_seed(SEED); OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    print("="*70); print("Layer-wise KV Compression Sensitivity v2"); print("="*70)
    print(f"Model: {MODEL_PATH}\nDataset: {DATA_PATH}\nSamples: {NUM_SAMPLES}\nSeq len: {SEQ_LEN}\nCompressed: {COMPRESSED_LEN}\nDevice: {DEVICE}")
    tok=AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    dtype=torch.bfloat16 if DEVICE=="cuda" else torch.float32
    model=AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=dtype, trust_remote_code=True).to(DEVICE).eval()
    df=pd.read_json(DATA_PATH,lines=True); text=df["text"].str.cat(sep="\n")
    ids=tok(text,return_tensors="pt").input_ids[0]
    need=NUM_SAMPLES*SEQ_LEN
    if len(ids)<need: raise ValueError(f"Need {need} tokens, got {len(ids)}")
    ids=ids[:need].reshape(NUM_SAMPLES,SEQ_LEN).to(DEVICE)
    selected=make_indices(SEQ_LEN,COMPRESSED_LEN,DEVICE)
    print("Selected positions:",selected[:20].tolist(),"...")
    hidden=model.model.embed_tokens(ids)
    pos=torch.arange(SEQ_LEN,device=DEVICE).unsqueeze(0).expand(NUM_SAMPLES,-1)
    results=[]
    for li,layer in enumerate(model.model.layers):
        print(f"\n[{li:02d}/{len(model.model.layers)-1:02d}] processing...")
        if not finite(f"layer_{li}_input",hidden):
            results.append({"layer":li,"status":"non_finite_input"}); break
        q,k,v=get_qkv(model,hidden,li,pos)
        if not (finite("q",q) and finite("k",k) and finite("v",v)):
            results.append({"layer":li,"status":"non_finite_qkv"}); break
        full_pos=torch.arange(SEQ_LEN,device=DEVICE)
        full=attention(q,k,v,full_pos)
        kc=k[:,:,selected,:] if TEST_K else k
        vc=v[:,:,selected,:] if TEST_V else v
        kp=selected if TEST_K else full_pos
        comp=attention(q,kc,vc,kp)
        if not finite("full_attention",full) or not finite("compressed_attention",comp):
            results.append({"layer":li,"status":"non_finite_attention"}); break
        c,m,r=metrics(full,comp)
        results.append({"layer":li,"cosine":c,"mse":m,"relative_l2":r,"status":"ok"})
        print(f"    cosine={c:.6f}  mse={m:.6e}  relativeL2={r:.6f}")
        # Real, unmodified full-model forward to produce next layer's hidden state.
        residual=hidden
        x=layer.input_layernorm(hidden)
        # attn_out=layer.self_attn(x, position_ids=pos)
        
        
        cos, sin = model.model.rotary_emb(x, pos)
        # 2. 构造 causal mask（加性，上三角为 -inf）
        T_len = x.shape[1]
        causal_mask = torch.triu(
            torch.full(
                (1, 1, T_len, T_len),
                torch.finfo(x.dtype).min,
                device=x.device,
                dtype=x.dtype,
            ),
            diagonal=1,
        )

        # 3. 用正确的签名调用
        attn_out = layer.self_attn(
            x,
            position_embeddings=(cos, sin),
            attention_mask=causal_mask,
        )
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        
        
        if isinstance(attn_out,tuple): attn_out=attn_out[0]
        hidden=residual+attn_out
        residual=hidden
        x=layer.post_attention_layernorm(hidden)
        hidden=residual+layer.mlp(x)
        if DEVICE=="cuda": torch.cuda.empty_cache()
    out={"model":MODEL_PATH,"dataset":DATA_PATH,"num_samples":NUM_SAMPLES,"seq_len":SEQ_LEN,"compressed_len":COMPRESSED_LEN,"test_k":TEST_K,"test_v":TEST_V,"selection":"uniform_original_positions","results":results}
    with open(OUTPUT_JSON,"w",encoding="utf-8") as f: json.dump(out,f,indent=2,ensure_ascii=False)
    print("\n"+"="*70); print("FINAL RESULTS"); print("="*70)
    print(f"{'Layer':>5} {'Cosine':>12} {'MSE':>14} {'Rel-L2':>12} {'Status':>24}")
    for r in results:
        if r["status"]=="ok": print(f"{r['layer']:5d} {r['cosine']:12.4f} {r['mse']:14.6e} {r['relative_l2']:12.4f} {r['status']:>24}")
        else: print(f"{r['layer']:5d} {'-':>12} {'-':>14} {'-':>12} {r['status']:>24}")
    print(f"Saved to: {OUTPUT_JSON}")

if __name__=="__main__": main()
