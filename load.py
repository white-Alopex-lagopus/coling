import os
import time
# import sys
# sys.path.insert(0, '/home/baimingyu/workspace/gpu_monitor')

# from gpu_detector import GPUDetector, wait_for_gpu

# gpu_id = wait_for_gpu(gpu_ids=[0, 1])  # 无限等待
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
import torch
import gc


# 直接读取为 DataFrame
df = pd.read_json('../dataset/wikitext-02/kvtest.jsonl', lines=True)
# print(df.head())

text = df['text'].str.cat(sep="\n")

tokenizer = AutoTokenizer.from_pretrained(
    "../model/Qwen3-0.6B-base"
)

tokens = tokenizer(
    text,
    add_special_tokens=False,
    return_tensors="pt"
)["input_ids"][0]

nums = 50
seq_len = 512

tokens = tokens[:nums * seq_len]

input_ids = tokens.reshape(
    nums,
    seq_len
)

print(input_ids.shape)

torch.save(input_ids.cpu(), "./kv_cache/input_ids.pt")
print("saved input_ids.pt")

raise SystemExit

# model_path = "../model/Qwen3-0.6B-base"
model_path = {
    "0.6B" : "../model/Qwen3-0.6B-base", 
    "1.7B" : "../model/Qwen3-1.7B-base",
    "4B" : "../model/Qwen3-4B-base",
}

os.makedirs("kv_cache", exist_ok=True)

K_flat = []

for tag, p in model_path.items():
    model = AutoModelForCausalLM.from_pretrained(
        p,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    model.eval()
    
    input_ids = input_ids.to(model.device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=True
        )
        
    layer_id = model.config.num_hidden_layers // 2
    
    # key = outputs.past_key_values[layer_id][0]
    # val = outputs.past_key_values[layer_id][1]
    
    # key_cpu = key.float().cpu().contiguous()
    # val_cpu = val.float().cpu().contiguous()
    
    #  分析头的时候不展平，保留 [50, 8, 512, 128]
    # key_flat = key.reshape(-1, key.shape[-1])
    # K_flat.append(key_flat.float())
    
    # K_flat.append(key.float())
    
    # save_path = f"kv_cache/K_{tag}.pt"
    # save_path = f"kv_cache/V_{tag}.pt"
    # torch.save(key_cpu, save_path)
    # torch.save(val_cpu, save_path)
    # print(f"{tag}: {tuple(val_cpu.shape)} -> {save_path}")
    
    del model, outputs
    torch.cuda.empty_cache()
    gc.collect()
    
    

# from head_LOO import loo_headwise
# pairs = [
#     ("0.6B + 1.7B -> 4B", K_flat[0], K_flat[1], K_flat[2]),
#     ("0.6B + 4B -> 1.7B", K_flat[0], K_flat[2], K_flat[1]),
#     ("1.7B + 4B -> 0.6B", K_flat[1], K_flat[2], K_flat[0]),
# ]

# for name, K1, K2, Ktest in pairs:

#     print("\n==============================")
#     print(name)
#     print("==============================")

#     loo_headwise(K1, K2, Ktest, k=60)    
    

# head_PCA

# import PCA
# models = ["0.6B", "1.7B", "4B"]

# ks = [20, 40, 60, 80, 100]

# for i in range(0, 3):

#     print(f"\n========== {models[i]} ==========")

#     for head in range(8):

#         K_head = K_flat[i][:, head, :, :].reshape(-1, 128)

#         mean, U = PCA.fit_pca_basis(K_head, 60)
#         ev = PCA.explained_variance(K_head, mean, U)

#         print(f"head {head}: EV@60 = {ev:.4f}")




#  PCA
# import PCA
# models = ["0.6B", "1.7B", "4B"]
# ks = [10, 20, 30, 40, 50, 60, 80, 100]

# for i in range(0, 3):

#     print(f"\n===== {models[i]} Self-PCA =====")

#     for k in ks:

#         mean, U = PCA.fit_pca_basis(K_flat[i], k)

#         ev = PCA.explained_variance(
#             K_flat[i],
#             mean,
#             U
#         )

#         print(f"k={k:3d}, EV={ev:.4f}")



# import LOO
# train = torch.cat(
#     [
#         # K_flat[0],
#         K_flat[1],
#         K_flat[2]
#     ],
#     dim=0
# )

# for k in [10, 20, 30, 40, 50, 60, 80, 100]:
#     mean, U = LOO.fit_pca_basis(
#         train,
#         k=k
#     )

#     score = LOO.explained_variance(
#         K_flat[0],
#         mean,
#         U
#     )

#     # print(score)
#     print(
#         f"k={k:3d}, EV={score:.4f}"
#     )



# K_joint = torch.cat(K_flat, dim=0)
    
# print(K_joint.shape)

# K_joint = K_joint.float()

# mean = K_joint.mean(dim=0, keepdim=True)
# K_centered = K_joint - mean

# cov = K_centered.T @ K_centered / (K_centered.shape[0] - 1)

# eigenvalues, eigenvectors = torch.linalg.eigh(cov)

# idx = torch.argsort(eigenvalues, descending=True)
# eigenvalues = eigenvalues[idx]
# eigenvectors = eigenvectors[:, idx]

# explained_variance_ratio = (
#     eigenvalues.cumsum(dim=0) / eigenvalues.sum()
# )

# for threshold in [0.90, 0.95, 0.99]:
#     dim = torch.where(
#         explained_variance_ratio >= threshold
#     )[0][0].item() + 1

#     print(f"Joint d_{threshold:.2f} = {dim}")
    
# for k in [1, 2, 5, 10, 20, 30, 40, 50, 64, 80, 100, 128]:
#     print(
#         f"{k}: "
#         f"{explained_variance_ratio[k-1].item():.4f}"
#     )




# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# input_ids = input_ids.to(model.device)

# with torch.no_grad():
#     outputs = model(
#         input_ids=input_ids,
#         use_cache=True
#     )


# layer_id = model.config.num_hidden_layers // 2

# key = outputs.past_key_values[layer_id][0]

# print("K:", key.shape)

# key_flat = key.reshape(-1, key.shape[-1])

# print("K_flat:", key_flat.shape)

# #--------------------------------------------------------------------------------
# K_flat = key_flat.float()

# 1. Center
# mean = K_flat.mean(dim=0, keepdim=True)
# K_centered = K_flat - mean

# # 2. Covariance
# cov = K_centered.T @ K_centered / (K_centered.shape[0] - 1)

# # 3. Eigen decomposition
# eigenvalues, eigenvectors = torch.linalg.eigh(cov)

# # 4. Sort descending
# idx = torch.argsort(eigenvalues, descending=True)
# eigenvalues = eigenvalues[idx]
# eigenvectors = eigenvectors[:, idx]

# # 5. Explained variance
# explained_variance_ratio = (
#     eigenvalues.cumsum(dim=0) / eigenvalues.sum()
# )

# print("\nExplained variance:")
# for k in [1, 2, 5, 10, 20, 30, 40, 50, 64, 80, 100, 128]:
#     print(f"{k:3d} : {explained_variance_ratio[k-1].item():.4f}")
    
    
# for threshold in [0.90, 0.95, 0.99]:
#     dim = torch.where(
#         explained_variance_ratio >= threshold
#     )[0][0].item() + 1

#     print(f"d_{threshold:.2f} = {dim}")