import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# ============================================================
# Config
# ============================================================

MODEL_A = "../model/Qwen3-0.6B-base"
MODEL_B = "../model/Qwen3-1.7B-base"

K_A_PATH = "./kv_cache/K_0.6B.pt"
V_A_PATH = "./kv_cache/V_0.6B.pt"

K_B_PATH = "./kv_cache/K_1.7B.pt"
V_B_PATH = "./kv_cache/V_1.7B.pt"

INPUT_IDS_PATH = "./kv_cache/input_ids.pt"

DEVICE = "cuda"
DTYPE = torch.float16

PCA_DIM = 60

# Qwen3:
# 0.6B: 28 layers -> 14
# 1.7B: 28 layers -> 14
LAYER_ID = 14


# ============================================================
# PCA
# ============================================================

def fit_pca(X, k):
    """
    X: [N, D]

    return:
        mean: [1, D]
        U:    [D, k]
    """

    X = X.float()

    mean = X.mean(dim=0, keepdim=True)

    X_centered = X - mean

    # covariance matrix
    cov = X_centered.T @ X_centered
    cov = cov / X_centered.shape[0]

    eigvals, eigvecs = torch.linalg.eigh(cov)

    # descending
    indices = torch.argsort(
        eigvals,
        descending=True
    )

    U = eigvecs[:, indices[:k]]

    return mean, U


def project_to_subspace(X, mean, U):
    """
    X:
        [N, D]

    mean:
        [1, D]

    U:
        [D, k]

    Return:
        PCA reconstruction in original D-dimensional space
    """

    X_centered = X - mean

    Z = X_centered @ U

    X_hat = Z @ U.T + mean

    return X_hat


# ============================================================
# Cosine
# ============================================================

def cosine_similarity(A, B):
    """
    Global cosine similarity.
    """

    A = A.float().reshape(-1)
    B = B.float().reshape(-1)

    return F.cosine_similarity(
        A.unsqueeze(0),
        B.unsqueeze(0)
    ).item()


def mean_token_cosine(A, B):
    """
    A/B:
        [B, H, T, D]

    First flatten heads and D for each token,
    then calculate cosine for each token,
    finally average.
    """

    A = A.float()
    B = B.float()

    # [B, H, T, D]
    A = A.transpose(1, 2)
    B = B.transpose(1, 2)

    # [B, T, H*D]
    A = A.reshape(
        A.shape[0],
        A.shape[1],
        -1
    )

    B = B.reshape(
        B.shape[0],
        B.shape[1],
        -1
    )

    cos = F.cosine_similarity(
        A,
        B,
        dim=-1
    )

    return cos.mean().item()


# ============================================================
# Qwen3 attention
# ============================================================

def repeat_kv(hidden_states, n_rep):
    """
    Qwen3 GQA.

    Input:
        [B, num_kv_heads, T, D]

    Output:
        [B, num_attention_heads, T, D]
    """

    if n_rep == 1:
        return hidden_states

    B, H, T, D = hidden_states.shape

    hidden_states = (
        hidden_states[:, :, None, :, :]
        .expand(B, H, n_rep, T, D)
    )

    return hidden_states.reshape(
        B,
        H * n_rep,
        T,
        D
    )


def attention_forward(
    query_states,
    key_states,
    value_states,
    num_key_value_groups,
):
    """
    Qwen3 eager attention.

    query:
        [B, num_heads, T, D]

    key/value:
        [B, num_kv_heads, T, D]
    """

    key_states = repeat_kv(
        key_states,
        num_key_value_groups
    )

    value_states = repeat_kv(
        value_states,
        num_key_value_groups
    )
    
    # jia
    query_states = query_states.to(key_states.dtype)
    value_states = value_states.to(key_states.dtype)
    

    head_dim = query_states.shape[-1]

    scaling = head_dim ** -0.5

    # --------------------------------------------------------
    # Attention scores
    # --------------------------------------------------------

    attn_weights = torch.matmul(
        query_states,
        key_states.transpose(-2, -1)
    )

    attn_weights = attn_weights * scaling

    # --------------------------------------------------------
    # Causal mask
    # --------------------------------------------------------

    Tq = query_states.shape[-2]
    Tk = key_states.shape[-2]

    causal_mask = torch.triu(
        torch.ones(
            Tq,
            Tk,
            device=query_states.device,
            dtype=torch.bool
        ),
        diagonal=1
    )

    attn_weights = attn_weights.masked_fill(
        causal_mask,
        torch.finfo(attn_weights.dtype).min
    )

    # --------------------------------------------------------
    # Softmax
    # --------------------------------------------------------

    attn_weights = F.softmax(
        attn_weights,
        dim=-1,
        dtype=torch.float32
    ).to(query_states.dtype)

    # --------------------------------------------------------
    # Attention output
    # --------------------------------------------------------

    attn_output = torch.matmul(
        attn_weights,
        value_states
    )

    # [B, H, T, D]

    return attn_output, attn_weights


# ============================================================
# Get Q from Qwen3
# ============================================================

@torch.no_grad()
def get_q_from_hidden(
    model,
    hidden_states,
    layer_id,
    position_ids,
):
    """
    Reproduce Qwen3 attention's Q computation.

    Qwen3:

        q_proj
          ↓
        reshape
          ↓
        q_norm
          ↓
        transpose
          ↓
        RoPE
    """

    layer = model.model.layers[layer_id]

    attn = layer.self_attn

    B, T, _ = hidden_states.shape

    head_dim = model.config.head_dim
    num_heads = model.config.num_attention_heads

    # --------------------------------------------------------
    # q_proj
    # --------------------------------------------------------

    q = attn.q_proj(hidden_states)

    # [B, T, H*D]
    q = q.view(
        B,
        T,
        num_heads,
        head_dim
    )

    # --------------------------------------------------------
    # q_norm
    # --------------------------------------------------------

    q = attn.q_norm(q)

    # [B, T, H, D]
    # -> [B, H, T, D]

    q = q.transpose(1, 2)

    # --------------------------------------------------------
    # RoPE
    # --------------------------------------------------------

    # Qwen3 model-level rotary embedding
    position_embeddings = model.model.rotary_emb(
        hidden_states,
        position_ids
    )

    cos, sin = position_embeddings

    q = apply_rotary(
        q,
        cos,
        sin
    )

    return q


# ============================================================
# RoPE
# ============================================================

def rotate_half(x):
    """
    Qwen3 / Llama style RoPE.
    """

    x1 = x[..., : x.shape[-1] // 2]

    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat(
        (-x2, x1),
        dim=-1
    )


def apply_rotary(q, cos, sin):
    """
    q:
        [B, H, T, D]

    cos/sin:
        [B, T, D]
    """

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_embed = (
        q * cos
        +
        rotate_half(q) * sin
    )

    return q_embed


# ============================================================
# Main
# ============================================================

@torch.no_grad()
def main():

    print("=" * 70)
    print("Qwen3 Cross-Model KV Attention Transfer")
    print("=" * 70)

    # ========================================================
    # 1. Load KV
    # ========================================================

    print("\nLoading KV...")

    K_A = torch.load(
        K_A_PATH,
        map_location="cpu"
    ).float()

    V_A = torch.load(
        V_A_PATH,
        map_location="cpu"
    ).float()

    K_B = torch.load(
        K_B_PATH,
        map_location="cpu"
    ).float()

    V_B = torch.load(
        V_B_PATH,
        map_location="cpu"
    ).float()

    print("K_A:", K_A.shape)
    print("V_A:", V_A.shape)
    print("K_B:", K_B.shape)
    print("V_B:", V_B.shape)

    assert K_A.shape == K_B.shape
    assert V_A.shape == V_B.shape

    B, H_kv, T, D = K_B.shape

    # ========================================================
    # 2. Load input_ids
    # ========================================================

    input_ids = torch.load(
        INPUT_IDS_PATH,
        map_location="cpu"
    )

    input_ids = input_ids[:B]

    print("input_ids:", input_ids.shape)

    # ========================================================
    # 3. Load target model
    # ========================================================

    print("\nLoading Qwen3-1.7B...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_B
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_B,
        dtype=DTYPE,
        device_map="auto"
    )

    model.eval()

    # --------------------------------------------------------
    # Make sure target model is actually Qwen3
    # --------------------------------------------------------

    config = model.config

    print("\nModel config:")
    print("hidden_size:", config.hidden_size)
    print("num_attention_heads:",
          config.num_attention_heads)
    print("num_key_value_heads:",
          config.num_key_value_heads)
    print("head_dim:",
          getattr(
              config,
              "head_dim",
              config.hidden_size // config.num_attention_heads
          ))

    assert config.num_key_value_heads == H_kv
    assert config.head_dim == D

    # ========================================================
    # 4. Get hidden states BEFORE target layer
    # ========================================================

    print("\nRunning target model...")

    input_ids_device = input_ids.to(
        model.device
    )

    outputs = model(
        input_ids=input_ids_device,
        output_hidden_states=True,
        use_cache=False
    )

    # HuggingFace:
    #
    # hidden_states[0] = embedding output
    # hidden_states[i] = input to layer i
    #
    hidden_states = outputs.hidden_states[LAYER_ID]

    print(
        "hidden_states:",
        hidden_states.shape
    )

    # ========================================================
    # 5. Qwen3 Q
    # ========================================================

    position_ids = torch.arange(
        T,
        device=hidden_states.device
    ).unsqueeze(0)

    Q = get_q_from_hidden(
        model,
        hidden_states,
        LAYER_ID,
        position_ids
    )

    print("Q:", Q.shape)

    # ========================================================
    # 6. Move K/V to GPU
    # ========================================================

    K_A = K_A.to(
        device=Q.device,
        dtype=Q.dtype
    )

    V_A = V_A.to(
        device=Q.device,
        dtype=Q.dtype
    )

    K_B = K_B.to(
        device=Q.device,
        dtype=Q.dtype
    )

    V_B = V_B.to(
        device=Q.device,
        dtype=Q.dtype
    )

    # ========================================================
    # 7. Baseline:
    #
    # 1.7B Q
    # +
    # 1.7B K/V
    #
    # ========================================================

    print("\nComputing baseline attention...")

    O_real, A_real = attention_forward(
        Q,
        K_B,
        V_B,
        config.num_attention_heads
        // config.num_key_value_heads
    )

    print(
        "O_real:",
        O_real.shape
    )

    # ========================================================
    # 8. Fit PCA on 1.7B K
    # ========================================================

    print("\nFitting target K PCA...")

    K_B_flat = K_B.reshape(
        -1,
        D
    )

    K_mean_B, K_U_B = fit_pca(
        K_B_flat,
        PCA_DIM
    )

    # ========================================================
    # 9. Fit PCA on 1.7B V
    # ========================================================

    print("Fitting target V PCA...")

    V_B_flat = V_B.reshape(
        -1,
        D
    )

    V_mean_B, V_U_B = fit_pca(
        V_B_flat,
        PCA_DIM
    )

    # ========================================================
    # 10. Project 0.6B K/V into 1.7B subspace
    # ========================================================

    print("\nProjecting 0.6B K/V into 1.7B PCA subspace...")

    K_A_flat = K_A.reshape(
        -1,
        D
    )

    V_A_flat = V_A.reshape(
        -1,
        D
    )

    K_transfer = project_to_subspace(
        K_A_flat,
        K_mean_B,
        K_U_B
    )

    V_transfer = project_to_subspace(
        V_A_flat,
        V_mean_B,
        V_U_B
    )

    K_transfer = K_transfer.reshape(
        B,
        H_kv,
        T,
        D
    )

    V_transfer = V_transfer.reshape(
        B,
        H_kv,
        T,
        D
    )

    # ========================================================
    # 11. Cross-model attention
    #
    # Q = 1.7B
    # K/V = 0.6B projected into 1.7B PCA space
    # ========================================================

    print("\nComputing cross-model attention...")

    O_transfer, A_transfer = attention_forward(
        Q,
        K_transfer,
        V_transfer,
        config.num_attention_heads
        // config.num_key_value_heads
    )

    print(
        "O_transfer:",
        O_transfer.shape
    )

    # ========================================================
    # 12. Compare attention outputs
    # ========================================================

    global_cos = cosine_similarity(
        O_real,
        O_transfer
    )

    token_cos = mean_token_cosine(
        O_real,
        O_transfer
    )

    # ========================================================
    # 13. Compare attention distributions
    # ========================================================

    attn_cos = cosine_similarity(
        A_real,
        A_transfer
    )

    # ========================================================
    # 14. K/V reconstruction cosine
    # ========================================================

    K_recon_cos = cosine_similarity(
        K_B,
        K_transfer
    )

    V_recon_cos = cosine_similarity(
        V_B,
        V_transfer
    )

    # ========================================================
    # 15. Results
    # ========================================================

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(f"PCA dimension       : {PCA_DIM}")

    print()
    print("K reconstruction")
    print(f"  cosine            : {K_recon_cos:.6f}")

    print()
    print("V reconstruction")
    print(f"  cosine            : {V_recon_cos:.6f}")

    print()
    print("Attention")
    print(f"  attention cosine  : {attn_cos:.6f}")

    print()
    print("Attention output")
    print(f"  global cosine     : {global_cos:.6f}")
    print(f"  token mean cosine : {token_cos:.6f}")

    print("=" * 70)


if __name__ == "__main__":
    main()