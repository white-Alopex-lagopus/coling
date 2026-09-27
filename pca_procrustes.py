import torch


# ============================================================
# 1. Fit PCA
# ============================================================

def fit_pca(K, k=60):
    """
    K: [N, seq_len, head_dim]
       or [N_samples, 128]

    return:
        mean: [1, 128]
        U:    [128, k]
    """

    if K.dim() == 3:
        X = K.reshape(-1, K.shape[-1]).float()
    else:
        X = K.float()

    mean = X.mean(dim=0, keepdim=True)
    X_centered = X - mean

    # covariance
    cov = X_centered.T @ X_centered / (X_centered.shape[0] - 1)

    eigvals, eigvecs = torch.linalg.eigh(cov)

    # descending
    idx = torch.argsort(eigvals, descending=True)
    eigvecs = eigvecs[:, idx]

    U = eigvecs[:, :k]

    return mean, U


# ============================================================
# 2. PCA encode / decode
# ============================================================

def encode(X, mean, U):
    X = X.float()

    if X.dim() == 3:
        shape = X.shape
        X = X.reshape(-1, X.shape[-1])

        Z = (X - mean) @ U

        return Z, shape

    return (X - mean) @ U


def decode(Z, mean, U):
    return Z @ U.T + mean


# ============================================================
# 3. Explained variance
# ============================================================

def explained_variance(X, X_hat):
    X = X.float()

    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])

    if X_hat.dim() == 3:
        X_hat = X_hat.reshape(-1, X_hat.shape[-1])

    mean = X.mean(dim=0, keepdim=True)

    total = torch.sum((X - mean) ** 2)
    residual = torch.sum((X - X_hat) ** 2)

    return (1.0 - residual / total).item()


# ============================================================
# 4. Orthogonal Procrustes
# ============================================================

def procrustes_rotation(U_A, U_B):
    """
    U_A, U_B:
        [128, k]

    Find orthogonal R such that:

        U_A R ~= U_B

    Returns:
        R: [k, k]
    """

    M = U_A.T @ U_B

    # SVD
    P, _, Qt = torch.linalg.svd(M)

    R = P @ Qt

    return R


# ============================================================
# 5. Main experiment
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load K
    # --------------------------------------------------------

    K_06B = torch.load("./kv_cache/K_0.6B.pt").float()
    K_17B = torch.load("./kv_cache/K_1.7B.pt").float()

    print("K shapes:")
    print("0.6B:", K_06B.shape)
    print("1.7B:", K_17B.shape)

    # --------------------------------------------------------
    # For the first experiment:
    #
    # use all heads together
    #
    # [50, 8, 512, 128]
    # -> [204800, 128]
    # --------------------------------------------------------

    X_A = K_06B.reshape(-1, 128)
    X_B = K_17B.reshape(-1, 128)

    print()
    print("Flattened:")
    print("A:", X_A.shape)
    print("B:", X_B.shape)

    # --------------------------------------------------------
    # PCA dimension
    # --------------------------------------------------------

    k = 60

    print()
    print("=" * 70)
    print(f"PCA dimension k = {k}")
    print("=" * 70)

    # --------------------------------------------------------
    # Fit PCA independently
    # --------------------------------------------------------

    mean_A, U_A = fit_pca(X_A, k=k)
    mean_B, U_B = fit_pca(X_B, k=k)

    print("U_A:", U_A.shape)
    print("U_B:", U_B.shape)

    # ========================================================
    # Baseline 1:
    # Self PCA reconstruction
    # ========================================================

    Z_A = (X_A - mean_A) @ U_A
    X_A_hat = Z_A @ U_A.T + mean_A

    Z_B = (X_B - mean_B) @ U_B
    X_B_hat = Z_B @ U_B.T + mean_B

    ev_A = explained_variance(X_A, X_A_hat)
    ev_B = explained_variance(X_B, X_B_hat)

    print()
    print("Self PCA reconstruction")
    print("-----------------------")
    print(f"0.6B: {ev_A:.6f}")
    print(f"1.7B: {ev_B:.6f}")

    # ========================================================
    # Procrustes
    # ========================================================

    print()
    print("=" * 70)
    print("Procrustes alignment")
    print("=" * 70)

    R = procrustes_rotation(U_A, U_B)

    print("R shape:", R.shape)

    # ========================================================
    # A -> B
    #
    # X_A
    #   ↓
    # PCA_A
    #   ↓
    # Z_A
    #   ↓
    # Procrustes R
    #   ↓
    # Z_B-like
    #   ↓
    # PCA_B decoder
    #   ↓
    # X_B_hat
    # ========================================================

    Z_A = (X_A - mean_A) @ U_A

    Z_A_aligned = Z_A @ R

    X_B_hat_from_A = Z_A_aligned @ U_B.T + mean_B

    ev_A_to_B = explained_variance(
        X_B,
        X_B_hat_from_A
    )

    print()
    print("Cross-model reconstruction")
    print("---------------------------")
    print(f"0.6B -> 1.7B: {ev_A_to_B:.6f}")

    # ========================================================
    # Also B -> A
    # ========================================================

    R_reverse = procrustes_rotation(U_B, U_A)

    Z_B = (X_B - mean_B) @ U_B

    Z_B_aligned = Z_B @ R_reverse

    X_A_hat_from_B = Z_B_aligned @ U_A.T + mean_A

    ev_B_to_A = explained_variance(
        X_A,
        X_A_hat_from_B
    )

    print(f"1.7B -> 0.6B: {ev_B_to_A:.6f}")

    # ========================================================
    # Save
    # ========================================================

    torch.save(
        {
            "k": k,

            "mean_A": mean_A,
            "U_A": U_A,

            "mean_B": mean_B,
            "U_B": U_B,

            "R_A_to_B": R,

            "ev_A_self": ev_A,
            "ev_B_self": ev_B,

            "ev_A_to_B": ev_A_to_B,
            "ev_B_to_A": ev_B_to_A,
        },
        "pca_procrustes_result.pt"
    )

    print()
    print("Saved:")
    print("pca_procrustes_result.pt")