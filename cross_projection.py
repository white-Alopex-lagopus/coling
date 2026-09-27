import torch


def fit_pca(X, k=60):

    X = X.float()

    mean = X.mean(dim=0, keepdim=True)
    X_centered = X - mean

    cov = X_centered.T @ X_centered / (X_centered.shape[0] - 1)

    eigvals, eigvecs = torch.linalg.eigh(cov)

    idx = torch.argsort(eigvals, descending=True)

    U = eigvecs[:, idx[:k]]

    return mean, U


def explained_variance(X, mean, U):

    X = X.float()

    X_centered = X - mean

    Z = X_centered @ U

    X_hat = Z @ U.T

    total = torch.sum(X_centered ** 2)

    residual = torch.sum(
        (X_centered - X_hat) ** 2
    )

    return (1 - residual / total).item()


K_06B = torch.load("./kv_cache/K_0.6B.pt").float()
K_17B = torch.load("./kv_cache/K_1.7B.pt").float()

X_A = K_06B.reshape(-1, 128)
X_B = K_17B.reshape(-1, 128)

k = 60

mean_A, U_A = fit_pca(X_A, k)
mean_B, U_B = fit_pca(X_B, k)


# ------------------------------------------------------------
# A data -> B subspace
# ------------------------------------------------------------

ev_A_in_B = explained_variance(
    X_A,
    mean_B,
    U_B
)


# ------------------------------------------------------------
# B data -> A subspace
# ------------------------------------------------------------

ev_B_in_A = explained_variance(
    X_B,
    mean_A,
    U_A
)


print("=" * 70)
print("Cross-model subspace projection")
print("=" * 70)

print(f"0.6B self PCA:       "
      f"{explained_variance(X_A, mean_A, U_A):.6f}")

print(f"1.7B self PCA:       "
      f"{explained_variance(X_B, mean_B, U_B):.6f}")

print()

print(f"0.6B in 1.7B space:  "
      f"{ev_A_in_B:.6f}")

print(f"1.7B in 0.6B space:  "
      f"{ev_B_in_A:.6f}")