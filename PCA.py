import torch

def fit_pca_basis(K, k):
    K = K.float()

    mean = K.mean(dim=0, keepdim=True)
    X = K - mean

    cov = X.T @ X / (X.shape[0] - 1)

    eigvals, eigvecs = torch.linalg.eigh(cov)

    idx = torch.argsort(eigvals, descending=True)
    eigvecs = eigvecs[:, idx]

    U = eigvecs[:, :k]

    return mean, U


def explained_variance(K_test, mean, U):
    K_test = K_test.float()

    X = K_test - mean

    Z = X @ U
    X_hat = Z @ U.T

    total = torch.sum(X ** 2)
    residual = torch.sum((X - X_hat) ** 2)

    return (1 - residual / total).item()


# =========================
# Self-PCA
# =========================
