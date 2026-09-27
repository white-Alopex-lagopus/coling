import torch
import numpy as np

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

def get_head_pca_basis(K_head, k=60):
    """
    K_head:
        [N, seq_len, head_dim]
        e.g. [50, 512, 128]

    return:
        U: [128, k]
    """

    X = K_head.reshape(-1, K_head.shape[-1]).float()
    
    # center
    mean = X.mean(dim=0, keepdim=True)
    X = X - mean

    # covariance
    cov = X.T @ X / (X.shape[0] - 1)

    # eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(cov)

    # descending
    idx = torch.argsort(eigvals, descending=True)
    eigvecs = eigvecs[:, idx]

    U = eigvecs[:, :k]

    return U
    
def subspace_similarity(U1, U2):
    """
    U1, U2:
        [128, k]

    Returns:
        scalar in [0, 1]
    """

    M = U1.T @ U2

    similarity = torch.sum(M ** 2) / U1.shape[1]

    return similarity.item()


def build_similarity_matrix(K_A, K_B, k=60):

    num_heads = K_A.shape[1]

    bases_A = []
    bases_B = []

    print("Computing PCA bases...")

    for h in range(num_heads):

        print(f"  model A head {h}")

        U_A = get_head_pca_basis(
            K_A[:, h, :, :],
            k=k
        )

        bases_A.append(U_A)

    for h in range(num_heads):

        print(f"  model B head {h}")

        U_B = get_head_pca_basis(
            K_B[:, h, :, :],
            k=k
        )

        bases_B.append(U_B)

    # similarity matrix
    S = torch.zeros(num_heads, num_heads)

    for i in range(num_heads):
        for j in range(num_heads):

            S[i, j] = subspace_similarity(
                bases_A[i],
                bases_B[j]
            )

    return S

def print_matrix(S, name):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print("       " + " ".join([f"h{j:>6}" for j in range(S.shape[1])]))

    for i in range(S.shape[0]):

        values = " ".join(
            [f"{S[i,j]:7.4f}" for j in range(S.shape[1])]
        )

        print(f"h{i}:   {values}")
        
def print_best_matches(S, name):

    print()
    print("=" * 70)
    print(f"Best head matching: {name}")
    print("=" * 70)

    for i in range(S.shape[0]):

        j = torch.argmax(S[i]).item()
        value = S[i, j].item()

        diagonal = S[i, i].item()

        print(
            f"A head {i} -> B head {j} "
            f"| similarity={value:.4f} "
            f"| diagonal={diagonal:.4f}"
        )