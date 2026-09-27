import torch

def random_orthogonal_basis(d=128, k=60):
    X = torch.randn(d, k)
    Q, R = torch.linalg.qr(X)
    return Q[:, :k]


def subspace_similarity(U1, U2):
    M = U1.T @ U2
    return torch.sum(M ** 2).item() / U1.shape[1]


values = []

for _ in range(1000):

    U1 = random_orthogonal_basis(128, 60)
    U2 = random_orthogonal_basis(128, 60)

    s = subspace_similarity(U1, U2)

    values.append(s)


values = torch.tensor(values)

print("Random baseline")
print("-------------------------")
print(f"mean = {values.mean():.4f}")
print(f"std  = {values.std():.4f}")
print(f"min  = {values.min():.4f}")
print(f"max  = {values.max():.4f}")

for q in [0.01, 0.05, 0.50, 0.95, 0.99]:

    print(
        f"q{q:.2f} = "
        f"{torch.quantile(values, q):.4f}"
    )