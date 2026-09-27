import torch


# ============================================================
# Configuration
# ============================================================

K_A_PATH = "./kv_cache/K_0.6B.pt"
K_B_PATH = "./kv_cache/K_1.7B.pt"

LAMBDA = 1.0

TRAIN_RATIO = 0.8


# ============================================================
# Load KV
# ============================================================

K_A = torch.load(K_A_PATH).float()
K_B = torch.load(K_B_PATH).float()

print("K_A:", K_A.shape)
print("K_B:", K_B.shape)


# ============================================================
# Flatten
#
# [batch, kv_heads, seq_len, head_dim]
# -> [N, head_dim]
# ============================================================

X = K_A.reshape(-1, K_A.shape[-1])
Y = K_B.reshape(-1, K_B.shape[-1])

print("X:", X.shape)
print("Y:", Y.shape)


# ============================================================
# Train / Test split
# ============================================================

N = X.shape[0]

generator = torch.Generator()
generator.manual_seed(42)

perm = torch.randperm(N, generator=generator)

train_size = int(N * TRAIN_RATIO)

train_idx = perm[:train_size]
test_idx = perm[train_size:]

X_train = X[train_idx]
Y_train = Y[train_idx]

X_test = X[test_idx]
Y_test = Y[test_idx]

print()
print("Train:", X_train.shape)
print("Test :", X_test.shape)


# ============================================================
# Normalize / center
# ============================================================

mean_X = X_train.mean(dim=0, keepdim=True)
mean_Y = Y_train.mean(dim=0, keepdim=True)

X_train_c = X_train - mean_X
Y_train_c = Y_train - mean_Y

X_test_c = X_test - mean_X


# ============================================================
# Ridge
#
# W = (X^T X + lambda I)^(-1) X^T Y
# ============================================================

d = X_train_c.shape[1]

I = torch.eye(d, dtype=X_train_c.dtype)

A = X_train_c.T @ X_train_c

B = X_train_c.T @ Y_train_c

W = torch.linalg.solve(
    A + LAMBDA * I,
    B
)

print()
print("W:", W.shape)


# ============================================================
# Prediction
# ============================================================

Y_pred = X_test_c @ W + mean_Y


# ============================================================
# Metrics
# ============================================================

mse = torch.mean(
    (Y_test - Y_pred) ** 2
).item()

ss_res = torch.sum(
    (Y_test - Y_pred) ** 2
)

mean_test = Y_test.mean(dim=0, keepdim=True)

ss_tot = torch.sum(
    (Y_test - mean_test) ** 2
)

ev = (1.0 - ss_res / ss_tot).item()


print()
print("=" * 70)
print("Ridge result")
print("=" * 70)

print(f"lambda = {LAMBDA}")
print(f"MSE    = {mse:.6f}")
print(f"EV     = {ev:.6f}")