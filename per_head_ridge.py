import torch


def ridge_regression(X, Y, lam=1e-3):
    """
    X: [N, d]
    Y: [N, d]

    return:
        W: [d, d]
    """
    X = X.float()
    Y = Y.float()

    d = X.shape[1]

    I = torch.eye(d, device=X.device)

    A = X.T @ X + lam * I
    B = X.T @ Y

    W = torch.linalg.solve(A, B)

    return W


def r2_score(Y, Y_hat):
    """
    全维度 R² / explained variance
    """
    Y = Y.float()
    Y_hat = Y_hat.float()

    mean = Y.mean(dim=0, keepdim=True)

    ss_total = torch.sum((Y - mean) ** 2)
    ss_res = torch.sum((Y - Y_hat) ** 2)

    return (1 - ss_res / ss_total).item()


def main():

    # ============================================================
    # 1. Load KV
    # ============================================================

    K_A = torch.load("./kv_cache/K_0.6B.pt", map_location="cpu").float()
    K_B = torch.load("./kv_cache/K_1.7B.pt", map_location="cpu").float()

    print("K_A:", K_A.shape)
    print("K_B:", K_B.shape)

    # 应该是：
    # [50, 8, 512, 128]

    assert K_A.ndim == 4
    assert K_B.ndim == 4

    assert K_A.shape == K_B.shape

    num_samples, num_heads, seq_len, head_dim = K_A.shape

    print()
    print("num_samples:", num_samples)
    print("num_heads:", num_heads)
    print("seq_len:", seq_len)
    print("head_dim:", head_dim)

    # ============================================================
    # 2. Per-head Ridge
    # ============================================================

    lam = 1e-3

    results = []

    print()
    print("=" * 70)
    print("Per-head Ridge")
    print("=" * 70)
    print("lambda =", lam)

    for h in range(num_heads):

        # --------------------------------------------------------
        # 取出一个 head
        #
        # [50, 512, 128]
        # --------------------------------------------------------

        X = K_A[:, h, :, :].reshape(-1, head_dim)
        Y = K_B[:, h, :, :].reshape(-1, head_dim)

        # --------------------------------------------------------
        # Center
        # --------------------------------------------------------

        mean_X = X.mean(dim=0, keepdim=True)
        mean_Y = Y.mean(dim=0, keepdim=True)

        X_centered = X - mean_X
        Y_centered = Y - mean_Y

        # --------------------------------------------------------
        # Ridge
        # --------------------------------------------------------

        W = ridge_regression(
            X_centered,
            Y_centered,
            lam=lam
        )

        # --------------------------------------------------------
        # Prediction
        # --------------------------------------------------------

        Y_hat = X_centered @ W + mean_Y

        # --------------------------------------------------------
        # R²
        # --------------------------------------------------------

        score = r2_score(Y, Y_hat)

        results.append(score)

        print(f"head {h}: EV = {score:.6f}")

    # ============================================================
    # 3. Mean
    # ============================================================

    mean_score = sum(results) / len(results)

    print()
    print("-" * 70)
    print(f"Mean EV = {mean_score:.6f}")
    print("-" * 70)


if __name__ == "__main__":
    main()