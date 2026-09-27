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
    Y = Y.float()
    Y_hat = Y_hat.float()

    mean = Y.mean(dim=0, keepdim=True)

    ss_total = torch.sum((Y - mean) ** 2)
    ss_res = torch.sum((Y - Y_hat) ** 2)

    return (1 - ss_res / ss_total).item()


if __name__ == "__main__":

    K_A = torch.load("./kv_cache/K_0.6B.pt").float()
    K_B = torch.load("./kv_cache/K_1.7B.pt").float()

    X = K_A.reshape(-1, 128)
    Y = K_B.reshape(-1, 128)

    print("X:", X.shape)
    print("Y:", Y.shape)

    # 中心化
    mean_X = X.mean(dim=0, keepdim=True)
    mean_Y = Y.mean(dim=0, keepdim=True)

    X_centered = X - mean_X
    Y_centered = Y - mean_Y

    # Ridge
    lam = 1e-3

    W = ridge_regression(
        X_centered,
        Y_centered,
        lam=lam
    )

    # prediction
    Y_hat = X_centered @ W + mean_Y

    score = r2_score(Y, Y_hat)

    print()
    print("=" * 70)
    print("Ridge baseline")
    print("=" * 70)

    print("lambda:", lam)
    print("W:", W.shape)
    print("R2:", score)