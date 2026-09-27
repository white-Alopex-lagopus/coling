import torch

from PCA import fit_pca_basis, explained_variance

def loo_headwise(K1, K2, Ktest, k=60):

    # K: [50, 8, 512, 128]

    results = []

    for head in range(8):

        K1_h = K1[:, head, :, :].reshape(-1, 128)
        K2_h = K2[:, head, :, :].reshape(-1, 128)
        Kt_h = Ktest[:, head, :, :].reshape(-1, 128)

        # train
        K_train = torch.cat([K1_h, K2_h], dim=0)

        mean, U = fit_pca_basis(K_train, k)

        # test
        ev = explained_variance(Kt_h, mean, U)

        results.append(ev)

        print(f"head {head}: {ev:.4f}")

    print(f"mean: {sum(results)/len(results):.4f}")

    return results
