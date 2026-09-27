import torch
import PCA

if __name__ == "__main__":

    # --------------------------------------------------------
    # 改成你实际保存 K 的路径
    # --------------------------------------------------------

    K_06B = torch.load("./kv_cache/K_0.6B.pt")
    K_17B = torch.load("./kv_cache/K_1.7B.pt")
    K_4B  = torch.load("./kv_cache/K_4B.pt")

    print("Shapes:")
    print("0.6B:", K_06B.shape)
    print("1.7B:", K_17B.shape)
    print("4B  :", K_4B.shape)

    k = 60

    # --------------------------------------------------------
    # 0.6B vs 1.7B
    # --------------------------------------------------------

    S_06_17 = PCA.build_similarity_matrix(
        K_06B,
        K_17B,
        k=k
    )

    PCA.print_matrix(
        S_06_17,
        "Qwen3-0.6B vs Qwen3-1.7B"
    )

    PCA.print_best_matches(
        S_06_17,
        "Qwen3-0.6B -> Qwen3-1.7B"
    )

    # --------------------------------------------------------
    # 0.6B vs 4B
    # --------------------------------------------------------

    S_06_4 = PCA.build_similarity_matrix(
        K_06B,
        K_4B,
        k=k
    )

    PCA.print_matrix(
        S_06_4,
        "Qwen3-0.6B vs Qwen3-4B"
    )

    PCA.print_best_matches(
        S_06_4,
        "Qwen3-0.6B -> Qwen3-4B"
    )

    # --------------------------------------------------------
    # 1.7B vs 4B
    # --------------------------------------------------------

    S_17_4 = PCA.build_similarity_matrix(
        K_17B,
        K_4B,
        k=k
    )

    PCA.print_matrix(
        S_17_4,
        "Qwen3-1.7B vs Qwen3-4B"
    )

    PCA.print_best_matches(
        S_17_4,
        "Qwen3-1.7B -> Qwen3-4B"
    )

    # --------------------------------------------------------
    # save
    # --------------------------------------------------------

    torch.save(
        {
            "06B_17B": S_06_17,
            "06B_4B": S_06_4,
            "17B_4B": S_17_4,
        },
        "head_subspace_similarity.pt"
    )

    print()
    print("Saved to head_subspace_similarity.pt")