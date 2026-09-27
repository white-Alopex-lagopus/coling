import numpy as np
import matplotlib.pyplot as plt

cka = np.load("cka_matrix_0.6B_vs_1.7B.npy")

plt.figure(figsize=(8, 5))
plt.hist(cka.flatten(), bins=50)
plt.xlabel("CKA")
plt.ylabel("Count")
plt.title("CKA distribution")
plt.axvline(cka.mean(), color='r', linestyle='--', label=f"mean={cka.mean():.3f}")
plt.legend()
plt.tight_layout()
plt.savefig("cka_hist.png", dpi=150)