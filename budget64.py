import json
import numpy as np

with open("budget_utility_curves.json", "r", encoding="utf-8") as f:
    rows = json.load(f)

U = {}
for r in rows:
    U.setdefault(r["layer"], {})[r["budget"]] = r["cosine"]

layers = sorted(U.keys())

# 64 附近的边际收益
print("=" * 80)
print("Marginal utility around budget=64")
print("=" * 80)
print(f"{'Layer':>5}  {'48->64':>10}  {'64->80':>10}  {'64->96':>10}  {'64->128':>10}")
print("-" * 80)

marginal_64_80 = []
marginal_64_96 = []

for l in layers:
    d_48_64 = U[l][64] - U[l][48]
    d_64_80 = U[l][80] - U[l][64]
    d_64_96 = U[l][96] - U[l][64]
    d_64_128 = U[l][128] - U[l][64]

    marginal_64_80.append(d_64_80)
    marginal_64_96.append(d_64_96)

    print(f"{l:5d}  {d_48_64:10.4f}  {d_64_80:10.4f}  {d_64_96:10.4f}  {d_64_128:10.4f}")

print()
print("=" * 80)
print("Marginal utility spread (64->80)")
print("=" * 80)
m = np.array(marginal_64_80)
print(f"Mean   : {m.mean():.4f}")
print(f"Std    : {m.std():.4f}")
print(f"Min    : {m.min():.4f}  (layer {layers[int(np.argmin(m))]})")
print(f"Max    : {m.max():.4f}  (layer {layers[int(np.argmax(m))]})")
print(f"Spread (max-min): {m.max() - m.min():.4f}")

print()
print("=" * 80)
print("Marginal utility spread (64->96)")
print("=" * 80)
m2 = np.array(marginal_64_96)
print(f"Mean   : {m2.mean():.4f}")
print(f"Std    : {m2.std():.4f}")
print(f"Min    : {m2.min():.4f}  (layer {layers[int(np.argmin(m2))]})")
print(f"Max    : {m2.max():.4f}  (layer {layers[int(np.argmax(m2))]})")
print(f"Spread (max-min): {m2.max() - m2.min():.4f}")