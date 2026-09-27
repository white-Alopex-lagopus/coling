import json
import numpy as np

with open("budget_utility_curves.json", "r", encoding="utf-8") as f:
    rows = json.load(f)

U = {}
for r in rows:
    U.setdefault(r["layer"], {})[r["budget"]] = r["cosine"]

layers = sorted(U.keys())
budgets = sorted(U[layers[0]].keys())

print("=" * 80)
print("Monotonicity check")
print("=" * 80)
print(f"{'Layer':>5}  {'Status':>16}  {'Min budget cos':>16}  {'Max budget cos':>16}  {'Range':>8}")
print("-" * 80)

non_mono = []
weak_mono = []

for l in layers:
    cos_list = [U[l][b] for b in budgets]

    # 严格单调递增：cos[i] <= cos[i+1]，允许相等
    is_mono = all(cos_list[i] <= cos_list[i+1] for i in range(len(cos_list) - 1))

    # 严格递增：cos[i] < cos[i+1]
    is_strict = all(cos_list[i] < cos_list[i+1] for i in range(len(cos_list) - 1))

    # 找违反单调的位置
    violations = []
    for i in range(len(cos_list) - 1):
        if cos_list[i] > cos_list[i+1]:
            violations.append(
                f"{budgets[i]}->{budgets[i+1]}: "
                f"{cos_list[i]:.4f}->{cos_list[i+1]:.4f}"
            )

    if is_strict:
        status = "STRICT_MONO"
    elif is_mono:
        status = "WEAK_MONO"
        weak_mono.append(l)
    else:
        status = "NON_MONO"
        non_mono.append((l, violations))

    rng = max(cos_list) - min(cos_list)

    print(f"{l:5d}  {status:>16}  {cos_list[0]:16.4f}  {cos_list[-1]:16.4f}  {rng:8.4f}")

    if not is_mono:
        for v in violations:
            print(f"         ⚠️  {v}")

print()
print("=" * 80)
print("Summary")
print("=" * 80)
print(f"Total layers        : {len(layers)}")
print(f"Strict monotonic    : {len(layers) - len(weak_mono) - len(non_mono)}")
print(f"Weak monotonic      : {len(weak_mono)}  {weak_mono}")
print(f"Non-monotonic       : {len(non_mono)}  {[x[0] for x in non_mono]}")

# 额外：看 range 分布，判断信噪比
print()
print("=" * 80)
print("Curve range distribution (max - min cosine across budgets)")
print("=" * 80)
ranges = []
for l in layers:
    cos_list = [U[l][b] for b in budgets]
    ranges.append(max(cos_list) - min(cos_list))

ranges = np.array(ranges)
print(f"Mean range   : {ranges.mean():.4f}")
print(f"Median range : {np.median(ranges):.4f}")
print(f"Min range    : {ranges.min():.4f}  (layer {layers[int(np.argmin(ranges))]})")
print(f"Max range    : {ranges.max():.4f}  (layer {layers[int(np.argmax(ranges))]})")

# 如果 range 很小，说明这条曲线几乎没变化，Oracle 分配没意义
print()
print("Layers with range < 0.05 (nearly flat, 分配无意义):")
flat = [l for l, r in zip(layers, ranges) if r < 0.05]
print(f"  {flat}")