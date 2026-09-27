import json
import numpy as np

# ============================================================
# 1. 读数据
# ============================================================
with open("budget_utility_curves.json", "r", encoding="utf-8") as f:
    rows = json.load(f)

U = {}
for r in rows:
    U.setdefault(r["layer"], {})[r["budget"]] = r["cosine"]

layers = sorted(U.keys())
budgets = sorted(U[layers[0]].keys())
L = len(layers)
min_b = min(budgets)
max_b = max(budgets)

print(f"Layers  : {L}")
print(f"Budgets : {budgets}")


# ============================================================
# 2. DP 求 Oracle 最优分配
# ============================================================
def solve_oracle(U, layers, budgets, target_sum):
    """
    返回：
        oracle_budget: {layer: budget}
        oracle_mean_cos: 平均 cosine
    """
    L = len(layers)
    max_b = max(budgets)
    max_sum = max_b * L

    if target_sum < min(budgets) * L or target_sum > max_b * L:
        return None, None

    INF_NEG = -1e18
    dp = np.full((L + 1, max_sum + 1), INF_NEG, dtype=np.float64)
    choice = np.full((L + 1, max_sum + 1), -1, dtype=np.int32)

    dp[0][0] = 0.0

    for l in range(L):
        layer = layers[l]
        for s in range(max_sum + 1):
            if dp[l][s] == INF_NEG:
                continue
            for b in budgets:
                ns = s + b
                if ns > max_sum:
                    continue
                val = dp[l][s] + U[layer][b]
                if val > dp[l + 1][ns]:
                    dp[l + 1][ns] = val
                    choice[l + 1][ns] = b

    if dp[L][target_sum] == INF_NEG:
        return None, None

    oracle_budget = {}
    s = target_sum
    for l in range(L, 0, -1):
        b = choice[l][s]
        oracle_budget[layers[l - 1]] = int(b)
        s -= b

    oracle_mean_cos = dp[L][target_sum] / L
    return oracle_budget, oracle_mean_cos


# ============================================================
# 3. 对所有合法平均 budget 算 Oracle vs Uniform
# ============================================================
print()
print("=" * 80)
print("Oracle vs Uniform across average budgets")
print("=" * 80)
print(f"{'AvgB':>6}  {'Uniform':>10}  {'Oracle':>10}  {'Gain':>10}  {'RelGain':>10}")
print("-" * 80)

results = []

for B_avg in budgets:   # 只遍历 budgets 里真实存在的点

    target_sum = L * B_avg

    uniform_cos = np.mean([U[l][B_avg] for l in layers])

    oracle_budget, oracle_cos = solve_oracle(U, layers, budgets, target_sum)

    if oracle_cos is None:
        print(f"{B_avg:6d}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}")
        continue

    gain = oracle_cos - uniform_cos
    rel_gain = gain / (uniform_cos + 1e-8)

    print(f"{B_avg:6d}  {uniform_cos:10.4f}  {oracle_cos:10.4f}  "
          f"{gain:+10.4f}  {rel_gain:+10.2%}")

    results.append({
        "avg_budget": B_avg,
        "uniform_cos": float(uniform_cos),
        "oracle_cos": float(oracle_cos),
        "gain": float(gain),
        "rel_gain": float(rel_gain),
        "oracle_budget": oracle_budget,
    })


# ============================================================
# 4. Summary：过滤掉极端 budget
# ============================================================
print()
print("=" * 80)
print("Summary")
print("=" * 80)

valid = [
    r for r in results
    if r["avg_budget"] != min_b and r["avg_budget"] != max_b
]

if valid:
    gains = [r["gain"] for r in valid]
    best = valid[int(np.argmax(gains))]

    print(f"Max gain: {best['gain']:+.4f} at avg_budget={best['avg_budget']}")
    print(f"  Uniform: {best['uniform_cos']:.4f}")
    print(f"  Oracle : {best['oracle_cos']:.4f}")
    print()

    print("All valid results (excluding extreme budgets):")
    for r in valid:
        print(f"  avg_budget={r['avg_budget']:3d}  "
              f"gain={r['gain']:+.4f}  rel={r['rel_gain']:+.2%}")

    # 平均 gain
    mean_gain = np.mean([r["gain"] for r in valid])
    mean_rel = np.mean([r["rel_gain"] for r in valid])
    print()
    print(f"Mean gain across valid budgets: {mean_gain:+.4f}  ({mean_rel:+.2%})")
else:
    print("No valid results.")


# ============================================================
# 5. 对最优 budget，打印 Oracle 分配详情
# ============================================================
if valid:
    best = valid[int(np.argmax([r["gain"] for r in valid]))]
    B_avg = best["avg_budget"]

    print()
    print("=" * 80)
    print(f"Oracle allocation detail (avg_budget={B_avg})")
    print("=" * 80)
    print(f"{'Layer':>5}  {'Uniform':>10}  {'Oracle':>10}  {'Δcos':>10}")
    print("-" * 80)

    for l in layers:
        ob = best["oracle_budget"][l]
        d = U[l][ob] - U[l][B_avg]
        print(f"{l:5d}  {B_avg:10d}  {ob:10d}  {d:+10.4f}")

    print()
    print(f"Oracle avg budget: "
          f"{sum(best['oracle_budget'].values()) / L:.2f}  "
          f"(target {B_avg})")


# ============================================================
# 6. 存结果
# ============================================================
with open("oracle_gain_vs_budget.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print()
print("Saved: oracle_gain_vs_budget.json")