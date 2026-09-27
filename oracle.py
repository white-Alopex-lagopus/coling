import json
import numpy as np

# ============================================================
# 1. 读数据
# ============================================================
with open("budget_utility_curves.json", "r", encoding="utf-8") as f:
    rows = json.load(f)

# rows = [{"layer": 0, "budget": 16, "cosine": 0.31, "rel_l2": 0.8}, ...]

# 整理成 U[layer][budget] = cosine
U = {}
for r in rows:
    U.setdefault(r["layer"], {})[r["budget"]] = r["cosine"]

layers = sorted(U.keys())
budgets = sorted(U[layers[0]].keys())
L = len(layers)
B_avg = 64
target_sum = L * B_avg          # 28 * 64 = 1792

print(f"Layers: {L}, Budgets: {budgets}, Target sum: {target_sum}")


# ============================================================
# 2. DP 求 Oracle 最优分配
#    dp[l][s] = 前 l 层，总 budget = s 时的最大 sum cosine
# ============================================================
INF_NEG = -1e18
max_sum = max(budgets) * L

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

# 回溯
oracle_budget = {}
s = target_sum
for l in range(L, 0, -1):
    b = choice[l][s]
    oracle_budget[layers[l - 1]] = b
    s -= b

oracle_cos = dp[L][target_sum] / L


# ============================================================
# 3. Uniform 64
# ============================================================
uniform_budget = {l: 64 for l in layers}
uniform_cos = np.mean([U[l][64] for l in layers])


# ============================================================
# 4. 输出对比
# ============================================================
print()
print("=" * 70)
print(f"Uniform 64 mean cosine : {uniform_cos:.6f}")
print(f"Oracle    mean cosine  : {oracle_cos:.6f}")
print(f"Gain (Oracle - Uniform): {oracle_cos - uniform_cos:+.6f}")
print("=" * 70)

print()
print("Layer  Uniform  Oracle   Δ")
print("-" * 40)
for l in layers:
    d = U[l][oracle_budget[l]] - U[l][64]
    print(f"{l:5d}  {64:7d}  {oracle_budget[l]:6d}  {d:+.4f}")

# 平均 budget 验证
print()
print(f"Oracle avg budget: {sum(oracle_budget.values()) / L:.2f}")