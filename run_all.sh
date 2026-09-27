#!/bin/bash
set -e

# 要跑的层号列表，直接打表
LAYERS_LIST=(
  "14"
  "0, 7, 14, 21, 27"
  "0, 4, 8, 12, 15, 19, 23, 27"
  "0, 2, 5, 7, 9, 11, 14, 16, 18, 20, 23, 25, 27"
)

for L in "${LAYERS_LIST[@]}"; do
    echo ">>> 跑层: [$L]"

    # 用 sed 把 train.py 里的 LAYERS_TO_TRAIN 那一行替换掉
    sed -i "s/^LAYERS_TO_TRAIN = .*/LAYERS_TO_TRAIN = [$L]/" improve_V_PPL_mutie2e_anti_dataleak.py

    python improve_V_PPL_mutie2e_anti_dataleak.py
done