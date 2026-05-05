#!/usr/bin/env bash
##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## seeds - multi-seed reproducibility run (mean +/- 95% CI)
##

set -euo pipefail

SEEDS=(0 1 2 3 4)

echo "============================================"
echo "  Start Trek | multi-seed run (5 seeds)"
echo "============================================"

for s in "${SEEDS[@]}"; do
  echo ""
  echo "=================== seed $s ==================="

  echo "[random]    train + eval"
  python train.py --policy random    --config config/default.yaml --episodes 100 --seed "$s"
  python eval.py  --policy random    --config config/default.yaml --episodes 100 --seed "$s"

  echo "[heuristic] train + eval"
  python train.py --policy heuristic --config config/default.yaml --episodes 100 --seed "$s"
  python eval.py  --policy heuristic --config config/default.yaml --episodes 100 --seed "$s"

  echo "[dqn]       train + eval"
  python train.py --policy dqn       --config config/dqn.yaml                    --seed "$s"
  python eval.py  --policy dqn       --config config/dqn.yaml     --episodes 100 --seed "$s"
done

echo ""
echo "Aggregating per-seed metrics..."
python aggregate.py

echo ""
echo "Done. Summary written to logs/seeds_summary.csv"
