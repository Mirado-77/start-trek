#!/usr/bin/env bash
##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## bonus seeds - Double DQN over 5 seeds for a fair comparison vs plain DQN
##

set -euo pipefail

# Same convention as the main seeds.sh: no --video on multi-seed runs.
SEEDS=(0 1 2 3 4)

mkdir -p logs/scripts

echo "============================================"
echo "  Start Trek | bonus: Double DQN, 5 seeds"
echo "============================================"

for s in "${SEEDS[@]}"; do
  echo ""
  echo "=================== seed $s ==================="

  echo "[ddqn] train + eval"
  python train.py --policy dqn --config bonus/configs/ddqn.yaml --seed "$s"
  python eval.py --policy dqn --config bonus/configs/ddqn.yaml --episodes 100 --seed "$s"
done

echo ""
echo "Aggregating per-seed metrics..."
python aggregate.py --runs ddqn --out logs/seeds_summary_ddqn.csv

echo ""
echo "Done. Summary written to logs/seeds_summary_ddqn.csv"
