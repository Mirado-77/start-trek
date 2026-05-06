#!/usr/bin/env bash
##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## bonus run - Double DQN, with and without wind
##

set -euo pipefail

mkdir -p logs/scripts

echo "============================================"
echo "  Start Trek | bonus: Double DQN + wind"
echo "============================================"

echo ""
echo "[1/4] Training Double DQN (no wind)..."
python train.py --policy dqn --config bonus/configs/ddqn.yaml

echo ""
echo "[2/4] Evaluating Double DQN (no wind)..."
python eval.py --policy dqn --config bonus/configs/ddqn.yaml --episodes 100

echo ""
echo "[3/4] Training Double DQN (with wind)..."
python train.py --policy dqn --config bonus/configs/ddqn_wind.yaml

echo ""
echo "[4/4] Evaluating Double DQN (with wind)..."
python eval.py --policy dqn --config bonus/configs/ddqn_wind.yaml --episodes 100

echo ""
echo "Done. CSVs in logs/, models in models/."
