#!/usr/bin/env bash
##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## one-click
##

# Usage: bash one-click.sh
# one-click.sh => Regenerate all logs, CSVs and plots from scratch.

set -euo pipefail

echo "============================================"
echo "  Start Trek | full reproduction run"
echo "============================================"

# --- Baselines ---

echo ""
echo "[1/5] Training baselines..."
python train.py --policy random    --episodes 100 --config config/default.yaml --video
python train.py --policy heuristic --episodes 100 --config config/default.yaml --video

echo ""
echo "[2/5] Evaluating baselines..."
python eval.py --policy random    --episodes 100 --config config/default.yaml --video
python eval.py --policy heuristic --episodes 100 --config config/default.yaml --video

# --- DQN ---

echo ""
echo "[3/5] Training DQN..."
python train.py --policy dqn --config config/dqn.yaml --video

echo ""
echo "[4/5] Evaluating DQN..."
python eval.py --policy dqn --episodes 100 --config config/dqn.yaml --video

# --- Double DQN ---
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

# --- Plots ---

echo ""
echo "[5/5] Generating plots..."
python plot.py --out assets/plots

echo ""
echo "Done. Plots saved to assets/plots/"