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
echo "[1/4] Training baselines..."
python train.py --policy random --episodes 100 --config config/default.yaml
python train.py --policy heuristic --episodes 100 --config config/default.yaml

echo ""
echo "[2/4] Evaluating baselines..."
python eval.py --policy random --episodes 100 --config config/default.yaml
python eval.py --policy heuristic --episodes 100 --config config/default.yaml

# --- DQN ---

echo ""
echo "[3/4] Training DQN..."
python train.py --policy dqn --config config/dqn.yaml

echo ""
echo "[3/4] Evaluating DQN..."
python eval.py --policy dqn --episodes 100 --config config/dqn.yaml

# --- Plots ---

echo ""
echo "[4/4] Generating plots..."
python plot.py --out assets/plots

echo ""
echo "Done. Plots saved to assets/plots/"