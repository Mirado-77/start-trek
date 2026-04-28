#!/usr/bin/env bash
# one-click.sh => Regenerate all logs, CSVs and plots from scratch.
# Usage: bash repro.sh
#
# To add the DQN run when ready:
#   1. Uncomment the DQN section below
#   2. Add "dqn" to POLICIES in plot.py

set -euo pipefail

echo "============================================"
echo "  Start Trek | full reproduction run"
echo "============================================"

# --- Baselines ---

echo ""
echo "[1/3] Training baselines..."
python train.py --policy random --episodes 100 --config config/default.yaml
python train.py --policy heuristic --episodes 100 --config config/default.yaml

echo ""
echo "[2/3] Evaluating baselines..."
python eval.py --policy random --episodes 100 --config config/default.yaml
python eval.py --policy heuristic --episodes 100 --config config/default.yaml

# --- DQN ---
# echo ""
# echo "[X/X] Training DQN..."
# python train.py --policy dqn --episodes 500 --config config/dqn.yaml
#
# echo "[X/X] Evaluating DQN..."
# python eval.py --policy dqn --episodes 100 --config config/dqn.yaml

# --- Plots ---

echo ""
echo "[3/3] Generating plots..."
python plot.py --out assets/plots

echo ""
echo "Done. Plots saved to assets/plots/"