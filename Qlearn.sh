##
## EPITECH PROJECT, 2026
## G-AIA-401-PAR-4-1-starttrek-12
## File description:
## Qlearn
##

set -euo pipefail

echo "============================================"
echo "  Start Trek | ablation study"
echo "============================================"

# --- Baseline DQN (reference) ---

echo ""
echo "[1/8] Training DQN baseline..."
python train.py --policy qlearn --config config/qlearn.yaml

echo ""
echo "[2/8] Evaluating DQN baseline..."
python eval.py --policy qlearn --episodes 100 --config config/qlearn.yaml 

