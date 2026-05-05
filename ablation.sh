#!/usr/bin/env bash
##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## ablation
##

set -euo pipefail

echo "============================================"
echo "  Start Trek | ablation study"
echo "============================================"

# --- Baseline DQN (reference) ---

echo ""
echo "[1/8] Training DQN baseline..."
python train.py --policy dqn --config config/dqn.yaml

echo ""
echo "[2/8] Evaluating DQN baseline..."
python eval.py --policy dqn --episodes 100 --config config/dqn.yaml

# --- Ablation: buffer size ---

echo ""
echo "[3/8] Training DQN ablation: buffer_size=10k..."
python train.py --policy dqn --config config/ablation_buffer.yaml

echo ""
echo "[4/8] Evaluating DQN ablation: buffer_size=10k..."
python eval.py --policy dqn --episodes 100 --config config/ablation_buffer.yaml

# --- Ablation: soft update ---

echo ""
echo "[5/8] Training DQN ablation: soft update..."
python train.py --policy dqn --config config/ablation_soft_update.yaml

echo ""
echo "[6/8] Evaluating DQN ablation: soft update..."
python eval.py --policy dqn --episodes 100 --config config/ablation_soft_update.yaml

# --- Ablation: obs normalization ---

echo ""
echo "[7/8] Training DQN ablation: obs normalization..."
python train.py --policy dqn --config config/ablation_obs_norm.yaml

echo ""
echo "[8/8] Evaluating DQN ablation: obs normalization..."
python eval.py --policy dqn --episodes 100 --config config/ablation_obs_norm.yaml

echo ""
echo "Done. Run 'python plot.py' to generate comparison plots."