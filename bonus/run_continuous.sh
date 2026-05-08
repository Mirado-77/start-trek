#!/usr/bin/env bash
##
## EPITECH PROJECT, 2026
## starttrek
## File description:
## bonus run continuous - TD3 on LunarLanderContinuous
##

set -euo pipefail

mkdir -p logs/scripts

echo "=========================================================="
echo "  Start Trek | bonus continuous: TD3 on LunarLander cont."
echo "=========================================================="

echo ""
echo "[1/2] Training TD3 (continuous control)..."
python bonus/train_td3.py --config bonus/configs/td3.yaml --video

echo ""
echo "[2/2] Evaluating TD3..."
python bonus/eval_td3.py --config bonus/configs/td3.yaml --episodes 100 --video

echo ""
echo "Done. CSVs in bonus/logs/, model in bonus/models/td3.pt, videos in bonus/assets/videos/."
