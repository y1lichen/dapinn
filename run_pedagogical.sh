#!/bin/bash

set -euo pipefail  # stop on error

# Usage: ./run_pedagogical.sh [workdir]
# If workdir not provided, default to current directory
WORKDIR="${1:-.}"

# export WANDB_MODE=offline/

# Define output directories to keep results separate
DAPINN_DIR="results/pedagogical/dapinn"

echo ""
echo "============================================================"
echo " DAPINN (With Corrector)"
echo " Goal: Discover missing reaction term lambda*u(1-u)"
echo " Output: ${WORKDIR}/${DAPINN_DIR}"
echo "============================================================"

# --- Stage 1: Pre-training ---
# Learns du/dt = f(t) without data (smooth initialization)
echo ">>> [Step 1/2] Pre-training DAPINN (Physics only)..."
python -m examples.pedagogical_example.main \
    --mode=train \
    --use_corrector=True \
    --run_pretrain=True \
    --run_finetune=False \
    --save_subdir="${DAPINN_DIR}" \
    --workdir="${WORKDIR}"

# --- Stage 2: Fine-tuning ---
# Loads pretrained weights, adds data, enables corrector
echo ">>> [Step 2/2] Fine-tuning DAPINN (Data + Corrector)..."
python -m examples.pedagogical_example.main \
    --mode=train \
    --use_corrector=True \
    --run_pretrain=False \
    --run_finetune=True \
    --load_pretrained=True \
    --save_subdir="${DAPINN_DIR}" \
    --workdir="${WORKDIR}"

echo ""
echo "============================================================"
echo " EVALUATION "
echo "============================================================"

echo ">>> Evaluating DAPINN..."
python -m examples.pedagogical_example.main \
    --mode=eval \
    --use_corrector=True \
    --save_subdir="${DAPINN_DIR}" \
    --workdir="${WORKDIR}"
    
echo "============================================================"
echo " All experiments completed."
echo "============================================================"