#!/bin/bash

set -euo pipefail  # stop on error

# Usage: ./run_pedagogical.sh [workdir]
# If workdir not provided, default to current directory
WORKDIR="${1:-.}"

export WANDB_MODE=offline

# Define output directories to keep results separate
BASELINE_DIR="results/pedagogical_baseline_comparison/baseline"
DAPINN_DIR="results/pedagogical_baseline_comparison/dapinn"

echo "============================================================"
echo " EXPERIMENT 1: Baseline (Standard PINN)"
echo " Goal: Show failure when model assumes du/dt=f(t) only"
echo " Output: ${WORKDIR}/${BASELINE_DIR}"
echo "============================================================"

# Baseline: No corrector, No pretrain needed, Just finetune on data
python -m examples.pedagogical_baseline_comparison.main \
    --mode=train \
    --use_corrector=False \
    --run_pretrain=False \
    --run_finetune=True \
    --load_pretrained=False \
    --save_subdir="${BASELINE_DIR}" \
    --workdir="${WORKDIR}"

echo ""
echo "============================================================"
echo " EXPERIMENT 2: DAPINN (With Corrector)"
echo " Goal: Discover missing reaction term lambda*u(1-u)"
echo " Output: ${WORKDIR}/${DAPINN_DIR}"
echo "============================================================"

# --- Stage 1: Pre-training ---
# Learns du/dt = f(t) without data (smooth initialization)
echo ">>> [Step 1/2] Pre-training DAPINN (Physics only)..."
python -m examples.pedagogical_baseline_comparison.main \
    --mode=train \
    --use_corrector=True \
    --run_pretrain=True \
    --run_finetune=False \
    --save_subdir="${DAPINN_DIR}" \
    --workdir="${WORKDIR}"

# --- Stage 2: Fine-tuning ---
# Loads pretrained weights, adds data, enables corrector
echo ">>> [Step 2/2] Fine-tuning DAPINN (Data + Corrector)..."
python -m examples.pedagogical_baseline_comparison.main \
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

echo ">>> Evaluating Baseline..."
python -m examples.pedagogical_baseline_comparison.main \
    --mode=eval \
    --use_corrector=False \
    --save_subdir="${BASELINE_DIR}" \
    --workdir="${WORKDIR}"

echo ">>> Evaluating DAPINN..."
python -m examples.pedagogical_baseline_comparison.main \
    --mode=eval \
    --use_corrector=True \
    --save_subdir="${DAPINN_DIR}" \
    --workdir="${WORKDIR}"
    
echo "============================================================"
echo " All experiments completed."
echo "============================================================"