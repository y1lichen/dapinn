#!/bin/bash

set -euo pipefail  # stop on error

# Usage: ./run_pedagogical.sh [workdir]
# If workdir not provided, default to current directory
WORKDIR="${1:-.}"

# export WANDB_MODE=offline

# Define output directories to keep results separate
BASELINE_DIR="cmpinns/results/pedagogical_baseline_comparison/baseline"


python -m cmpinns.examples.pedagogical_baseline_comparison.main \
    --mode=train \
    --use_corrector=True \
    --run_pretrain=False \
    --run_finetune=True \
    --load_pretrained=False \
    --save_subdir="${BASELINE_DIR}" \
    --workdir="${WORKDIR}"


echo ">>> Evaluating Baseline..."
python -m cmpinns.examples.pedagogical_baseline_comparison.main \
    --mode=eval \
    --use_corrector=True \
    --save_subdir="${BASELINE_DIR}" \
    --workdir="${WORKDIR}"

    
echo "============================================================"
echo " All experiments completed."
echo "============================================================"