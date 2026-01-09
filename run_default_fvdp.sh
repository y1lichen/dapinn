#!/bin/bash

set -euo pipefail

WORKDIR="${1:-.}"
export WANDB_MODE=offline

DAPINN_DIR="results/fvdp_default/dapinn"

echo ""
echo "============================================================"
echo " DAPINN Fractional Van der Pol (α=0.9, μ=1.0)"
echo " Output: ${WORKDIR}/${DAPINN_DIR}"
echo "============================================================"

echo ">>> [Step 1/2] Pre-training..."
python -m examples.fractional_van_der_pol.main \
    --mode=train \
    --is_pretrained=True \
    --workdir="${WORKDIR}"

echo ""
echo ">>> [Step 2/2] Fine-tuning..."
python -m examples.fractional_van_der_pol.main \
    --mode=train \
    --is_pretrained=False \
    --finetune_sample_size=10 \
    --workdir="${WORKDIR}"

echo ""
echo ">>> [Step 3/3] Evaluation..."
python -m examples.fractional_van_der_pol.main \
    --mode=eval \
    --is_pretrained=False \
    --workdir="${WORKDIR}"

echo "Done!"
