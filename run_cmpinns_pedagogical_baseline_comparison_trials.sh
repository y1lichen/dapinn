#!/bin/bash

set -euo pipefail

WORKDIR="${1:-.}"
NUM_TRIALS=30
BASE_SEED=42

echo "Starting 10 Trials Experiment..."

for ((i=1; i<=NUM_TRIALS; i++)); do
    CURRENT_SEED=$((BASE_SEED + i))
    # 為每個 Trial 建立獨立資料夾
    SAVE_SUBDIR="cmpinns/results/pedagogical_baseline_comparison/trials/trial_${i}_seed_${CURRENT_SEED}"
    
    echo "------------------------------------------------------------"
    echo "Running Trial ${i}/10 (Seed: ${CURRENT_SEED})"
    echo "Save Path: ${SAVE_SUBDIR}"
    echo "------------------------------------------------------------"

    # 執行訓練
    python -m cmpinns.examples.pedagogical_baseline_comparison.main \
        --mode=train \
        --use_corrector=True \
        --run_pretrain=False \
        --run_finetune=True \
        --load_pretrained=False \
        --seed="${CURRENT_SEED}" \
        --save_subdir="${SAVE_SUBDIR}" \
        --workdir="${WORKDIR}"

    # 執行評估
    python -m cmpinns.examples.pedagogical_baseline_comparison.main \
        --mode=eval \
        --use_corrector=True \
        --seed="${CURRENT_SEED}" \
        --save_subdir="${SAVE_SUBDIR}" \
        --workdir="${WORKDIR}"
done

echo "============================================================"
echo " All 10 experiments completed."
echo " Results are located in cmpinns/results/pedagogical_baseline_comparison/"
echo "============================================================"