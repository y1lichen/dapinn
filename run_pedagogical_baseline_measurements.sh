#!/bin/bash

# 實驗參數設定
SAMPLE_SIZES=(30 50 100 200 500)
NUM_TRIALS=10  # 每個 sample size 跑 10 次
BASE_SEED=42   # 起始 seed，每次 trial 會增加
WORKDIR="${1:-.}"

# 確保輸出根目錄存在
mkdir -p "${WORKDIR}/results/pedagogical_baseline_comparison"

echo "Starting Measurements Experiment..."
echo "Sample Sizes: ${SAMPLE_SIZES[*]}"
echo "Trials per size: ${NUM_TRIALS}"

for SIZE in "${SAMPLE_SIZES[@]}"; do
    echo "------------------------------------------------------------"
    echo " >>> TARGET SAMPLE SIZE: ${SIZE} <<< "
    echo "------------------------------------------------------------"

    for ((i=1; i<=NUM_TRIALS; i++)); do
        CURRENT_SEED=$((BASE_SEED + 5))
        SAVE_SUBDIR="results/pedagogical_baseline_comparison/size_${SIZE}/trial_${i}_seed_${CURRENT_SEED}"
        
        echo "[Size ${SIZE} | Trial ${i}] Using Seed: ${CURRENT_SEED}"
        echo "Saving to: ${SAVE_SUBDIR}"

        # --- Step 1: Pre-training ---
        # 註：如果每個 trial 的 pretrain 都一樣，可以考慮只跑一次共用的
        python -m examples.pedagogical_baseline_comparison.main \
            --mode=train \
            --use_corrector=True \
            --run_pretrain=True \
            --run_finetune=False \
            --sample_size="${SIZE}" \
            --seed="${CURRENT_SEED}" \
            --save_subdir="${SAVE_SUBDIR}" \
            --workdir="${WORKDIR}"

        # --- Step 2: Fine-tuning ---
        python -m examples.pedagogical_baseline_comparison.main \
            --mode=train \
            --use_corrector=True \
            --run_pretrain=False \
            --run_finetune=True \
            --load_pretrained=True \
            --sample_size="${SIZE}" \
            --seed="${CURRENT_SEED}" \
            --save_subdir="${SAVE_SUBDIR}" \
            --workdir="${WORKDIR}"

        # --- Step 3: Evaluation & Symbolic Regression ---
        # 這步驟會產出 PySR 的結果，方便之後手動或腳本紀錄 Coefficient Error
        python -m examples.pedagogical_baseline_comparison.main \
            --mode=eval \
            --use_corrector=True \
            --sample_size="${SIZE}" \
            --seed="${CURRENT_SEED}" \
            --save_subdir="${SAVE_SUBDIR}" \
            --workdir="${WORKDIR}"
            
    done
done

echo "============================================================"
echo " All measurements completed. "
echo " Results are stored in results/pedagogical_baseline_comparison/size_N/trial_i_seed_S "
echo "============================================================"