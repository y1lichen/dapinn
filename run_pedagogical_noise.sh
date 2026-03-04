#!/bin/bash

# --- 實驗參數設定 ---
NOISE_LEVELS=(0.01 0.02 0.03 0.05)
SAMPLE_SIZES=(30 100)
NUM_TRIALS=30
BASE_SEED=42       
WORKDIR="${1:-.}"

# 確保輸出根目錄存在
mkdir -p "${WORKDIR}/results/noise_experiment"

echo "Starting/Resuming Noise Levels & Data Scarcity Experiment..."

for NOISE in "${NOISE_LEVELS[@]}"; do
    for SIZE in "${SAMPLE_SIZES[@]}"; do
        echo "============================================================"
        echo " >>> CONFIG: Noise=${NOISE} | SampleSize=${SIZE} <<< "
        echo "============================================================"

        for ((i=1; i<=NUM_TRIALS; i++)); do
            CURRENT_SEED=$((BASE_SEED + i))
            SAVE_SUBDIR="results/pedagogical/noise_experiment/noise_${NOISE}/size_${SIZE}/trial_${i}_seed_${CURRENT_SEED}"
            
            # --- 關鍵：檢查評估結果是否存在，若存在則跳過 ---
            # 根據你的 eval.py，指標會存在這個路徑下的 evaluation_results.json
            RESULT_FILE="${WORKDIR}/${SAVE_SUBDIR}/evaluation_results.json"
            
            if [ -f "$RESULT_FILE" ]; then
                echo "[SKIP] Noise ${NOISE} | Size ${SIZE} | Trial ${i} already exists. Skipping..."
                continue
            fi

            echo "[RUN] Noise ${NOISE} | Size ${SIZE} | Trial ${i} | Seed: ${CURRENT_SEED}"

            # 執行 Training
            python -m examples.pedagogical_example.main \
                --mode=train \
                --use_corrector=True \
                --run_pretrain=True \
                --run_finetune=True \
                --load_pretrained=False \
                --sample_size="${SIZE}" \
                --noise="${NOISE}" \
                --seed="${CURRENT_SEED}" \
                --save_subdir="${SAVE_SUBDIR}" \
                --workdir="${WORKDIR}"

            # 執行 Evaluation
            python -m examples.pedagogical_example.main \
                --mode=eval \
                --use_corrector=True \
                --sample_size="${SIZE}" \
                --noise="${NOISE}" \
                --seed="${CURRENT_SEED}" \
                --save_subdir="${SAVE_SUBDIR}" \
                --workdir="${WORKDIR}"
                
        done
    done
done

echo "============================================================"
echo " All remaining experiments completed. "
echo "============================================================"