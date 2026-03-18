#!/bin/bash

# --- 實驗參數設定 ---
NOISE_LEVELS=(0.01 0.03 0.05 0.1)
SAMPLE_SIZES=(30 100) 
NUM_TRIALS=30     
BASE_SEED=42          
WORKDIR="${1:-.}"

# 確保輸出根目錄存在
mkdir -p "${WORKDIR}/results/pedagogical_baseline_comparison/noise_experiment"

echo "Starting/Resuming Combined Noise & Sample Size Experiment..."
echo "Noise Levels: ${NOISE_LEVELS[*]}"
echo "Sample Sizes: ${SAMPLE_SIZES[*]}"

for NOISE in "${NOISE_LEVELS[@]}"; do
    for SIZE in "${SAMPLE_SIZES[@]}"; do
        echo "------------------------------------------------------------"
        echo " >>> CONFIG: Noise=${NOISE} | Size=${SIZE} <<< "
        echo "------------------------------------------------------------"

        for ((i=1; i<=NUM_TRIALS; i++)); do
            CURRENT_SEED=$((BASE_SEED + i))
            
            # 定義儲存路徑
            SAVE_SUBDIR="results/pedagogical_baseline_comparison/noise_experiment/noise_${NOISE}/size_${SIZE}/trial_${i}_seed_${CURRENT_SEED}"
            
            # --- 續傳邏輯：檢查結果檔案是否存在 ---
            # 根據 eval.py 的 update_metrics 函式，指標會存於 evaluation_results.json
            RESULT_FILE="${WORKDIR}/${SAVE_SUBDIR}/evaluation_results.json"
            
            if [ -f "$RESULT_FILE" ]; then
                echo "[SKIP] Trial ${i} for Noise ${NOISE}/Size ${SIZE} already completed."
                continue
            fi

            echo "[RUN] Noise ${NOISE} | Size ${SIZE} | Trial ${i} | Seed: ${CURRENT_SEED}"

            # 執行 Training (包含 Pre-train 與 Fine-tune)
            # 注意：這裡會利用 Corrector (ADPC) 進行物理修正
            python -m examples.pedagogical_baseline_comparison.main \
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
            # 計算包含 u_l2_relative_error 與 s_l2_relative_error 的指標並寫入 JSON
            python -m examples.pedagogical_baseline_comparison.main \
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
echo " All experiments completed."
echo "============================================================"