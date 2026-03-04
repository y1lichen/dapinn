#!/bin/bash

set -euo pipefail

# 設定工作目錄
WORKDIR="${1:-.}"

# --- 實驗參數設定 ---
NOISE_LEVELS=(0.01 0.03 0.05 0.1)
SAMPLE_SIZES=(30 100) # 針對 30 點與 100 點分別測試
NUM_TRIALS=30         # 每個組合跑 30 次以獲得統計意義
BASE_SEED=42          

# 定義結果儲存根目錄
RESULT_BASE="cmpinns/results/pedagogical_baseline_comparison/noise_experiment"

echo "Starting/Resuming CMPINN Baseline Noise & Size Experiment..."
echo "Noise Levels: ${NOISE_LEVELS[*]}"
echo "Sample Sizes: ${SAMPLE_SIZES[*]}"
echo "Trials per combination: ${NUM_TRIALS}"

for NOISE in "${NOISE_LEVELS[@]}"; do
    for SIZE in "${SAMPLE_SIZES[@]}"; do
        echo "------------------------------------------------------------"
        echo " >>> CONFIG: Noise=${NOISE} | Size=${SIZE} <<< "
        echo "------------------------------------------------------------"

        for ((i=1; i<=NUM_TRIALS; i++)); do
            CURRENT_SEED=$((BASE_SEED + i))
            
            # 定義儲存路徑：這行確保了 noise 資料夾下會有 size 資料夾
            SAVE_SUBDIR="${RESULT_BASE}/noise_${NOISE}/size_${SIZE}/trial_${i}_seed_${CURRENT_SEED}"
            
            # --- 續傳邏輯：檢查評估指標 JSON 是否已產出 ---
            # 指標包含 u_l2_relative_error (預測誤差) 與 s_l2_relative_error (物理修正誤差)
            RESULT_FILE="${WORKDIR}/${SAVE_SUBDIR}/evaluation_results.json"
            
            if [ -f "$RESULT_FILE" ]; then
                echo "[SKIP] Noise ${NOISE} | Size ${SIZE} | Trial ${i} exists."
                continue
            fi

            echo "[RUN] Noise ${NOISE} | Size ${SIZE} | Trial ${i} | Seed: ${CURRENT_SEED}"

            # --- 執行訓練 ---
            # 此階段會利用 ADPC 機制同步優化 PINN 與 Corrector
            python -m cmpinns.examples.pedagogical_baseline_comparison.main \
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

            # --- 執行評估 ---
            # 產出評估圖表與 evaluation_results.json
            python -m cmpinns.examples.pedagogical_baseline_comparison.main \
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
echo " Results stored in: ${WORKDIR}/${RESULT_BASE}"
echo "============================================================"