#!/bin/bash

set -euo pipefail

# 設定工作目錄
WORKDIR="${1:-.}"

# --- 實驗參數設定 ---
NOISE_LEVELS=(0.01 0.03 0.05 0.1)
SAMPLE_SIZES=(30 100) # 新增：兩種樣本數對照
NUM_TRIALS=30         # 每個組合跑 30 次以獲得統計意義
BASE_SEED=42          

# 定義結果儲存根目錄
RESULT_BASE="cmpinns/results/pedagogical_example/noise_experiment"

echo "Starting/Resuming CMPINN Noise & Size Experiment..."
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
            
            # 定義儲存路徑：加入 size_${SIZE} 節點
            SAVE_SUBDIR="${RESULT_BASE}/noise_${NOISE}/size_${SIZE}/trial_${i}_seed_${CURRENT_SEED}"
            
            # --- 續傳邏輯：檢查結果檔案是否存在 ---
            # 根據 eval.py，完整評估後會產出 evaluation_results.json
            RESULT_FILE="${WORKDIR}/${SAVE_SUBDIR}/evaluation_results.json"
            
            if [ -f "$RESULT_FILE" ]; then
                echo "[SKIP] Noise ${NOISE} | Size ${SIZE} | Trial ${i} already completed."
                continue
            fi

            echo "[RUN] Noise ${NOISE} | Size ${SIZE} | Trial ${i} | Seed: ${CURRENT_SEED}"

            # --- 步驟 1: 執行訓練 (Pre-training + Fine-tuning) ---
            # 訓練過程中會利用 Corrector ($s_\psi$) 進行物理補償
            python -m cmpinns.examples.pedagogical_example.main \
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

            # --- 步驟 2: 執行評估 ---
            # 計算 u_l2_relative_error 與 s_l2_relative_error 指標
            python -m cmpinns.examples.pedagogical_example.main \
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
echo " All Noise & Size experiments completed."
echo " Results stored in: ${WORKDIR}/${RESULT_BASE}"
echo "============================================================"