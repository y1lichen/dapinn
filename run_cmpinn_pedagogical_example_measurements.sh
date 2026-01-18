#!/bin/bash

set -euo pipefail

# 設定工作目錄，預設為當前目錄
WORKDIR="${1:-.}"

# 實驗參數設定
SAMPLE_SIZES=(30 50 100 200 500)
NUM_TRIALS=30
BASE_SEED=42  # 基礎種子，確保實驗可重複性

echo "Starting Measurement Experiments..."
echo "Sample Sizes: ${SAMPLE_SIZES[*]}"
echo "Trials per size: ${NUM_TRIALS}"

for SIZE in "${SAMPLE_SIZES[@]}"; do
    echo "------------------------------------------------------------"
    echo " >>> TARGET SAMPLE SIZE: ${SIZE} <<< "
    echo "------------------------------------------------------------"

    for ((i=1; i<=NUM_TRIALS; i++)); do
        # 每個 trial 使用不同的 seed
        CURRENT_SEED=$((BASE_SEED + i))
        
        # 定義獨立的儲存路徑，避免數據覆蓋
        SAVE_SUBDIR="cmpinns/results/pedagogical_example/size_${SIZE}/trial_${i}_seed_${CURRENT_SEED}"
        
        echo "[Size ${SIZE} | Trial ${i}] Seed: ${CURRENT_SEED} -> ${SAVE_SUBDIR}"

        # 執行訓練
        # 注意：這裡將 run_pretrain 設為 True，因為每個 trial 需要重新開始
        # 並將 sample_size 傳入以控制數據點數
        python -m cmpinns.examples.pedagogical_example.main \
            --mode=train \
            --use_corrector=True \
            --run_pretrain=True \
            --run_finetune=True \
            --load_pretrained=False \
            --sample_size="${SIZE}" \
            --seed="${CURRENT_SEED}" \
            --save_subdir="${SAVE_SUBDIR}" \
            --workdir="${WORKDIR}"

        # 執行評估，計算 L2 error 與 MSE
        python -m cmpinns.examples.pedagogical_example.main \
            --mode=eval \
            --use_corrector=True \
            --sample_size="${SIZE}" \
            --save_subdir="${SAVE_SUBDIR}" \
            --workdir="${WORKDIR}"
            
    done
done

echo "============================================================"
echo " All measurements completed."
echo " Results stored in cmpinns/results/measurements/"
echo "============================================================"