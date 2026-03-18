#!/bin/bash

set -euo pipefail

# 設定工作目錄，預設為當前目錄
WORKDIR="${1:-.}"

# --- 實驗參數設定 ---
NUM_TRIALS=30
BASE_SEED=42
SAMPLE_SIZE=30   # 樣本點數

# 定義結果儲存根目錄
RESULT_BASE="results/pedagogical_baseline_comparison/dapinn_trials"

echo "============================================================"
echo " Starting DAPINN 10-Trial Experiment Pipeline"
echo " Total Trials: ${NUM_TRIALS}"
echo "============================================================"

for ((i=1; i<=NUM_TRIALS; i++)); do
    CURRENT_SEED=$((BASE_SEED + i))
    
    # 定義該次 Trial 的專屬儲存路徑
    TRIAL_DIR="${RESULT_BASE}/trial_${i}_seed_${CURRENT_SEED}"
    
    echo ""
    echo "------------------------------------------------------------"
    echo " >>> Starting Trial ${i}/${NUM_TRIALS} | Seed: ${CURRENT_SEED}"
    echo " >>> Directory: ${TRIAL_DIR}"
    echo "------------------------------------------------------------"

    # --- 步驟 1: Pre-training (物理引導初始化) ---
    # 此階段通常不涉及數據雜訊，但設定 seed 是好習慣
    echo ">>> [Trial ${i}] Step 1/2: Pre-training (Physics only)..."
    python -m examples.pedagogical_baseline_comparison.main \
        --mode=train \
        --use_corrector=True \
        --run_pretrain=True \
        --run_finetune=False \
        --seed="${CURRENT_SEED}" \
        --save_subdir="${TRIAL_DIR}" \
        --workdir="${WORKDIR}"

    # --- 步驟 2: Fine-tuning (數據擬合 + 修正器啟動) ---
    # 此階段會載入 Pretrain 權重，並使用帶有雜訊的數據集
    echo ">>> [Trial ${i}] Step 2/2: Fine-tuning (Data + Corrector)..."
    python -m examples.pedagogical_baseline_comparison.main \
        --mode=train \
        --use_corrector=True \
        --run_pretrain=False \
        --run_finetune=True \
        --load_pretrained=True \
        --sample_size="${SAMPLE_SIZE}" \
        --seed="${CURRENT_SEED}" \
        --save_subdir="${TRIAL_DIR}" \
        --workdir="${WORKDIR}"

    # --- 步驟 3: Evaluation (評估並產出 JSON 指標) ---
    echo ">>> [Trial ${i}] Step 3: Evaluating..."
    python -m examples.pedagogical_baseline_comparison.main \
        --mode=eval \
        --use_corrector=True \
        --save_subdir="${TRIAL_DIR}" \
        --workdir="${WORKDIR}"

done

echo ""
echo "============================================================"
echo " All 10 DAPINN Trials completed successfully."
echo " Results stored in: ${WORKDIR}/${RESULT_BASE}"
echo "============================================================"