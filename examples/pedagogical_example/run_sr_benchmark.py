import os
import re
from copy import deepcopy

import ml_collections
import numpy as np
import pandas as pd
import torch

from examples.pedagogical_example.configs.default import get_config
from examples.pedagogical_example.sr import execute_sr
from examples.pedagogical_example.trainner import train


def analyze_pysr_result(sr_dir, target_lambda=0.2):
    """
    分析 PySR 產生的 hall_of_fame.csv
    回傳: (是否結構等價, 係數誤差%)
    """
    csv_path = os.path.join(sr_dir, "hall_of_fame.csv")
    if not os.path.exists(csv_path):
        return False, None

    try:
        df = pd.read_csv(csv_path)
        # 取得 Loss 最低且複雜度適中的方程式 (通常是最後一列)
        # 我們尋找包含 'u' 且最接近我們預期形式的公式
        best_eq = df.iloc[-1]["Equation"]

        # 結構檢查邏輯：
        # 1. 必須包含 u
        # 2. 最好包含 u*u 或 u^2
        # 3. 不應包含 du (導數項，除非雜訊很大)
        has_u = "u" in best_eq
        has_u2 = "u * u" in best_eq or "u**2" in best_eq or "u^2" in best_eq
        has_du = "du" in best_eq

        is_equivalent = has_u and has_u2 and not has_du

        # 係數提取邏輯 (使用正則表達式尋找 u 前面的數字)
        # 假設公式形如: 0.198 * u - 0.201 * u * u
        # 我們提取第一個 u 的係數作為 lambda 的估計
        coeffs = re.findall(r"[-+]?\d*\.\d+|\d+", best_eq)
        if coeffs:
            found_lambda = float(coeffs[0])
            coef_error = abs(found_lambda - target_lambda) / target_lambda * 100
        else:
            coef_error = None

        return is_equivalent, coef_error
    except Exception as e:
        print(f"Error analyzing CSV: {e}")
        return False, None


def run_table11_benchmark(num_trials=10):
    """
    執行 Table 11 實驗
    num_trials: 每個樣本點數重複實驗的次數 (論文中為 10)
    """
    sample_sizes = [10, 15, 30, 100, 1000]
    final_stats = {}

    workdir = "."
    base_cfg = get_config()
    base_cfg.use_corrector = True
    base_cfg.run_pretrain = True  # 先統一做一次預訓練

    # --- Step 0: 共享預訓練 ---
    print("\n>>> Pre-training initial model...")
    shared_pretrain_dir = "results/results_table11/shared_pretrain"
    base_cfg.saving.save_dir = shared_pretrain_dir
    train(base_cfg, workdir)
    shared_pretrained_path = os.path.join(
        workdir, shared_pretrain_dir, "pretrained", "best_model.pt"
    )

    for size in sample_sizes:
        print(f"\n" + "=" * 50)
        print(f">>> TESTING SAMPLE SIZE: {size}")
        print("=" * 50)

        trial_results = []
        for trial in range(num_trials):
            print(f"\n--- Trial {trial + 1}/{num_trials} (Size: {size}) ---")

            cfg = deepcopy(base_cfg)
            cfg.sample_size = size
            cfg.seed = 100 + trial  # 確保每次種子不同
            cfg.run_pretrain = False
            cfg.run_finetune = True
            cfg.load_pretrained = True

            # 設定實驗路徑
            exp_dir = f"results/results_table11/size_{size}/trial_{trial}"
            cfg.saving.save_dir = exp_dir

            # 準備預訓練權重
            os.makedirs(os.path.join(workdir, exp_dir, "pretrained"), exist_ok=True)
            import shutil

            shutil.copy(
                shared_pretrained_path,
                os.path.join(workdir, exp_dir, "pretrained", "best_model.pt"),
            )

            # 1. 訓練 DAPINN
            train(cfg, workdir)

            # 2. 執行 PySR 符號回歸
            execute_sr(cfg, workdir)

            # 3. 分析 SR 結果
            sr_res_dir = os.path.join(
                workdir, exp_dir, cfg.saving.corrector_path, "sr_results"
            )
            is_equiv, c_err = analyze_pysr_result(sr_res_dir)

            trial_results.append({"equiv": is_equiv, "error": c_err})
            print(
                f"Trial {trial + 1} Result: StructEquiv={is_equiv}, CoefError={c_err}"
            )

        # 統計該樣本點數的結果
        equiv_count = sum([1 for r in trial_results if r["equiv"]])
        valid_errors = [r["error"] for r in trial_results if r["error"] is not None]

        avg_err = np.mean(valid_errors) if valid_errors else 0
        std_err = np.std(valid_errors) if valid_errors else 0

        final_stats[size] = {
            "equiv": f"{equiv_count}/{num_trials}",
            "mean_err": avg_err,
            "std_err": std_err,
        }

    # --- 最終列印表格 ---
    print("\n\n" + " " * 10 + "FINAL TABLE 11 RESULTS")
    print("=" * 70)
    print(
        f"{'Points':<15} | {'10':<10} | {'15':<10} | {'30':<10} | {'100':<10} | {'1000':<10}"
    )
    print("-" * 70)

    equiv_row = f"{'Structural Equiv':<15} | " + " | ".join(
        [f"{final_stats[s]['equiv']:<10}" for s in sample_sizes]
    )
    mean_row = f"{'Mean Error %':<15} | " + " | ".join(
        [f"{final_stats[s]['mean_err']:<10.2f}" for s in sample_sizes]
    )
    std_row = f"{'Std Error':<15} | " + " | ".join(
        [f"({final_stats[s]['std_err']:<8.2f})" for s in sample_sizes]
    )

    print(equiv_row)
    print(mean_row)
    print(std_row)
    print("=" * 70)

    print(mean_row)
    print(std_row)
    print("=" * 70)

    print(std_row)
    print("=" * 70)

    print("=" * 70)


if __name__ == "__main__":
    run_table11_benchmark(num_trials=1)
