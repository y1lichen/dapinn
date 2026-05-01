import os
import subprocess

# 定義要測試的採樣點數量與 Seed 次數
N_LIST = [10, 15, 20, 30, 100, 1000]
NUM_TRIALS = 30  # 建議第一次測試時先改為 2，確認沒問題再放著讓它跑完 30 次
BASE_SAVE_DIR = "results_scarcity"

def run_dapinn_scarcity():
    print("========== 啟動資料稀缺性 (Data Scarcity) 自動化實驗 ==========")

    for N in N_LIST:
        print(f"\n" + "="*50)
        print(f" 開始執行採樣點數量 N = {N} 的實驗 ")
        print("="*50)

        for seed in range(1, NUM_TRIALS + 1):
            print(f"\n[N={N} | Seed={seed}] 執行中...")

            # 設定存檔路徑 (對應到繪圖腳本會去抓的路徑)
            save_subdir = os.path.join(BASE_SAVE_DIR, f"N_{N}", f"seed_{seed}")

            # 確保使用錯誤的物理公式作為先驗 (DAPINNs 必須在錯誤領域 pretrain)
            env = os.environ.copy()
            env["USE_CORRECT_PHYSICS"] = "0"

            # 組合共用指令
            # 使用 ml_collections 的內建參數覆寫機制：--config.sample_size={N}
            base_cmd = [
                "python", "-m", "examples.pedagogical_baseline_comparison.main",
                f"--seed={seed}",
                f"--save_subdir={save_subdir}",
                "--use_corrector=True",
                "--run_pretrain=True",
                "--load_pretrained=True",
                "--run_finetune=True",
                f"--config.sample_size={N}"  # <--- 動態改變模型訓練用的採樣點數量
            ]

            try:
                # 1. 執行訓練 (Train)
                print(f"  -> Training (Pretrain + Finetune + Corrector)...")
                train_cmd = base_cmd + ["--mode=train"]
                subprocess.run(train_cmd, env=env, check=True)

                # 2. 執行評估 (Eval)，這會產出 evaluation_results.json
                print(f"  -> Evaluating...")
                eval_cmd = base_cmd + ["--mode=eval"]
                subprocess.run(eval_cmd, env=env, check=True)

                print(f"  -> [成功] 結果已儲存至 {save_subdir}/evaluation_results.json")

            except subprocess.CalledProcessError as e:
                print(f"  -> [錯誤] N={N}, Seed={seed} 執行失敗: {e}")
                # 發生錯誤時繼續跑下一個 seed，不要讓整個 180 次的迴圈中斷
                continue

    print("\n========== 所有資料稀缺性實驗已執行完畢！ ==========")

if __name__ == "__main__":
    run_dapinn_scarcity()