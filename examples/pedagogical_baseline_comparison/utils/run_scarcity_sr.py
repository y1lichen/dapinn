import os
import torch
import numpy as np
import pandas as pd
import multiprocessing as mp
import warnings

import glob

# ========================================================
# 1. 處理 PySR / Julia 與 PyTorch 的衝突 (參照您的 sr.py)
# ========================================================
warnings.filterwarnings("ignore", message="torch was imported before juliacall")
warnings.filterwarnings("ignore", message="juliacall module already imported")
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

from examples.pedagogical_baseline_comparison.configs.default import get_config
from examples.pedagogical_baseline_comparison.models import PedagogicalBaselineComaprison, Corrector

N_LIST = [10, 15, 30, 100, 1000]
NUM_TRIALS = 10  # 只要檢查前 10 個 seed 即可
RESULTS_DIR = "results_scarcity"


# ========================================================
# 2. PySR 獨立執行函數 (完全參照您 sr.py 的成功配置)
# ========================================================
def run_pysr_task(X, y, out_dir):
    """在子進程中執行 PySR"""
    from pysr import PySRRegressor
    
    # 完全使用您 sr.py 支援的參數格式：只給 output_directory
    model = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*"], # 目標是 a*u - b*u^2，加減乘即可
        population_size=50,
        maxsize=10,
        elementwise_loss="L2DistLoss()",
        progress=False, # 關閉進度條避免 50 次洗版
        output_directory=out_dir 
    )
    
    # 指定變數名稱為 u
    model.fit(X, y, variable_names=["u"])


# ========================================================
# 3. 收集預測資料並呼叫 PySR
# ========================================================
def perform_sr_for_all():
    print("========== 開始執行 PySR 符號迴歸 (Without cos) ==========")
    config = get_config()
    device = config.device
    
    # 產生密集的 t 來產生數據給 PySR 訓練 (0 到 T=1.0)
    T = config.system_pedagogical.system_params['T']
    t_np = np.linspace(0, T, 500)
    t_tensor = torch.tensor(t_np, dtype=torch.float32).reshape(-1, 1).to(device)
    
    found_equations = {N: [] for N in N_LIST}
    
    for N in N_LIST:
        print(f"\n>>> 正在處理 N = {N} ...")
        for seed in range(1, NUM_TRIALS + 1):
            seed_dir = os.path.join(RESULTS_DIR, f"N_{N}", f"seed_{seed}")
            ckpt_model = os.path.join(seed_dir, "finetuned", "final_model.pt")
            ckpt_corr = os.path.join(seed_dir, "corrector", "final_corrector.pt")
            
            sr_out_dir = os.path.join(seed_dir, "sr_results")
            os.makedirs(sr_out_dir, exist_ok=True)
            
            if not (os.path.exists(ckpt_model) and os.path.exists(ckpt_corr)):
                print(f"  [Skip] N={N}, Seed={seed} 找不到模型權重。")
                found_equations[N].append("N/A")
                continue
                
            # 1. 載入模型預測 u 與 s
            model = PedagogicalBaselineComaprison(config).to(device)
            corrector = Corrector(config).to(device)
            model.load_state_dict(torch.load(ckpt_model, map_location=device, weights_only=True)["model_state_dict"])
            corrector.load_state_dict(torch.load(ckpt_corr, map_location=device, weights_only=True)["model_state_dict"])
            model.eval(); corrector.eval()
            
            # with torch.no_grad():
            #     u_pred_t = model(t_tensor)
            #     # 您的 models.py 目前設定 corrections_inputs = u
            #     s_pred_t = corrector(u_pred_t)
            #     X = u_pred_t.cpu().numpy().reshape(-1, 1)
            #     y = s_pred_t.cpu().numpy().flatten()
            
            with torch.no_grad():
                u_pred_t = model(t_tensor)
                # 您的 models.py 目前設定 corrections_inputs = u
                s_pred_t = corrector(u_pred_t)
                
                # =========================================================
                # 【學長提點的關鍵修改】：
                # Corrector 學到的是殘差: s = lambda*u(1-u) - lambda*cos(u)
                # 我們必須把錯誤假設的 lambda*cos(u) 加回去，還原出完整的反應項，
                # 這樣 PySR 才能去尋找純粹的 a*u - b*u^2 結構！
                # =========================================================
                lambda_param = config.system_pedagogical.system_params['lambda']
                phi_pred_t = s_pred_t + lambda_param * torch.cos(u_pred_t)
                
            X = u_pred_t.cpu().numpy().reshape(-1, 1)
            
            # y 改成我們還原出來的完整物理項 (ADPC + cos(u))
            y = phi_pred_t.cpu().numpy().flatten()
            
            # 2. 開啟子進程跑 PySR (Spawn 模式防止 Segfault)
            ctx = mp.get_context('spawn')
            process = ctx.Process(target=run_pysr_task, args=(X, y, sr_out_dir))
            process.start()
            process.join()
            
            # 3. 讀取結果 (支援自動搜尋 PySR 生成的時間戳記子目錄)
            search_pattern = os.path.join(sr_out_dir, "**", "hall_of_fame.csv")
            csv_files = glob.glob(search_pattern, recursive=True)
            
            if csv_files:
                # 拿找到的第一個 CSV 檔案
                csv_path = csv_files[0] 
                try:
                    df_sr = pd.read_csv(csv_path)
                    best_eq = df_sr.iloc[-1]['equation']
                    found_equations[N].append(best_eq)
                    print(f"  [Seed {seed:2d}] 找到: {best_eq}")
                except Exception as e:
                    found_equations[N].append("Read Error")
                    print(f"  [Seed {seed:2d}] 讀取 CSV 失敗: {e}")
            else:
                found_equations[N].append("SR Failed")
                print(f"  [Seed {seed:2d}] SR 失敗")
    return found_equations


# ========================================================
# 4. 輸出人工判讀 Table
# ========================================================
def print_manual_evaluation_table(equations_dict):
    print("\n\n" + "="*80)
    print(" 🛠️ 請人工檢視以下方程式，並填寫 Structural Equivalence 表格 🛠️")
    print(" 目標結構應該類似於: a * u - b * u^2  (對應 lambda * u * (1 - u))")
    print("="*80)
    
    for N in N_LIST:
        print(f"\n[ N = {N} ] 的方程式列表:")
        for idx, eq in enumerate(equations_dict[N]):
            print(f"  Seed {idx+1:2d}: {eq}")

    print("\n" + "="*60)
    print("Discrepancy identification results for DAPINNs with ADPC using PySR.")
    print("-" * 60)
    print(f"{'Number of measurement points':<30} " + "  ".join([f"{N:>5}" for N in N_LIST]))
    print(f"{'Structural equivalence':<30} " + "  ".join([f" ?/10" for N in N_LIST]))
    print("="*60 + "\n")


if __name__ == "__main__":
    eq_dict = perform_sr_for_all()
    print_manual_evaluation_table(eq_dict)