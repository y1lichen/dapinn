import os
import ml_collections
import torch
import numpy as np
import multiprocessing as mp # 建議明確標註 mp
import warnings
import re
import pandas as pd
from . import trainner


# 1. 嚴格處理 Julia 與 PyTorch 的衝突
warnings.filterwarnings("ignore", message="torch was imported before juliacall")
warnings.filterwarnings("ignore", message="juliacall module already imported")

# 強制禁用某些信號處理，減少 Segmentation Fault 機率
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

from examples.pedagogical_baseline_comparison.models import Corrector

def run_pysr(X, s, corrector_sr_dir):
    """在獨立的子進程中執行，確保 Julia 環境乾淨"""
    # 延遲導入，防止父進程的環境干擾
    from pysr import PySRRegressor
    
    # 修正：FutureWarning 提到 variable_names 應該在 fit 時傳入
    model = PySRRegressor(
        niterations=50,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["cos", "sin"],
        population_size=100,
        maxsize=15,
        elementwise_loss="L2DistLoss()",
        progress=True,
        output_directory=corrector_sr_dir,
    )

    print(f"\n[SR] Fitting symbolic expressions to Corrector outputs...")
    # 在此處傳入變數名稱
    
    model.fit(X, s, variable_names=["u", "du"])
    # model.fit(X, s)
    print("\n[SR] Best symbolic equation found:")
    print(model)

def execute_sr(config: ml_collections.ConfigDict, workdir: str):
    """符號回歸的主要工作流程"""
    device = config.device
    
    # 1. 準備路徑
    save_root = os.path.join(workdir, config.saving.save_dir)
    corrector_dir = os.path.join(save_root, config.saving.corrector_path)
    corrector_sr_dir = os.path.join(corrector_dir, "sr_results")
    os.makedirs(corrector_sr_dir, exist_ok=True)

    # 2. 載入模型與資料
    corrector = Corrector(config).to(device)
    checkpoint_path = os.path.join(corrector_dir, config.corrector_model_name)
    
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(corrector_dir, "final_corrector.pt")

    print(f"[SR] Loading Corrector from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        corrector.load_state_dict(checkpoint["model_state_dict"])
        
        if 'corrector_inputs' in checkpoint:
            # 轉為 Numpy 並移出 GPU 是避開 Segfault 的關鍵
            X = checkpoint['corrector_inputs'].detach().cpu().numpy()
            print("[SR] Corrector inputs loaded from checkpoint.")
        else:
            print("[WARN] No corrector_inputs in checkpoint.")
            return
    except Exception as e:
        print(f"[ERROR] Loading failed: {e}")
        return

    # 3. 獲取 Corrector 輸出
    corrector.eval()
    with torch.no_grad():
        s_pred = corrector(torch.tensor(X).to(device))
        s = s_pred.cpu().numpy().flatten()

    # 4. 使用 'spawn' 模式啟動子進程 (解決 Segfault 的核心)
    # Fork 模式會複製父進程的 CUDA 狀態，導致 Julia 初始化失敗
    ctx = mp.get_context('spawn')
    process = ctx.Process(target=run_pysr, args=(X, s, corrector_sr_dir))
    
    print("[SR] Starting PySR search...")
    process.start()
    process.join()


def parse_pysr_results(sr_dir, target_coeff):
    """
    解析 PySR 產生的 csv，檢查結構是否等價並計算係數誤差。
    """
    csv_path = os.path.join(sr_dir, "hall_of_fame.csv")
    if not os.path.exists(csv_path):
        return False, None
    
    df_sr = pd.read_csv(csv_path)
    # 取得 Loss 最低或 Complexity 適中的最佳方程式 (最後一列通常是最佳)
    best_eq = df_sr.iloc[-1] 
    equation_str = best_eq['equation']
    
    # 1. 結構檢查：是否只包含 u 且為線性關係 (簡單的正則表達式)
    # 預期格式如: (1.234 * u) 或 (u * 1.234)
    is_struct_match = bool(re.search(r'u', equation_str)) and not bool(re.search(r'du', equation_str))
    
    # 2. 係數提取
    try:
        # 尋找算式中的浮點數
        coeffs = re.findall(r"[-+]?\d*\.\d+|\d+", equation_str)
        if is_struct_match and coeffs:
            pred_coeff = float(coeffs[0])
            error = abs(pred_coeff - target_coeff) / abs(target_coeff) * 100
            return True, error
    except:
        pass
        
    return is_struct_match, None

def generate_discrepancy_table(config, workdir, sample_sizes=[50, 100, 200, 500, 1000], num_trials=10):
    """
    執行完整實驗流程並生成 Table 3 格式。
    """
    final_results = []
    target_coeff = config.system_pedagogical.system_params['lambda']
    
    print(f"\n[Table Generator] Starting experiments for lambda={target_coeff}")

    for N in sample_sizes:
        print(f"\n>>> Testing Sample Size: {N}")
        success_count = 0
        errors = []
        
        for trial in range(num_trials):
            # 設定唯一的存檔路徑
            config.sample_size = N
            config.seed = trial + 42 # 改變種子確保隨機性
            trial_subdir = f"results_N{N}_trial{trial}"
            config.saving.save_dir = os.path.join(workdir, trial_subdir)
            
            # 1. 執行訓練 (假設 trainer.train 已定義)
            trainner.train(config, workdir)
            
            # 2. 執行 SR
            execute_sr(config, workdir)
            
            # 3. 解析結果
            sr_dir = os.path.join(config.saving.save_dir, config.saving.corrector_path, "sr_results")
            is_success, error = parse_pysr_results(sr_dir, target_coeff)
            
            if is_success:
                success_count += 1
                if error is not None:
                    errors.append(error)
            
            print(f"Trial {trial}: {'Success' if is_success else 'Fail'}, Error: {error:.2f}%" if error else f"Trial {trial}: Fail")

        # 彙整該 N 的統計
        avg_error = np.mean(errors) if errors else 0.0
        final_results.append({
            "Sample Points": N,
            "Structural Equivalence": f"{success_count}/{num_trials}",
            "Coefficient Error (%)": f"{avg_error:.2f}"
        })

    # 生成表格
    df_table = pd.DataFrame(final_results).set_index("Sample Points").T
    print("\n" + "="*50)
    print("Discrepancy Identification Results")
    print("="*50)
    print(df_table)
    print("="*50)
    
    return df_table

