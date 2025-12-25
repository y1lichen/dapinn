import os
import ml_collections
import torch
import numpy as np
import multiprocessing as mp # 建議明確標註 mp
import warnings

# 1. 嚴格處理 Julia 與 PyTorch 的衝突
warnings.filterwarnings("ignore", message="torch was imported before juliacall")
warnings.filterwarnings("ignore", message="juliacall module already imported")

# 強制禁用某些信號處理，減少 Segmentation Fault 機率
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

from examples.pedagogical_example.models import Corrector

def run_pysr(X, s, corrector_sr_dir):
    """在獨立的子進程中執行，確保 Julia 環境乾淨"""
    # 延遲導入，防止父進程的環境干擾
    from pysr import PySRRegressor
    
    # 修正：FutureWarning 提到 variable_names 應該在 fit 時傳入
    model = PySRRegressor(
        niterations=50,
        binary_operators=["+", "-", "*", "/"],
        # unary_operators=["abs"],
        population_size=100,
        maxsize=15,
        elementwise_loss="L2DistLoss()",
        progress=True,
        output_directory=corrector_sr_dir,
    )

    print(f"\n[SR] Fitting symbolic expressions to Corrector outputs...")
    # 在此處傳入變數名稱
    model.fit(X, s, variable_names=["u", "du"])

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

if __name__ == "__main__":
    pass