import os
import numpy as np
import torch
import ml_collections
from copy import deepcopy
from examples.pedagogical_example.trainner import train
from examples.pedagogical_example.models import Pedagogical
from examples.pedagogical_example.utils import generate_reaction_ode_dataset

def get_l2_error(config, workdir):
    """計算當前模型在整個定義域上的 Relative L2 Error"""
    device = config.device
    params = config.system_pedagogical.system_params
    T, u0, n_t = params['T'], params['u0'], params['n_t']
    
    # 1. 產生 Ground Truth (Full PDE domain)
    _, _, _, sol = generate_reaction_ode_dataset(params, T=T, u0=u0, n_t=n_t)
    t_test = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1).to(device)
    u_true = sol.y[0]

    # 2. 載入模型
    model = Pedagogical(config).to(device)
    model_path = os.path.join(workdir, config.saving.save_dir, config.saving.finetune_path, "final_model.pt")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. 預測與計算誤差
    with torch.no_grad():
        u_pred = model(t_test).cpu().numpy().ravel()
    
    relative_l2 = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
    return relative_l2

def run_benchmarks(base_config, num_runs=3):
    """
    針對不同的觀測點數量跑實驗
    num_runs: 每個點數跑幾次不同的 seed 來計算 std
    """
    sample_sizes = [10, 15, 30, 50, 100]
    results = {size: [] for size in sample_sizes}

    for size in sample_sizes:
        print(f"\n>>> Testing Measurement Points: {size}")
        for run in range(num_runs):
            # 複製一份新的 config 避免污染
            config = deepcopy(base_config)
            config.sample_size = size
            config.seed = 42 + run  # 每次使用不同的 seed
            
            # 設定獨立的儲存路徑防止覆蓋
            config.saving.save_dir = f"results_benchmark/size_{size}_run_{run}"
            workdir = "."
            
            print(f"    Run {run+1}/{num_runs} (Seed: {config.seed})...")
            
            # 執行訓練 (DAPINN 流程)
            train(config, workdir)
            
            # 評估並記錄誤差
            error = get_l2_error(config, workdir)
            results[size].append(error)
            print(f"    Error: {error:.4e}")

    # --- 輸出表格 ---
    print("\n" + "="*50)
    print("Relative L2 error (mean and std) for DAPINNs")
    print("="*50)
    
    header = "Points:      " + " | ".join([f"{s:^10}" for s in sample_sizes])
    print(header)
    print("-" * len(header))

    means = [np.mean(results[s]) for s in sample_sizes]
    stds = [np.std(results[s]) for s in sample_sizes]

    mean_row = "Mean:        " + " | ".join([f"{m:.2e}" for m in means])
    std_row  = "(Std):" + " | ".join([f"({s:.2e})" for s in stds])

    print(mean_row)
    print(std_row)
    print("="*50)

if __name__ == "__main__":
    from examples.pedagogical_example.configs.default import get_config
    cfg = get_config()
    # 確保開啟 DAPINN 模式
    cfg.use_corrector = True
    cfg.run_pretrain = True 
    cfg.run_finetune = True
    
    run_benchmarks(cfg, num_runs=5) # 實際論文通常 num_runs=5 或 10