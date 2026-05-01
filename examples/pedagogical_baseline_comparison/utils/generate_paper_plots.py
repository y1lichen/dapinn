import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from examples.pedagogical_baseline_comparison.configs.default import get_config
from examples.pedagogical_baseline_comparison.models import PedagogicalBaselineComaprison, Corrector
from examples.pedagogical_baseline_comparison.utils import generate_reaction_ode_dataset

# 設定圖表風格與論文截圖一致
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 12,
    'lines.linewidth': 2
})

NUM_TRIALS = 30 # 根據您實際跑的次數調整

def get_predictions(scenario_dir, model_type, config, t_tensor):
    """讀取 30 個 seed 的權重並回傳所有預測結果的陣列"""
    preds = []
    for seed in range(1, NUM_TRIALS + 1):
        seed_dir = os.path.join(scenario_dir, f"seed_{seed}")
        
        try:
            model = PedagogicalBaselineComaprison(config).to(config.device)
            
            if model_type == "corrector":
                ckpt_model = os.path.join(seed_dir, "finetuned", "final_model.pt")
                ckpt_corr = os.path.join(seed_dir, "corrector", "final_corrector.pt")
                corrector = Corrector(config).to(config.device)
                
                if os.path.exists(ckpt_model) and os.path.exists(ckpt_corr):
                    model.load_state_dict(torch.load(ckpt_model, map_location=config.device, weights_only=True)["model_state_dict"])
                    corrector.load_state_dict(torch.load(ckpt_corr, map_location=config.device, weights_only=True)["model_state_dict"])
                    model.eval(); corrector.eval()
                    with torch.no_grad():
                        u_pred = model(t_tensor)
                        s_pred = corrector(u_pred).cpu().numpy().ravel()
                    preds.append(s_pred)
            else:
                sub_path = "pretrained/best_model.pt" if model_type == "pretrain" else "finetuned/final_model.pt"
                ckpt = os.path.join(seed_dir, sub_path)
                if os.path.exists(ckpt):
                    model.load_state_dict(torch.load(ckpt, map_location=config.device, weights_only=True)["model_state_dict"])
                    model.eval()
                    with torch.no_grad():
                        pred = model(t_tensor).cpu().numpy().ravel()
                    preds.append(pred)
        except Exception as e:
            continue # 忽略讀取失敗的 seed
            
    return np.array(preds)

def plot_with_std(t, exact, preds, title, ylabel, save_name):
    """繪製 Reference, 平均值(紅虛線), 以及 +- 2 std 陰影帶(藍色)"""
    if len(preds) == 0:
        print(f"[警告] 找不到 {save_name} 的預測資料，跳過繪圖。")
        return

    mean_pred = np.mean(preds, axis=0)
    std_pred = np.std(preds, axis=0)
    
    plt.figure(figsize=(7, 5))
    
    # 真實解 (Exact)
    plt.plot(t, exact, 'k-', label='Reference')
    
    # 預測平均值 (Prediction)
    plt.plot(t, mean_pred, 'r--', label='Prediction')
    
    # +- 2 STD 陰影帶 (± 2 std)
    plt.fill_between(t, mean_pred - 2 * std_pred, mean_pred + 2 * std_pred, 
                     color='blue', alpha=0.3, label=r'$\pm 2$ std')
    
    plt.title(title)
    plt.xlabel(r'$t$')
    plt.ylabel(ylabel)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"[繪圖完成] {save_name} (基於 {len(preds)} 個 seeds)")

def main():
    print("========== 開始處理 30 次訓練的數據並繪製 STD 陰影圖 ==========")
    config = get_config()
    device = config.device
    os.makedirs("paper_plots", exist_ok=True)

    # 1. 產生高解析度的真值 (Ground Truth)
    params = config.system_pedagogical.system_params.to_dict()
    params['noise'] = 0.0 
    _, _, _, sol = generate_reaction_ode_dataset(params, T=params['T'], u0=params['u0'], n_t=params['n_t'])
    
    t_np = sol.t
    t_tensor = torch.tensor(t_np, dtype=torch.float32).reshape(-1, 1).to(device)
    u_true = sol.y[0]
    
    # =========================================================================
    # 【新增】：產生 Pretrain 專屬的 Ground Truth (只包含已知物理)
    # 透過將 lambda 設為 0，強迫系統只依據 du/dt = sin(3*pi*t) 產生軌跡
    # =========================================================================
    # params_pretrain = params.copy()
    # params_pretrain['lambda'] = 0.0
    # _, _, _, sol_pretrain = generate_reaction_ode_dataset(params_pretrain, T=params['T'], u0=params['u0'], n_t=params['n_t'])
    # u_pretrain_true = sol_pretrain.y[0]

    from scipy.integrate import solve_ivp
    
    def pretrain_ode(t, u):
        return np.sin(3 * np.pi * t) + params['lambda'] * np.cos(u)
        
    sol_pretrain = solve_ivp(pretrain_ode, [0.0, params['T']], [params['u0']], t_eval=t_np, rtol=1e-9, atol=1e-9)
    u_pretrain_true = sol_pretrain.y[0]
    # =========================================================================
    # s_true 定義
    # 真實缺失項 = 真實物理 (lambda*u*(1-u)) - 錯誤的物理 (0)
    # =========================================================================
    # s_true = params['lambda'] * (u_true * (1 - u_true))
    s_true = params['lambda'] * (u_true * (1 - u_true)) - params['lambda'] * np.cos(u_true)

    # ---------------------------------------------------------
    # 4.1 Correctly specified governing equation
    os.environ["USE_CORRECT_PHYSICS"] = "1"
    preds_A = get_predictions("results_scenario_A", "finetune", config, t_tensor)
    plot_with_std(t_np, u_true, preds_A, "Correctly specified governing equation", r"$u(t)$", "paper_plots/4_1_correct_physics.png")

    # ---------------------------------------------------------
    # 4.2 Misspecified equation omitting the reaction term
    os.environ["USE_CORRECT_PHYSICS"] = "0"
    preds_B = get_predictions("results_scenario_B", "finetune", config, t_tensor)
    plot_with_std(t_np, u_true, preds_B, "Misspecified equation", r"$u(t)$", "paper_plots/4_2_misspecified_physics.png")

    # ---------------------------------------------------------
    # 4.3 DAPINNs with ADPC reconstructing the damping term
    preds_C_fine = get_predictions("results_scenario_C", "finetune", config, t_tensor)
    plot_with_std(t_np, u_true, preds_C_fine, "DAPINNs with ADPC", r"$u(t)$", "paper_plots/4_3_dapinn_adpc.png")

    # ---------------------------------------------------------
    # 4.4 Pretrained model
    # 【修改】：這裡把原本的 u_true 換成了剛才算出來的 u_pretrain_true
    preds_C_pre = get_predictions("results_scenario_C", "pretrain", config, t_tensor)
    plot_with_std(t_np, u_pretrain_true, preds_C_pre, "Pretrained model", r"$u(t)$", "paper_plots/4_4_pretrained_model.png")

    # ---------------------------------------------------------
    # 4.5 ADPC compared with missing term
    preds_C_corr = get_predictions("results_scenario_C", "corrector", config, t_tensor)
    plot_with_std(t_np, s_true, preds_C_corr, "ADPC compared with missing term", r"$s_{\psi}(t)$", "paper_plots/4_5_adpc_missing_term.png")

    print("========== 實驗圖表生成完畢，請查看 paper_plots 目錄 ==========")

if __name__ == "__main__":
    main()