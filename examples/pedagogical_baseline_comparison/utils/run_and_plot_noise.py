# import os
# import subprocess
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# from examples.pedagogical_baseline_comparison.configs.default import get_config
# from examples.pedagogical_baseline_comparison.models import PedagogicalBaselineComaprison, Corrector
# from examples.pedagogical_baseline_comparison.utils import generate_reaction_ode_dataset

# # ========================================================
# # 1. 全局設定與美化
# # ========================================================
# sns.set_style("whitegrid")
# plt.rcParams.update({
#     'font.size': 12,
#     'axes.labelsize': 14,
#     'axes.titlesize': 14,
#     'legend.fontsize': 11,
#     'xtick.labelsize': 12,
#     'ytick.labelsize': 12,
#     'lines.linewidth': 2
# })

# # 定義實驗參數
# N_LIST = [100, 500]
# NOISE_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.10]
# NOISE_LABELS = ['1%', '2%', '3%', '5%', '10%']
# NUM_TRIALS = 30  # 建議測試時先改為 2，確認能跑再改成 30
# RESULTS_DIR = "results_noise"

# # ========================================================
# # 2. 自動化執行訓練腳本
# # ========================================================
# def run_noise_experiments():
#     print("========== 啟動雜訊強健性 (Noise Robustness) 自動化實驗 ==========")
#     env = os.environ.copy()
#     env["USE_CORRECT_PHYSICS"] = "0"  # 確保使用錯誤的物理假設

#     for N in N_LIST:
#         for noise in NOISE_LEVELS:
#             print(f"\n" + "="*50)
#             print(f" 開始執行 N = {N}, Noise = {noise} 的實驗 ")
#             print("="*50)

#             for seed in range(1, NUM_TRIALS + 1):
#                 save_subdir = os.path.join(RESULTS_DIR, f"N_{N}", f"noise_{noise}", f"seed_{seed}")
                
#                 if os.path.exists(os.path.join(save_subdir, "evaluation_results.json")):
#                     print(f"  -> [Skip] N={N}, Noise={noise}, Seed={seed} 已存在，跳過訓練。")
#                     continue
                    
#                 print(f"\n[N={N} | Noise={noise} | Seed={seed}] 執行中...")

#                 base_cmd = [
#                     "python", "-m", "examples.pedagogical_baseline_comparison.main",
#                     f"--seed={seed}",
#                     f"--save_subdir={save_subdir}",
#                     "--use_corrector=True",
#                     "--run_pretrain=True",
#                     "--load_pretrained=True",
#                     "--run_finetune=True",
#                     f"--sample_size={N}",
#                     f"--noise={noise}"
#                 ]

#                 try:
#                     subprocess.run(base_cmd + ["--mode=train"], env=env, check=True)
#                     subprocess.run(base_cmd + ["--mode=eval"], env=env, check=True)
#                 except subprocess.CalledProcessError as e:
#                     print(f"  -> [錯誤] 執行失敗: {e}")
#                     continue

# # ========================================================
# # 3. 從 Checkpoints 計算與提取軌跡
# # ========================================================
# def compute_metrics_from_checkpoints(config, t_tensor, u_true, s_true):
#     records = []
#     print("\n========== 正在從模型權重重新計算精確的 L2 Error (約需幾十秒) ==========")
    
#     for N in N_LIST:
#         for noise in NOISE_LEVELS:
#             for seed in range(1, NUM_TRIALS + 1):
#                 seed_dir = os.path.join(RESULTS_DIR, f"N_{N}", f"noise_{noise}", f"seed_{seed}")
#                 ckpt_model = os.path.join(seed_dir, "finetuned", "final_model.pt")
#                 ckpt_corr = os.path.join(seed_dir, "corrector", "final_corrector.pt")
                
#                 if not (os.path.exists(ckpt_model) and os.path.exists(ckpt_corr)):
#                     continue
                    
#                 try:
#                     model = PedagogicalBaselineComaprison(config).to(config.device)
#                     corrector = Corrector(config).to(config.device)
                    
#                     model.load_state_dict(torch.load(ckpt_model, map_location=config.device, weights_only=True)["model_state_dict"])
#                     corrector.load_state_dict(torch.load(ckpt_corr, map_location=config.device, weights_only=True)["model_state_dict"])
#                     model.eval(); corrector.eval()
                    
#                     with torch.no_grad():
#                         u_pred_tensor = model(t_tensor)
#                         s_pred_tensor = corrector(u_pred_tensor)
                        
#                     u_pred = u_pred_tensor.cpu().numpy().ravel()
#                     s_pred = s_pred_tensor.cpu().numpy().ravel()
                    
#                     u_err = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
#                     s_err = np.linalg.norm(s_pred - s_true) / np.linalg.norm(s_true)
                    
#                     records.append({"Measurement Points (N)": N, "Noise Level": noise, "Seed": seed, "u_err": u_err, "s_err": s_err})
#                 except Exception:
#                     pass
                    
#     return pd.DataFrame(records).dropna()

# def get_noise_time_series_predictions(N, noise_level, is_corrector, config, t_tensor):
#     """提取特定 N 與 Noise 下，所有 Seed 的時間序列軌跡"""
#     preds = []
#     for seed in range(1, NUM_TRIALS + 1):
#         seed_dir = os.path.join(RESULTS_DIR, f"N_{N}", f"noise_{noise_level}", f"seed_{seed}")
#         ckpt_model = os.path.join(seed_dir, "finetuned", "final_model.pt")
#         ckpt_corr = os.path.join(seed_dir, "corrector", "final_corrector.pt")

#         try:
#             model = PedagogicalBaselineComaprison(config).to(config.device)
#             if is_corrector:
#                 if os.path.exists(ckpt_model) and os.path.exists(ckpt_corr):
#                     corrector = Corrector(config).to(config.device)
#                     model.load_state_dict(torch.load(ckpt_model, map_location=config.device, weights_only=True)["model_state_dict"])
#                     corrector.load_state_dict(torch.load(ckpt_corr, map_location=config.device, weights_only=True)["model_state_dict"])
#                     model.eval(); corrector.eval()
#                     with torch.no_grad():
#                         preds.append(corrector(model(t_tensor)).cpu().numpy().ravel())
#             else:
#                 if os.path.exists(ckpt_model):
#                     model.load_state_dict(torch.load(ckpt_model, map_location=config.device, weights_only=True)["model_state_dict"])
#                     model.eval()
#                     with torch.no_grad():
#                         preds.append(model(t_tensor).cpu().numpy().ravel())
#         except Exception:
#             pass
#     return np.array(preds)


# # ========================================================
# # 4. 繪圖函數
# # ========================================================
# def plot_noise_robustness(df, y_col, title, ylabel, save_name):
#     """繪製 Error vs Noise 折線圖"""
#     plt.figure(figsize=(8, 6))
#     x_vals = np.array([int(n * 100) for n in NOISE_LEVELS])
    
#     plot_configs = {
#         100: {'color': 'red', 'label_mean': '100 pts $L_2$ error (mean)', 'label_std': '100 pts $\pm 2$ Std Dev'},
#         500: {'color': 'blue', 'label_mean': '500 pts $L_2$ error (mean)', 'label_std': '500 pts $\pm 2$ Std Dev'}
#     }
    
#     for N in N_LIST:
#         subset = df[df["Measurement Points (N)"] == N]
#         if subset.empty: continue
            
#         stats = subset.groupby("Noise Level")[y_col].agg(['mean', 'std']).reset_index()
#         stats['Noise Level'] = pd.Categorical(stats['Noise Level'], categories=NOISE_LEVELS, ordered=True)
#         stats = stats.sort_values('Noise Level')
        
#         mean_vals = stats["mean"].values
#         std_vals = stats["std"].values
#         cfg = plot_configs[N]
        
#         lower_bound = np.clip(mean_vals - 2 * std_vals, 1e-6, None)
#         upper_bound = mean_vals + 2 * std_vals
        
#         plt.fill_between(x_vals, lower_bound, upper_bound, color=cfg['color'], alpha=0.15, label=cfg['label_std'])
#         plt.plot(x_vals, mean_vals, color=cfg['color'], marker='o', linewidth=2, label=cfg['label_mean'])

#     plt.yscale("log")
#     plt.xticks(x_vals, labels=NOISE_LABELS)
#     plt.gca().xaxis.grid(False)
    
#     plt.title(title)
#     plt.ylabel(ylabel)
#     plt.xlabel("Noise Levels")
#     plt.legend(loc="upper left", framealpha=0.9)
#     plt.tight_layout()
#     plt.savefig(save_name, dpi=300)
#     plt.close()
#     print(f"[繪圖完成] {save_name}")

# def plot_time_series_comparison(t, exact, preds_100, preds_500, title, ylabel, save_name):
#     """繪製特定雜訊下，100 pts 與 500 pts 的預測軌跡對比"""
#     plt.figure(figsize=(8, 6))
#     plt.plot(t, exact, 'k-', linewidth=2.5, label='Reference')
    
#     # 繪製 100 pts 的結果 (紅色系)
#     if len(preds_100) > 0:
#         mean_100 = np.mean(preds_100, axis=0)
#         std_100 = np.std(preds_100, axis=0)
#         plt.fill_between(t, mean_100 - 2*std_100, mean_100 + 2*std_100, color='red', alpha=0.2, label=r'100 pts ($\pm 2$ Std Dev)')
#         plt.plot(t, mean_100, 'r--', linewidth=2, label='100 pts (Mean)')

#     # 繪製 500 pts 的結果 (藍色系)
#     if len(preds_500) > 0:
#         mean_500 = np.mean(preds_500, axis=0)
#         std_500 = np.std(preds_500, axis=0)
#         plt.fill_between(t, mean_500 - 2*std_500, mean_500 + 2*std_500, color='blue', alpha=0.3, label=r'500 pts ($\pm 2$ Std Dev)')
#         plt.plot(t, mean_500, 'b--', linewidth=2, label='500 pts (Mean)')

#     plt.title(title)
#     plt.xlabel("Time ($t$)")
#     plt.ylabel(ylabel)
#     plt.legend(loc="upper right", framealpha=0.9)
#     plt.tight_layout()
#     plt.savefig(save_name, dpi=300)
#     plt.close()
#     print(f"[繪圖完成] {save_name}")

# # ========================================================
# # 5. 主流程
# # ========================================================
# def main():
#     os.makedirs("paper_plots", exist_ok=True)
    
#     # 1. 執行所有實驗 (自動跳過已完成的 Seed)
#     run_noise_experiments()
    
#     config = get_config()
#     device = config.device
    
#     # 2. 準備 Ground Truth (無雜訊)
#     params = config.system_pedagogical.system_params.to_dict()
#     params['noise'] = 0.0
#     _, _, _, sol = generate_reaction_ode_dataset(params, T=params['T'], u0=params['u0'], n_t=params['n_t'])
#     t_np = sol.t
#     t_tensor = torch.tensor(t_np, dtype=torch.float32).reshape(-1, 1).to(device)
#     u_true = sol.y[0]
    
#     # 【關鍵】物理缺失項包含錯誤的 cos(u)
#     s_true = params['lambda'] * (u_true * (1 - u_true)) - params['lambda'] * np.cos(u_true)

#     # 3. 計算誤差並畫出 2 張折線圖
#     df_metrics = compute_metrics_from_checkpoints(config, t_tensor, u_true, s_true)
    
#     if df_metrics.empty:
#         print("[警告] 找不到任何實驗權重，無法繪圖。請確認是否成功跑完。")
#         return

#     print("\n========== 開始繪製 4 張雜訊強健性圖表 ==========")
    
#     plot_noise_robustness(
#         df_metrics, "u_err", 
#         "PINNs L2 error Noise Levels under Limited Data Regimes", 
#         "Relative $L_2$ Error of State $\\tilde{x}(t)$", 
#         "paper_plots/fig_noise_state_error_vs_noise.png"
#     )

#     plot_noise_robustness(
#         df_metrics, "s_err", 
#         "Corrector error Noise Levels under Limited Data Regimes", 
#         "Relative $L_2$ Error of Corrector $s_{\\psi}(t)$", 
#         "paper_plots/fig_noise_corrector_error_vs_noise.png"
#     )
    
#     # =======================================================
#     # 【新增】畫出 10% 雜訊下的時間序列軌跡比較 (100 pts vs 500 pts)
#     # =======================================================
#     target_noise = 0.10  # 選擇 10% 雜訊作為最具代表性的軌跡比較
    
#     # 提取 DAPINNs 狀態 u(t) 的軌跡
#     preds_u_100 = get_noise_time_series_predictions(100, target_noise, False, config, t_tensor)
#     preds_u_500 = get_noise_time_series_predictions(500, target_noise, False, config, t_tensor)
    
#     plot_time_series_comparison(
#         t_np, u_true, preds_u_100, preds_u_500, 
#         f"DAPINNs Prediction under 10% Noise", "$u(t)$", 
#         "paper_plots/fig_noise_dapinn_prediction_10percent.png"
#     )

#     # 提取 ADPC 修正項 s_psi(t) 的軌跡
#     preds_s_100 = get_noise_time_series_predictions(100, target_noise, True, config, t_tensor)
#     preds_s_500 = get_noise_time_series_predictions(500, target_noise, True, config, t_tensor)
    
#     plot_time_series_comparison(
#         t_np, s_true, preds_s_100, preds_s_500, 
#         f"ADPC Prediction under 10% Noise", "$s_{\psi}(t)$", 
#         "paper_plots/fig_noise_adpc_prediction_10percent.png"
#     )
    
#     print("\n========== 實驗與繪圖全部完成！請查看 paper_plots 目錄 ==========")

# if __name__ == "__main__":
#     main()



import os
import subprocess
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from examples.pedagogical_baseline_comparison.configs.default import get_config
from examples.pedagogical_baseline_comparison.models import PedagogicalBaselineComaprison, Corrector
from examples.pedagogical_baseline_comparison.utils import generate_reaction_ode_dataset

# ========================================================
# 1. 全局設定與美化
# ========================================================
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'lines.linewidth': 2
})

# 定義實驗參數
N_LIST = [100, 500]
NOISE_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.10]
NOISE_LABELS = ['1%', '2%', '3%', '5%', '10%']
NUM_TRIALS = 30  # 建議測試時先改為 2，確認能跑再改成 30

# 【關鍵修改 1】更改資料夾名稱，避免覆蓋有 cos 版本的實驗結果
RESULTS_DIR = "results_noise_without_cos"

# ========================================================
# 2. 自動化執行訓練腳本
# ========================================================
def run_noise_experiments():
    print("========== 啟動雜訊強健性 (Noise Robustness) 自動化實驗 (Without cos) ==========")
    env = os.environ.copy()
    env["USE_CORRECT_PHYSICS"] = "0"  # 確保使用錯誤的物理假設 (請確認 models.py 中已移除 cos)

    for N in N_LIST:
        for noise in NOISE_LEVELS:
            print(f"\n" + "="*50)
            print(f" 開始執行 N = {N}, Noise = {noise} 的實驗 ")
            print("="*50)

            for seed in range(1, NUM_TRIALS + 1):
                save_subdir = os.path.join(RESULTS_DIR, f"N_{N}", f"noise_{noise}", f"seed_{seed}")
                
                if os.path.exists(os.path.join(save_subdir, "evaluation_results.json")):
                    print(f"  -> [Skip] N={N}, Noise={noise}, Seed={seed} 已存在，跳過訓練。")
                    continue
                    
                print(f"\n[N={N} | Noise={noise} | Seed={seed}] 執行中...")

                base_cmd = [
                    "python", "-m", "examples.pedagogical_baseline_comparison.main",
                    f"--seed={seed}",
                    f"--save_subdir={save_subdir}",
                    "--use_corrector=True",
                    "--run_pretrain=True",
                    "--load_pretrained=True",
                    "--run_finetune=True",
                    f"--sample_size={N}",
                    f"--noise={noise}"
                ]

                try:
                    subprocess.run(base_cmd + ["--mode=train"], env=env, check=True)
                    subprocess.run(base_cmd + ["--mode=eval"], env=env, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"  -> [錯誤] 執行失敗: {e}")
                    continue

# ========================================================
# 3. 從 Checkpoints 計算與提取軌跡
# ========================================================
def compute_metrics_from_checkpoints(config, t_tensor, u_true, s_true):
    records = []
    print("\n========== 正在從模型權重重新計算精確的 L2 Error (約需幾十秒) ==========")
    
    for N in N_LIST:
        for noise in NOISE_LEVELS:
            for seed in range(1, NUM_TRIALS + 1):
                seed_dir = os.path.join(RESULTS_DIR, f"N_{N}", f"noise_{noise}", f"seed_{seed}")
                ckpt_model = os.path.join(seed_dir, "finetuned", "final_model.pt")
                ckpt_corr = os.path.join(seed_dir, "corrector", "final_corrector.pt")
                
                if not (os.path.exists(ckpt_model) and os.path.exists(ckpt_corr)):
                    continue
                    
                try:
                    model = PedagogicalBaselineComaprison(config).to(config.device)
                    corrector = Corrector(config).to(config.device)
                    
                    model.load_state_dict(torch.load(ckpt_model, map_location=config.device, weights_only=True)["model_state_dict"])
                    corrector.load_state_dict(torch.load(ckpt_corr, map_location=config.device, weights_only=True)["model_state_dict"])
                    model.eval(); corrector.eval()
                    
                    with torch.no_grad():
                        u_pred_tensor = model(t_tensor)
                        s_pred_tensor = corrector(u_pred_tensor)
                        
                    u_pred = u_pred_tensor.cpu().numpy().ravel()
                    s_pred = s_pred_tensor.cpu().numpy().ravel()
                    
                    u_err = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
                    s_err = np.linalg.norm(s_pred - s_true) / np.linalg.norm(s_true)
                    
                    records.append({"Measurement Points (N)": N, "Noise Level": noise, "Seed": seed, "u_err": u_err, "s_err": s_err})
                except Exception:
                    pass
                    
    return pd.DataFrame(records).dropna()

def get_noise_time_series_predictions(N, noise_level, is_corrector, config, t_tensor):
    """提取特定 N 與 Noise 下，所有 Seed 的時間序列軌跡"""
    preds = []
    for seed in range(1, NUM_TRIALS + 1):
        seed_dir = os.path.join(RESULTS_DIR, f"N_{N}", f"noise_{noise_level}", f"seed_{seed}")
        ckpt_model = os.path.join(seed_dir, "finetuned", "final_model.pt")
        ckpt_corr = os.path.join(seed_dir, "corrector", "final_corrector.pt")

        try:
            model = PedagogicalBaselineComaprison(config).to(config.device)
            if is_corrector:
                if os.path.exists(ckpt_model) and os.path.exists(ckpt_corr):
                    corrector = Corrector(config).to(config.device)
                    model.load_state_dict(torch.load(ckpt_model, map_location=config.device, weights_only=True)["model_state_dict"])
                    corrector.load_state_dict(torch.load(ckpt_corr, map_location=config.device, weights_only=True)["model_state_dict"])
                    model.eval(); corrector.eval()
                    with torch.no_grad():
                        preds.append(corrector(model(t_tensor)).cpu().numpy().ravel())
            else:
                if os.path.exists(ckpt_model):
                    model.load_state_dict(torch.load(ckpt_model, map_location=config.device, weights_only=True)["model_state_dict"])
                    model.eval()
                    with torch.no_grad():
                        preds.append(model(t_tensor).cpu().numpy().ravel())
        except Exception:
            pass
    return np.array(preds)


# ========================================================
# 4. 繪圖函數
# ========================================================
def plot_noise_robustness(df, y_col, title, ylabel, save_name):
    """繪製 Error vs Noise 折線圖"""
    plt.figure(figsize=(8, 6))
    x_vals = np.array([int(n * 100) for n in NOISE_LEVELS])
    
    plot_configs = {
        100: {'color': 'red', 'label_mean': '100 pts $L_2$ error (mean)', 'label_std': '100 pts $\pm 2$ Std Dev'},
        500: {'color': 'blue', 'label_mean': '500 pts $L_2$ error (mean)', 'label_std': '500 pts $\pm 2$ Std Dev'}
    }
    
    for N in N_LIST:
        subset = df[df["Measurement Points (N)"] == N]
        if subset.empty: continue
            
        stats = subset.groupby("Noise Level")[y_col].agg(['mean', 'std']).reset_index()
        stats['Noise Level'] = pd.Categorical(stats['Noise Level'], categories=NOISE_LEVELS, ordered=True)
        stats = stats.sort_values('Noise Level')
        
        mean_vals = stats["mean"].values
        std_vals = stats["std"].values
        cfg = plot_configs[N]
        
        lower_bound = np.clip(mean_vals - 2 * std_vals, 1e-6, None)
        upper_bound = mean_vals + 2 * std_vals
        
        plt.fill_between(x_vals, lower_bound, upper_bound, color=cfg['color'], alpha=0.15, label=cfg['label_std'])
        plt.plot(x_vals, mean_vals, color=cfg['color'], marker='o', linewidth=2, label=cfg['label_mean'])

    plt.yscale("log")
    plt.xticks(x_vals, labels=NOISE_LABELS)
    plt.gca().xaxis.grid(False)
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Noise Levels")
    plt.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"[繪圖完成] {save_name}")

def plot_time_series_comparison(t, exact, preds_100, preds_500, title, ylabel, save_name):
    """繪製特定雜訊下，100 pts 與 500 pts 的預測軌跡對比"""
    plt.figure(figsize=(8, 6))
    plt.plot(t, exact, 'k-', linewidth=2.5, label='Reference')
    
    # 繪製 100 pts 的結果 (紅色系)
    if len(preds_100) > 0:
        mean_100 = np.mean(preds_100, axis=0)
        std_100 = np.std(preds_100, axis=0)
        plt.fill_between(t, mean_100 - 2*std_100, mean_100 + 2*std_100, color='red', alpha=0.2, label=r'100 pts ($\pm 2$ Std Dev)')
        plt.plot(t, mean_100, 'r--', linewidth=2, label='100 pts (Mean)')

    # 繪製 500 pts 的結果 (藍色系)
    if len(preds_500) > 0:
        mean_500 = np.mean(preds_500, axis=0)
        std_500 = np.std(preds_500, axis=0)
        plt.fill_between(t, mean_500 - 2*std_500, mean_500 + 2*std_500, color='blue', alpha=0.3, label=r'500 pts ($\pm 2$ Std Dev)')
        plt.plot(t, mean_500, 'b--', linewidth=2, label='500 pts (Mean)')

    plt.title(title)
    plt.xlabel("Time ($t$)")
    plt.ylabel(ylabel)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"[繪圖完成] {save_name}")

# ========================================================
# 5. 主流程
# ========================================================
def main():
    os.makedirs("paper_plots", exist_ok=True)
    
    # 1. 執行所有實驗 (自動跳過已完成的 Seed)
    run_noise_experiments()
    
    config = get_config()
    device = config.device
    
    # 2. 準備 Ground Truth (無雜訊)
    params = config.system_pedagogical.system_params.to_dict()
    params['noise'] = 0.0
    _, _, _, sol = generate_reaction_ode_dataset(params, T=params['T'], u0=params['u0'], n_t=params['n_t'])
    t_np = sol.t
    t_tensor = torch.tensor(t_np, dtype=torch.float32).reshape(-1, 1).to(device)
    u_true = sol.y[0]
    
    # 【關鍵修改 2】物理缺失項完全省略反應項 (Without cos)
    s_true = params['lambda'] * (u_true * (1 - u_true))

    # 3. 計算誤差並畫出 2 張折線圖
    df_metrics = compute_metrics_from_checkpoints(config, t_tensor, u_true, s_true)
    
    if df_metrics.empty:
        print("[警告] 找不到任何實驗權重，無法繪圖。請確認是否成功跑完。")
        return

    print("\n========== 開始繪製 4 張雜訊強健性圖表 (Without cos) ==========")
    
    plot_noise_robustness(
        df_metrics, "u_err", 
        "PINNs L2 error Noise Levels under Limited Data Regimes", 
        "Relative $L_2$ Error of State $\\tilde{x}(t)$", 
        "paper_plots/fig_noise_state_error_vs_noise_nocos.png"
    )

    plot_noise_robustness(
        df_metrics, "s_err", 
        "Corrector error Noise Levels under Limited Data Regimes", 
        "Relative $L_2$ Error of Corrector $s_{\\psi}(t)$", 
        "paper_plots/fig_noise_corrector_error_vs_noise_nocos.png"
    )
    
    # =======================================================
    # 畫出 10% 雜訊下的時間序列軌跡比較 (100 pts vs 500 pts)
    # =======================================================
    target_noise = 0.10  # 選擇 10% 雜訊作為最具代表性的軌跡比較
    
    # 提取 DAPINNs 狀態 u(t) 的軌跡
    preds_u_100 = get_noise_time_series_predictions(100, target_noise, False, config, t_tensor)
    preds_u_500 = get_noise_time_series_predictions(500, target_noise, False, config, t_tensor)
    
    plot_time_series_comparison(
        t_np, u_true, preds_u_100, preds_u_500, 
        f"DAPINNs Prediction under 10% Noise", "$u(t)$", 
        "paper_plots/fig_noise_dapinn_prediction_10percent_nocos.png"
    )

    # 提取 ADPC 修正項 s_psi(t) 的軌跡
    preds_s_100 = get_noise_time_series_predictions(100, target_noise, True, config, t_tensor)
    preds_s_500 = get_noise_time_series_predictions(500, target_noise, True, config, t_tensor)
    
    plot_time_series_comparison(
        t_np, s_true, preds_s_100, preds_s_500, 
        f"ADPC Prediction under 10% Noise", "$s_{\psi}(t)$", 
        "paper_plots/fig_noise_adpc_prediction_10percent_nocos.png"
    )
    
    print("\n========== 實驗與繪圖全部完成！請查看 paper_plots 目錄 ==========")

if __name__ == "__main__":
    main()