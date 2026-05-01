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
RESULTS_DIR = "results_noise"

# ========================================================
# 2. 自動化執行訓練腳本
# ========================================================
def run_noise_experiments():
    print("========== 啟動雜訊強健性 (Noise Robustness) 自動化實驗 ==========")
    env = os.environ.copy()
    env["USE_CORRECT_PHYSICS"] = "0"  # 確保使用錯誤的物理假設 (含 cos)

    for N in N_LIST:
        for noise in NOISE_LEVELS:
            print(f"\n" + "="*50)
            print(f" 開始執行 N = {N}, Noise = {noise} 的實驗 ")
            print("="*50)

            for seed in range(1, NUM_TRIALS + 1):
                save_subdir = os.path.join(RESULTS_DIR, f"N_{N}", f"noise_{noise}", f"seed_{seed}")
                
                # 防呆：避免重複執行已完成的實驗 (自動續傳)
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
# 3. 讀取 Checkpoint 計算正確誤差 (避開 JSON 基準問題)
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
                    
                    # 評估必須對齊「無雜訊」的乾淨真值
                    u_err = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
                    s_err = np.linalg.norm(s_pred - s_true) / np.linalg.norm(s_true)
                    
                    records.append({
                        "Measurement Points (N)": N, 
                        "Noise Level": noise,
                        "Seed": seed, 
                        "u_err": u_err, 
                        "s_err": s_err
                    })
                except Exception as e:
                    pass
                    
    return pd.DataFrame(records).dropna()

# ========================================================
# 4. 繪圖函數
# ========================================================
def plot_noise_robustness(df, y_col, title, ylabel, save_name):
    plt.figure(figsize=(8, 6))
    
    # 映射到 1, 2, 3, 5, 10，讓 X 軸間距呈現真實的雜訊比例跨度
    x_vals = np.array([int(n * 100) for n in NOISE_LEVELS])
    
    plot_configs = {
        100: {'color': 'red', 'label_mean': '100 pts L2 error(mean)', 'label_std': r'100 pts $\pm 2$ Std Dev'},
        500: {'color': 'blue', 'label_mean': '500 pts L2 error(mean)', 'label_std': r'500 pts $\pm 2$ Std Dev'}
    }
    
    for N in N_LIST:
        subset = df[df["Measurement Points (N)"] == N]
        if subset.empty: continue
            
        stats = subset.groupby("Noise Level")[y_col].agg(['mean', 'std']).reset_index()
        # 確保順序正確
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

# ========================================================
# 5. 主流程
# ========================================================
def main():
    os.makedirs("paper_plots", exist_ok=True)
    
    # 1. 執行所有實驗 (自動跳過已完成的 Seed)
    # run_noise_experiments()
    
    config = get_config()
    device = config.device
    
    # 2. 準備 Ground Truth (無雜訊)
    params = config.system_pedagogical.system_params.to_dict()
    params['noise'] = 0.0
    _, _, _, sol = generate_reaction_ode_dataset(params, T=params['T'], u0=params['u0'], n_t=params['n_t'])
    t_np = sol.t
    t_tensor = torch.tensor(t_np, dtype=torch.float32).reshape(-1, 1).to(device)
    u_true = sol.y[0]
    
    # 使用 Misspecified as sin(3*pi*t) + lambda*cos(u) 版本
    s_true = params['lambda'] * (u_true * (1 - u_true)) - params['lambda'] * np.cos(u_true)

    # 3. 計算誤差
    df_metrics = compute_metrics_from_checkpoints(config, t_tensor, u_true, s_true)
    
    if df_metrics.empty:
        print("[警告] 找不到任何實驗權重，無法繪圖。請確認是否成功跑完。")
        return

    # 4. 畫圖
    print("\n========== 開始繪製 2 張雜訊強健性圖表 ==========")
    plot_noise_robustness(
        df_metrics, 
        "u_err", 
        "PINNs L2 error Noise Levels under Limited Data Regimes", 
        "Relative $L_2$ Error of State $\\tilde{x}(t)$", 
        "paper_plots/noise_robustness_state_error.png"
    )

    plot_noise_robustness(
        df_metrics, 
        "s_err", 
        "Corrector error Noise Levels under Limited Data Regimes", 
        "Relative $L_2$ Error of Corrector $s_{\\psi}(t)$", 
        "paper_plots/noise_robustness_corrector_error.png"
    )
    
    print("\n========== 實驗與繪圖全部完成！請查看 paper_plots 目錄 ==========")

if __name__ == "__main__":
    main()