import os
import json
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

N_LIST = [10, 15, 20, 30, 100, 1000]
NUM_TRIALS = 30
RESULTS_DIR = "results_scarcity"


# ========================================================
# 2. 核心：直接從模型權重計算正確的 L2 Error (取代讀取 JSON)
# ========================================================
def compute_metrics_from_checkpoints(config, t_tensor, u_true, s_true):
    """直接讀取所有權重，即時計算正確的 L2 Error，避免 JSON 基準錯誤的問題"""
    records = []
    print("正在從模型權重重新計算精確的 L2 Error (這會花費幾十秒時間)...")
    
    for N in N_LIST:
        for seed in range(1, NUM_TRIALS + 1):
            seed_dir = os.path.join(RESULTS_DIR, f"N_{N}", f"seed_{seed}")
            ckpt_model = os.path.join(seed_dir, "finetuned", "final_model.pt")
            ckpt_corr = os.path.join(seed_dir, "corrector", "final_corrector.pt")
            
            if not (os.path.exists(ckpt_model) and os.path.exists(ckpt_corr)):
                continue
                
            try:
                # 載入模型
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
                
                # 手動計算正確的相對 L2 誤差
                u_err = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
                s_err = np.linalg.norm(s_pred - s_true) / np.linalg.norm(s_true)
                
                records.append({"N": N, "Seed": seed, "u_err": u_err, "s_err": s_err})
            except Exception as e:
                pass
                
    return pd.DataFrame(records).dropna()

def print_scarcity_table(df):
    """在終端機印出不同 N 下的 DAPINN 與 ADPC 誤差表格"""
    print("\n" + "="*110)
    print(f"{'Measurement Points':<30}" + "".join([f"{str(N):>13}" for N in N_LIST]))
    print("-" * 110)

    def format_val(mean, std):
        if pd.isna(mean) or pd.isna(std):
            return f"{'N/A':>13}"
        return f"{mean:.3e}({std:.3e})".rjust(17)

    dapinn_row = f"{'DAPINNs (Relative L2 Error)':<30}"
    adpc_row = f"{'ADPC (Relative L2 Error)':<30}"
    
    for N in N_LIST:
        subset = df[df["N"] == N]
        if not subset.empty:
            u_mean, u_std = subset["u_err"].mean(), subset["u_err"].std()
            s_mean, s_std = subset["s_err"].mean(), subset["s_err"].std()
        else:
            u_mean, u_std, s_mean, s_std = np.nan, np.nan, np.nan, np.nan
            
        dapinn_row += format_val(u_mean, u_std)
        adpc_row += format_val(s_mean, s_std)
        
    print(dapinn_row)
    print(adpc_row)
    print("="*110 + "\n")


def get_time_series_predictions(N, model_type, config, t_tensor):
    """讀取特定 N 下所有 Seed 的權重，回傳隨時間變化的預測軌跡"""
    preds = []
    for seed in range(1, NUM_TRIALS + 1):
        seed_dir = os.path.join(RESULTS_DIR, f"N_{N}", f"seed_{seed}")
        
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
                ckpt = os.path.join(seed_dir, "finetuned", "final_model.pt")
                if os.path.exists(ckpt):
                    model.load_state_dict(torch.load(ckpt, map_location=config.device, weights_only=True)["model_state_dict"])
                    model.eval()
                    with torch.no_grad():
                        pred = model(t_tensor).cpu().numpy().ravel()
                    preds.append(pred)
        except Exception:
            continue
    return np.array(preds)


# ========================================================
# 3. 繪圖函數
# ========================================================
def plot_error_vs_n(df, y_col, title, ylabel, save_name):
    """圖 1 & 圖 3：L2 Error vs Scarce Data level"""
    if df.empty: return
    
    plt.figure(figsize=(8, 6))
    stats = df.groupby("N")[y_col].agg(['mean', 'std']).reset_index()
    N_vals = stats["N"].values
    mean_vals = stats["mean"].values
    std_vals = stats["std"].values
    
    lower_bound = np.clip(mean_vals - 2 * std_vals, 1e-6, None)
    upper_bound = mean_vals + 2 * std_vals
    
    # 畫陰影與線
    plt.fill_between(N_vals, lower_bound, upper_bound, color='blue', alpha=0.2, label=r"$\pm 2$ Std Dev")
    plt.plot(N_vals, mean_vals, color='blue', marker='o', label="$L_2$ error(mean)")
    
    plt.xscale("log"); plt.yscale("log")
    plt.xticks(N_LIST, labels=[str(n) for n in N_LIST])
    plt.gca().xaxis.grid(False) # 關閉垂直網格
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Measurement Points ($N$)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"[繪圖完成] {save_name}")

def plot_prediction_comparison(t, exact, preds_10, preds_30, title, ylabel, save_name):
    """圖 2 & 圖 4：Prediction(mean) 與 +- 2 Std Dev (N=10 vs N=30)"""
    plt.figure(figsize=(8, 6))
    
    # 真實軌跡
    plt.plot(t, exact, 'k-', linewidth=2.5, label='Reference')
    
    # 繪製 N=10 的結果 (紅色系)
    if len(preds_10) > 0:
        mean_10 = np.mean(preds_10, axis=0)
        std_10 = np.std(preds_10, axis=0)
        plt.fill_between(t, mean_10 - 2*std_10, mean_10 + 2*std_10, color='red', alpha=0.2, label=r'$N=10$ ($\pm 2$ Std)')
        plt.plot(t, mean_10, 'r--', linewidth=2, label='$N=10$ (Mean)')

    # 繪製 N=30 的結果 (藍色系)
    if len(preds_30) > 0:
        mean_30 = np.mean(preds_30, axis=0)
        std_30 = np.std(preds_30, axis=0)
        plt.fill_between(t, mean_30 - 2*std_30, mean_30 + 2*std_30, color='blue', alpha=0.3, label=r'$N=30$ ($\pm 2$ Std)')
        plt.plot(t, mean_30, 'b-.', linewidth=2, label='$N=30$ (Mean)')

    plt.title(title)
    plt.xlabel("$t$")
    plt.ylabel(ylabel)
    
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()
    print(f"[繪圖完成] {save_name}")


# ========================================================
# 4. 主流程
# ========================================================
def main():
    os.makedirs("paper_plots", exist_ok=True)
    config = get_config()
    device = config.device
    
    print("========== 準備資料與模型 ==========")
    # 1. 取得真實軌跡 (Ground Truth)
    params = config.system_pedagogical.system_params.to_dict()
    params['noise'] = 0.0
    _, _, _, sol = generate_reaction_ode_dataset(params, T=params['T'], u0=params['u0'], n_t=params['n_t'])
    t_np = sol.t
    t_tensor = torch.tensor(t_np, dtype=torch.float32).reshape(-1, 1).to(device)
    u_true = sol.y[0]
    
    # 物理缺失項為完全省略反應項 (Misspecified as sin(3*pi*t))
    # s_true = 真實動力學 - 錯誤假設 = [sin(3*pi*t) + lambda * u * (1 - u)] - [sin(3*pi*t)]
    # s_true = params['lambda'] * (u_true * (1 - u_true))
    s_true = params['lambda'] * (u_true * (1 - u_true)) - params['lambda'] * np.cos(u_true)

    # ============================================================
    # 【關鍵修改】：不再讀取 JSON！直接呼叫我們新寫的函數重新算誤差
    # ============================================================
    df_metrics = compute_metrics_from_checkpoints(config, t_tensor, u_true, s_true)

    # 3. 印出正確的 Table
    if not df_metrics.empty:
        print_scarcity_table(df_metrics)
    else:
        print("[警告] 未讀取到任何模型權重數據。")

    print("\n========== 開始繪製 4 張圖表 ==========")
    
    # 圖 1: DAPINNs L2 Error vs Scarce Data level
    plot_error_vs_n(df_metrics, "u_err", 
                    "DAPINNs L2 Error vs Scarce Data level", 
                    "Relative $L_2$ Error of $\\tilde{x}(t)$", 
                    "paper_plots/fig1_dapinn_error_vs_n.png")

    # 讀取 N=10 和 N=30 的 DAPINN 軌跡權重
    print("正在提取 DAPINN 軌跡 (N=10 & N=30)...")
    preds_u_10 = get_time_series_predictions(10, "finetune", config, t_tensor)
    preds_u_30 = get_time_series_predictions(30, "finetune", config, t_tensor)
    
    # 圖 2: DAPINNs Prediction(mean) (N=10 vs N=30)
    plot_prediction_comparison(t_np, u_true, preds_u_10, preds_u_30, 
                               "DAPINNs Prediction (mean)", "$u(t)$", 
                               "paper_plots/fig2_dapinn_prediction_comparison.png")

    # 圖 3: Corrector L2 Error vs Scarce Data level
    plot_error_vs_n(df_metrics, "s_err", 
                    "Corrector L2 Error vs Scarce Data level", 
                    "Relative $L_2$ Error of $s_{\\psi}(t)$", 
                    "paper_plots/fig3_corrector_error_vs_n.png")

    # 讀取 N=10 和 N=30 的 Corrector 軌跡權重
    print("正在提取 Corrector 軌跡 (N=10 & N=30)...")
    preds_s_10 = get_time_series_predictions(10, "corrector", config, t_tensor)
    preds_s_30 = get_time_series_predictions(30, "corrector", config, t_tensor)

    # 圖 4: Corrector Prediction(mean) (N=10 vs N=30)
    plot_prediction_comparison(t_np, s_true, preds_s_10, preds_s_30, 
                               "Corrector Prediction (mean)", "$s_{\\psi}(t)$", 
                               "paper_plots/fig4_corrector_prediction_comparison.png")

    print("\n========== 所有圖表生成完畢，請查看 paper_plots 資料夾 ==========")

if __name__ == "__main__":
    main()