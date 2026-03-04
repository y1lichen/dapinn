import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def collect_noise_data(results_root):
    # 結構: { size: { noise: [errors] } }
    data = defaultdict(lambda: defaultdict(list))
    
    if not os.path.exists(results_root):
        print(f"[ERROR] 找不到路徑: {os.path.abspath(results_root)}")
        return None

    # 遍歷 noise 資料夾 (例如 noise_0.01)
    for noise_dir in os.listdir(results_root):
        if not noise_dir.startswith("noise_"): continue
        try:
            noise_val = float(noise_dir.split("_")[1])
        except: continue
        
        noise_path = os.path.join(results_root, noise_dir)
        # 遍歷 size 資料夾 (例如 size_30)
        for size_dir in os.listdir(noise_path):
            if not size_dir.startswith("size_"): continue
            try:
                size_val = int(size_dir.split("_")[1])
            except: continue
            
            size_path = os.path.join(noise_path, size_dir)
            # 遍歷 trials
            for trial_dir in os.listdir(size_path):
                json_path = os.path.join(size_path, trial_dir, "evaluation_results.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        try:
                            res = json.load(f)
                            # 提取場量 u 的相對 L2 誤差
                            err = res.get("finetune", {}).get("u_l2_relative_error")
                            if err is not None:
                                data[size_val][noise_val].append(err)
                        except: continue
    return data

def plot_robustness(data):
    if not data:
        print("沒有收集到任何有效數據。")
        return

    plt.figure(figsize=(10, 6))
    # 這裡對齊您提供的圖表顏色
    colors = {30: 'blue', 100: 'red'}
    
    sorted_sizes = sorted(data.keys())
    for size in sorted_sizes:
        noises = sorted(data[size].keys())
        means = [np.mean(data[size][n]) for n in noises]
        stds = [np.std(data[size][n]) for n in noises]
        
        means = np.array(means)
        stds = np.array(stds)
        
        # 繪製主線 (Mean)
        plt.plot(noises, means, 'o-', color=colors.get(size, 'black'), label=f'{size} pts L2 Error (mean)')
        
        # 繪製陰影區域 (±2 Std Dev 信心區間)
        plt.fill_between(noises, means - 2*stds, means + 2*stds, 
                         color=colors.get(size, 'black'), alpha=0.15, label=f'{size} pts ±2 Std Dev')

    plt.xlabel("Noise Level")
    plt.ylabel("Relative L2 Error")
    plt.title("CMPINNs L2 Error across Noise Levels under Limited Data Regimes (30 vs. 100 Pts)")
    
    # 將 X 軸座標轉為百分比顯示
    all_noises = sorted(list(set([n for s in data for n in data[s]])))
    plt.xticks(all_noises, [f"{int(n*100)}%" for n in all_noises])
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_name = "noise_robustness_analysis.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"圖表已儲存至: {os.path.abspath(output_name)}")

if __name__ == "__main__":
    # --- 修正後的路徑 ---
    # 這裡必須對應您 Bash 腳本中的 RESULT_BASE
    # RESULTS_DIR = "results/pedagogical_baseline_comparison/noise_experiment"
    RESULTS_DIR = "cmpinns/results/pedagogical_baseline_comparison/noise_experiment"
    print(f"正在從 {RESULTS_DIR} 彙整數據...")
    raw_data = collect_noise_data(RESULTS_DIR)
    plot_robustness(raw_data)