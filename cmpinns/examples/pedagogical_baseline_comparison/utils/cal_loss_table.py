import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

def aggregate_dapinn_results(results_root):
    """
    遍歷目錄並收集所有 trial 的 u_l2_relative_error (預測) 與 s_l2_relative_error (修正項)
    """
    # 儲存結構: { 'u_l2': {size: [errors]}, 's_l2': {size: [errors]} }
    raw_data = {
        "u_l2": defaultdict(list),
        "s_l2": defaultdict(list)
    }
    
    for size_dir in os.listdir(results_root):
        if not size_dir.startswith("size_"):
            continue
            
        try:
            size_val = int(size_dir.split("_")[1])
        except ValueError:
            continue
            
        size_path = os.path.join(results_root, size_dir)
        
        for trial_dir in os.listdir(size_path):
            json_path = os.path.join(size_path, trial_dir, "evaluation_results.json")
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    try:
                        res = json.load(f)
                        # 1. 提取 PINN 模型預測 u 的誤差 (來自 finetune 區塊)
                        u_err = res.get("finetune", {}).get("u_l2_relative_error")
                        # 2. 提取 ADPC 修正項 s 的誤差 (來自 corrector 區塊)
                        s_err = res.get("corrector", {}).get("s_l2_relative_error")
                        
                        if u_err is not None:
                            raw_data["u_l2"][size_val].append(u_err)
                        if s_err is not None:
                            raw_data["s_l2"][size_val].append(s_err)
                    except json.JSONDecodeError:
                        continue

    return raw_data

def format_table(raw_data):
    """
    計算統計值並格式化為包含預測與修正項的表格
    """
    # 取得所有樣本大小並排序
    all_sizes = set(raw_data["u_l2"].keys()) | set(raw_data["s_l2"].keys())
    sorted_sizes = sorted(list(all_sizes))
    
    # 建立空的 DataFrame
    df = pd.DataFrame(index=["Prediction (u)", "Corrector (s)"], columns=sorted_sizes)
    
    for metric_key, row_name in [("u_l2", "Prediction (u)"), ("s_l2", "Corrector (s)")]:
        formatted_row = []
        for size in sorted_sizes:
            errors = raw_data[metric_key].get(size, [])
            if errors:
                m = np.mean(errors)
                s = np.std(errors)
                # 格式化為科學記號，標準差放在括號中
                formatted_row.append(f"{m:.1e}\n({s:.1e})")
            else:
                formatted_row.append("N/A")
        df.loc[row_name] = formatted_row
    
    return df

if __name__ == "__main__":
    # 請確保此路徑指向您的結果根目錄
    RESULTS_DIR = "cmpinns/results/pedagogical_baseline_comparison"
    
    print(f"Aggregating 30 trials per size from {RESULTS_DIR}...")
    data = aggregate_dapinn_results(RESULTS_DIR)
    
    if not data["u_l2"] and not data["s_l2"]:
        print("No data found. Please check your directories and JSON files.")
    else:
        result_table = format_table(data)
        
        print("\nTable 2: Relative L2 error (mean and std) under varying data scarcity")
        print("=" * 100)
        # 使用 to_string() 確保換行格式正確顯示
        print(result_table.to_string())
        print("=" * 100)
        
        # 存成 CSV 供論文使用
        # result_table.to_csv("dapinn_statistical_results.csv")