import os
import subprocess
import json
import numpy as np

NUM_TRIALS = 30 # 畫 std 陰影需要足夠的樣本數，建議跑完 30 次

results = {"A": [], "B": [], "C": []}

def run_experiment(scenario, seed, env, flags, save_subdir):
    print(f"\n[{scenario}] Running Seed {seed}...")
    
    base_cmd = ["python", "-m", "examples.pedagogical_baseline_comparison.main"]
    base_flags = [f"--seed={seed}", f"--save_subdir={save_subdir}"] + flags

    # 1. Train & 2. Eval
    subprocess.run(base_cmd + ["--mode=train"] + base_flags, env=env, check=True)
    subprocess.run(base_cmd + ["--mode=eval"] + base_flags, env=env, check=True)

    # 3. Read Results
    json_path = os.path.join(".", save_subdir, "evaluation_results.json")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            error = data["finetune"]["u_l2_relative_error"]
            results[scenario].append(error)
            print(f"[{scenario}] Seed {seed} Error: {error:.4e}")
    except Exception as e:
        print(f"[ERROR] Could not read metrics for {scenario} seed {seed}: {e}")

if __name__ == "__main__":
    for seed in range(1, NUM_TRIALS + 1):
        print("\n" + "="*40 + f"\n TRIAL {seed}/{NUM_TRIALS} \n" + "="*40)

        # 這裡的 save_subdir 加入了 seed，確保不會互相覆寫
        env_A = os.environ.copy(); env_A["USE_CORRECT_PHYSICS"] = "1"
        run_experiment("A", seed, env_A, ["--use_corrector=False", "--run_pretrain=False", "--run_finetune=True"], f"results_scenario_A/seed_{seed}")

        env_B = os.environ.copy(); env_B["USE_CORRECT_PHYSICS"] = "0"
        run_experiment("B", seed, env_B, ["--use_corrector=False", "--run_pretrain=False", "--run_finetune=True"], f"results_scenario_B/seed_{seed}")

        env_C = os.environ.copy(); env_C["USE_CORRECT_PHYSICS"] = "0"
        run_experiment("C", seed, env_C, ["--use_corrector=True", "--run_pretrain=True", "--load_pretrained=True", "--run_finetune=True"], f"results_scenario_C/seed_{seed}")



import os
import json
import numpy as np

NUM_TRIALS = 30

# 設定三個情境對應的資料夾名稱
scenarios = {
    "A": "results_scenario_A",
    "B": "results_scenario_B",
    "C": "results_scenario_C"
}

results = {"A": [], "B": [], "C": []}

# 1. 讀取所有 Seed 的評估結果
for scenario_key, folder_name in scenarios.items():
    for seed in range(1, NUM_TRIALS + 1):
        json_path = os.path.join(folder_name, f"seed_{seed}", "evaluation_results.json")
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # 抓取 finetune 階段的 u 的相對誤差
                    if "finetune" in data and "u_l2_relative_error" in data["finetune"]:
                        error = data["finetune"]["u_l2_relative_error"]
                        results[scenario_key].append(error)
            except Exception as e:
                print(f"[Warning] 讀取 {json_path} 失敗: {e}")
        else:
            print(f"[Warning] 找不到檔案: {json_path}")

# 2. 格式化輸出 Table
def format_result(errors):
    if not errors:
        return "N/A"
    mean_val = np.mean(errors)
    std_val = np.std(errors)
    # 格式化成科學記號，例如 4.0e-3 (1.0e-3)
    return f"{mean_val:.1e} ({std_val:.1e})"

print("\n" + "="*50)
print(" physical error")
print("="*50)
print(f"(A) PINNs with correct physics        {format_result(results['A'])}")
print(f"(B) PINNs with misspecified physics   {format_result(results['B'])}")
print(f"(C) DAPINNs with ADPC                 {format_result(results['C'])}")
print("="*50 + "\n")