# pedagogical_baseline_comparison/eval.py

import os
import torch
import ml_collections
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
import shutil
import json

from .models import PedagogicalBaselineComaprison, Corrector
from .utils import generate_reaction_ode_dataset

def update_metrics(save_root, stage_name, metrics_dict):
    json_path = os.path.join(save_root, "evaluation_results.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f: data = json.load(f)
        except json.JSONDecodeError: data = {}
    else: data = {}
    data[stage_name] = metrics_dict
    with open(json_path, 'w') as f: json.dump(data, f, indent=4)
    print(f"[INFO] Updated metrics for '{stage_name}' in {json_path}")

def evaluate_pretrained_pinns(config, workdir):
    sns.set_style("whitegrid")
    device = config.device
    
    # 修正：評估用的真值參數必須清除雜訊
    params = config.system_pedagogical.system_params.to_dict()
    params['noise'] = 0.0 
    
    T, u0, n_t = params['T'], params['u0'], params['n_t']
    _, _, _, sol = generate_reaction_ode_dataset(params, T=T, u0=u0, n_t=n_t)
    t_test = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1).to(device)
    u_true = sol.y[0]

    model = PedagogicalBaselineComaprison(config).to(device)
    save_root = os.path.join(workdir, config.saving.save_dir)
    pretrained_dir = os.path.join(save_root, "pretrained")
    best_ckpt = os.path.join(pretrained_dir, "best_model.pt")

    if not os.path.exists(best_ckpt): return

    checkpoint = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        best_pred = model(t_test).cpu().numpy().ravel()

    l2_error = float(np.linalg.norm(best_pred - u_true) / np.linalg.norm(u_true))
    mse = float(np.mean((best_pred - u_true)**2))
    update_metrics(save_root, "pretrain", {"u_l2_relative_error": l2_error, "u_mse": mse})

def evaluate_finetuned_pinns(config, workdir):
    sns.set_style("whitegrid")
    device = config.device
    
    # 關鍵修正：強制清除雜訊參數，獲得「純淨物理軌跡」
    params = config.system_pedagogical.system_params.to_dict()
    params['noise'] = 0.0 
    
    _, _, _, sol = generate_reaction_ode_dataset(params, T=params['T'], u0=params['u0'], n_t=params['n_t'])
    t_torch = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1).to(device)
    t_torch.requires_grad = True
    u_true = sol.y[0]
    f_true = np.sin(3 * np.pi * sol.t)

    model = PedagogicalBaselineComaprison(config).to(device)
    save_root = os.path.join(workdir, config.saving.save_dir)
    model_path = os.path.join(save_root, config.saving.finetune_path, "final_model.pt")
    
    if not os.path.exists(model_path): return

    model.load_finetuned_model(model_path)
    model.eval()

    u_pred_torch = model(t_torch)
    f_pred_torch = model.f_function(t_torch, params['lambda'], u_pred_torch) - params['lambda']*torch.cos(u_pred_torch)

    u_pred = u_pred_torch.detach().cpu().numpy().ravel()
    f_pred = f_pred_torch.detach().cpu().numpy().ravel()

    # 計算誤差：因為 u_true 不含雜訊，隨雜訊增加預測偏離真值越多，誤差才會上揚
    u_l2 = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
    f_mse = np.mean((f_pred - f_true)**2)
    update_metrics(save_root, "finetune", {"u_l2_relative_error": float(u_l2), "f_mse": float(f_mse)})

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, u_true, 'k-', label="Clean Truth")
    plt.plot(sol.t, u_pred, 'r--', label="CMPINN Prediction")
    plt.title(f"Finetuned u(t) Robustness, L2={u_l2:.2e}")
    plt.savefig(os.path.join(save_root, "prediction_u.png"), dpi=300)
    plt.close()

def evaluate_corrector(config, workdir):
    device = config.device
    params = config.system_pedagogical.system_params.to_dict()
    params['noise'] = 0.0 # 物理項真值評估同樣不含雜訊

    model = PedagogicalBaselineComaprison(config).to(device)
    corrector = Corrector(config).to(device)
    save_root = os.path.join(workdir, config.saving.save_dir)
    model_path = os.path.join(save_root, config.saving.finetune_path, "final_model.pt")
    corr_path = os.path.join(save_root, config.saving.corrector_path, "best_corrector.pt")

    if not os.path.exists(corr_path) or not os.path.exists(model_path): return

    model.load_finetuned_model(model_path)
    corrector.load_corrector_model(corr_path)
    model.eval()
    corrector.eval()

    _, _, _, sol = generate_reaction_ode_dataset(params, T=params['T'], u0=params['u0'], n_t=params['n_t'])
    t_torch = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1).to(device)
    
    u_pred_torch = model(t_torch)
    s_pred = corrector(t_torch).detach().cpu().numpy().ravel()
    u_true = sol.y[0]
    
    # 真實反應項 phi 與真值 s
    phi_true = params['lambda'] * (u_true * (1 - u_true))
    s_true = params['lambda'] * (u_true * (1 - u_true) - np.cos(u_true))
    
    # 預測重建
    u_pred_np = u_pred_torch.detach().cpu().numpy().ravel()
    phi_corrected = params['lambda'] * np.cos(u_pred_np) + s_pred

    s_l2 = np.linalg.norm(s_pred - s_true) / np.linalg.norm(s_true)
    phi_l2 = np.linalg.norm(phi_corrected - phi_true) / np.linalg.norm(phi_true)
    
    update_metrics(save_root, "corrector", {"s_l2_relative_error": float(s_l2), "phi_l2_relative_error": float(phi_l2)})

def evaluate(config, workdir):
    print("\n========== Evaluation (CMPINN Clean GT Mode) ==========")
    evaluate_pretrained_pinns(config, workdir)
    evaluate_finetuned_pinns(config, workdir)
    if config.use_corrector:
        evaluate_corrector(config, workdir)