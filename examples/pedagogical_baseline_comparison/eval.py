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
from .utils import generate_reaction_ode_dataset, generate_no_reaction_ode_dataset
from dapinns.samplers import RandomSampler

# ============================================================
# 0. Metrics Helper
# ============================================================
def update_metrics(save_root, stage_name, metrics_dict):
    """
    將 metrics 更新至 save_root/evaluation_results.json
    """
    json_path = os.path.join(save_root, "evaluation_results.json")
    
    # 如果檔案已存在，讀取舊數據；否則建立新字典
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    # 更新當前階段的數據
    data[stage_name] = metrics_dict

    # 寫回檔案
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"[INFO] Updated metrics for '{stage_name}' in {json_path}")


# ============================================================
# 1. Evaluate Pretrained PINNs (u only)
# ============================================================
def evaluate_pretrained_pinns(config: ml_collections.ConfigDict, workdir: str):
    sns.set_style("whitegrid")
    device = config.device
    params = config.system_pedagogical.system_params
    T, u0, n_t = params['T'], params['u0'], params['n_t']
    
    _, _, _, sol = generate_no_reaction_ode_dataset(params, T=T, u0=u0, n_t=n_t)
    t_test = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1).to(device)
    u_true = sol.y[0]

    model = PedagogicalBaselineComaprison(config).to(device)
    save_root = os.path.join(workdir, config.saving.save_dir)
    pretrained_dir = os.path.join(save_root, "pretrained")
    best_ckpt = os.path.join(pretrained_dir, "best_model.pt")

    if not os.path.exists(best_ckpt):
        print(f"[SKIP] No pretrained model found at {best_ckpt}")
        return

    checkpoint = torch.load(best_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        best_pred = model(t_test).cpu().numpy().ravel()

    # 計算 Metric
    l2_error = float(np.linalg.norm(best_pred - u_true) / np.linalg.norm(u_true))
    mse = float(np.mean((best_pred - u_true)**2))

    # 更新 JSON
    update_metrics(save_root, "pretrain", {"l2_relative_error": l2_error, "mse": mse})

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, u_true, 'k-', label="Truth")
    plt.plot(sol.t, best_pred, 'r--', label="Pretrained PINN")
    plt.title(f"Pretrained u(t), L2={l2_error:.2e}")
    plt.legend()
    plt.savefig(os.path.join(pretrained_dir, "u_pretrained.png"), dpi=300)
    plt.close()


# ============================================================
# 2. Evaluate Finetuned Model (u + f)
# ============================================================
def evaluate_finetuned_pinns(config: ml_collections.ConfigDict, workdir: str):
    sns.set_style("whitegrid")
    device = config.device
    params = config.system_pedagogical.system_params
    
    _, _, _, sol = generate_reaction_ode_dataset(params, T=params['T'], u0=params['u0'], n_t=params['n_t'])
    t = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1).to(device)
    t.requires_grad = True
    u_true = sol.y[0]

    model = PedagogicalBaselineComaprison(config).to(device)
    save_root = os.path.join(workdir, config.saving.save_dir)
    model_path = os.path.join(save_root, config.saving.finetune_path, "final_model.pt")

    if not os.path.exists(model_path):
        print(f"[SKIP] No finetuned model found at {model_path}")
        return

    model.load_finetuned_model(model_path)
    model.eval()

    u_pred_t = model(t)
    f_pred_t = model.f_function(t, params['lambda'], u_pred_t) - params['lambda'] * torch.cos(u_pred_t)

    u_pred = u_pred_t.detach().cpu().numpy().ravel()
    f_pred = f_pred_t.detach().cpu().numpy().ravel()
    f_true = np.sin(3 * np.pi * sol.t)

    # 計算 Metrics
    u_l2_error = float(np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true))
    f_mse = float(np.mean((f_pred - f_true)**2))

    # 更新 JSON
    update_metrics(save_root, "finetune", {"u_l2_relative_error": u_l2_error, "f_mse": f_mse})

    # Plot u
    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, u_true, 'k-', label="Truth")
    plt.plot(sol.t, u_pred, 'r--', label="Finetuned PINN")
    plt.title(f"Finetuned u(t), L2={u_l2_error:.2e}")
    plt.legend()
    plt.savefig(os.path.join(save_root, "prediction_u.png"), dpi=300)
    plt.close()

    # Plot f
    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, f_true, 'k-', label="True f(t)")
    plt.plot(sol.t, f_pred, 'b--', label="Predicted f(t)")
    plt.title("Prediction of f(t)")
    plt.legend()
    plt.savefig(os.path.join(save_root, "prediction_f.png"), dpi=300)
    plt.close()


# ============================================================
# 3. Evaluate Corrector (sψ)
# ============================================================
def evaluate_corrector(config: ml_collections.ConfigDict, workdir: str):
    device = config.device
    params = config.system_pedagogical.system_params

    model = PedagogicalBaselineComaprison(config).to(device)
    corrector = Corrector(config).to(device)

    save_root = os.path.join(workdir, config.saving.save_dir)
    model_path = os.path.join(save_root, config.saving.finetune_path, "final_model.pt")
    corr_path = os.path.join(save_root, config.saving.corrector_path, "final_corrector.pt")

    if not os.path.exists(corr_path):
        print(f"[SKIP] No corrector model found at {corr_path}")
        return

    model.load_finetuned_model(model_path)
    corrector.load_corrector_model(corr_path)
    model.eval()
    corrector.eval()

    _, _, _, sol = generate_reaction_ode_dataset(params, T=params['T'], u0=params['u0'], n_t=params['n_t'])
    t = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1).to(device)
    t.requires_grad = True

    u = model(t)
    du = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=False)[0]
    s_pred = corrector(torch.cat([u, du], dim=1)).detach().cpu().numpy().ravel()
    
    u_true = sol.y[0]
    s_true = params['lambda'] * (u_true * (1 - u_true) - np.cos(u_true))

    # 計算 Metrics
    s_mse = float(np.mean((s_pred - s_true)**2))
    s_l2_relative = float(np.linalg.norm(s_pred - s_true) / np.linalg.norm(s_true))

    # 更新 JSON
    update_metrics(save_root, "corrector", {"s_mse": s_mse, "s_l2_relative_error": s_l2_relative})

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, s_true, 'k-', label=r"True $\lambda u(1-u) - \lambda \cos(u)$")
    plt.plot(sol.t, s_pred, 'g--', label=r"Corrector $s_\psi$")
    plt.title(f"Corrector Prediction, MSE={s_mse:.2e}")
    plt.legend()
    plt.savefig(os.path.join(save_root, "prediction_corrector.png"), dpi=300)
    plt.close()


# ============================================================
# 4. Entry Point
# ============================================================
def evaluate(config: ml_collections.ConfigDict, workdir: str):
    print("\n========== Evaluation ==========")
    # 注意：這裡會依序嘗試執行，如果檔案不存在會自動 Skip 並保留 JSON 原有數據
    evaluate_pretrained_pinns(config, workdir)
    evaluate_finetuned_pinns(config, workdir)
    if config.use_corrector:
        evaluate_corrector(config, workdir)
    print("========== Done ==========\n")