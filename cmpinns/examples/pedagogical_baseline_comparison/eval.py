import os
import torch
import ml_collections
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
import shutil
import json  # 新增用於儲存結果

from .models import PedagogicalBaselineComaprison, Corrector
from .utils import generate_reaction_ode_dataset
from dapinns.samplers import RandomSampler


# ============================================================
# 0. Metrics Helper (新增)
# ============================================================
def update_metrics(save_root, stage_name, metrics_dict):
    """
    將計量指標更新至 save_root/evaluation_results.json
    """
    json_path = os.path.join(save_root, "evaluation_results.json")
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    data[stage_name] = metrics_dict

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
    
    _, _, _, sol = generate_reaction_ode_dataset(params, T=T, u0=u0, n_t=n_t)
    t_test = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1).to(device)
    u_true = sol.y[0]

    model = PedagogicalBaselineComaprison(config).to(device)
    save_root = os.path.join(workdir, config.saving.save_dir)
    pretrained_dir = os.path.join(save_root, "pretrained")

    checkpoint_files = glob.glob(os.path.join(pretrained_dir, "checkpoint_*.pt"))
    if not checkpoint_files:
        print("[WARN] No pretrained checkpoints found.")
        return

    best_l2 = float("inf")
    best_ckpt, best_pred = None, None

    for ckpt in checkpoint_files:
        try:
            checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            with torch.no_grad():
                pred = model(t_test).cpu().numpy().ravel()

            l2 = np.linalg.norm(pred - u_true) / np.linalg.norm(u_true)
            if l2 < best_l2:
                best_l2 = l2
                best_ckpt = ckpt
                best_pred = pred
        except:
            continue

    shutil.copy(best_ckpt, os.path.join(pretrained_dir, "best_model.pt"))
    
    # --- 計算並儲存 Metrics ---
    mse = np.mean((best_pred - u_true)**2)
    update_metrics(save_root, "pretrain", {"u_l2_relative_error": float(best_l2), "u_mse": float(mse)})

    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, u_true, 'k-', label="Truth")
    plt.plot(sol.t, best_pred, 'r--', label="Pretrained PINN")
    plt.title(f"Pretrained u(t), L2={best_l2:.2e}")
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
    t_torch = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1).to(device)
    t_torch.requires_grad = True

    u_true = sol.y[0]
    f_true = np.sin(3 * np.pi * sol.t)

    model = PedagogicalBaselineComaprison(config).to(device)
    save_root = os.path.join(workdir, config.saving.save_dir)
    model_path = os.path.join(save_root, config.saving.finetune_path, "final_model.pt")
    
    if not os.path.exists(model_path):
        print(f"[SKIP] No finetuned model found at {model_path}")
        return

    model.load_finetuned_model(model_path)
    model.eval()

    u_pred_torch = model(t_torch)
    # 計算外部驅動力 f
    f_pred_torch = model.f_function(t_torch, params['lambda'], u_pred_torch) - params['lambda']*torch.cos(u_pred_torch)

    u_pred = u_pred_torch.detach().cpu().numpy().ravel()
    f_pred = f_pred_torch.detach().cpu().numpy().ravel()

    # --- 計算並儲存 Metrics ---
    u_l2 = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
    f_mse = np.mean((f_pred - f_true)**2)
    update_metrics(save_root, "finetune", {"u_l2_relative_error": float(u_l2), "f_mse": float(f_mse)})

    # Plot u
    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, u_true, 'k-', label="Truth")
    plt.plot(sol.t, u_pred, 'r--', label="Finetuned PINN")
    plt.title(f"Finetuned u(t), L2={u_l2:.2e}")
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

    if not os.path.exists(corr_path) or not os.path.exists(model_path):
        print("[SKIP] Finetuned model or corrector not found.")
        return

    model.load_finetuned_model(model_path)
    corrector.load_corrector_model(corr_path)

    model.eval()
    corrector.eval()

    _, _, _, sol = generate_reaction_ode_dataset(params, T=params['T'], u0=params['u0'], n_t=params['n_t'])
    t_torch = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1).to(device)
    
    # 預測
    u_pred_torch = model(t_torch)
    s_pred = corrector(t_torch).detach().cpu().numpy().ravel()
    
    u_pred = u_pred_torch.detach().cpu().numpy().ravel()
    u_true = sol.y[0]
    
    # 真實反應項 phi 與真值 s
    phi_true = params['lambda'] * (u_true * (1 - u_true))
    s_true = params['lambda'] * (u_true * (1 - u_true) - np.cos(u_true))
    
    # 預測重建反應項
    phi_corrected = params['lambda'] * np.cos(u_pred) + s_pred

    # --- 計算並儲存 Metrics ---
    s_mse = np.mean((s_pred - s_true)**2)
    s_l2 = np.linalg.norm(s_pred - s_true) / np.linalg.norm(s_true)
    phi_l2 = np.linalg.norm(phi_corrected - phi_true) / np.linalg.norm(phi_true)
    
    update_metrics(save_root, "corrector", {
        "s_l2_relative_error": float(s_l2), 
        "s_mse": float(s_mse),
        "phi_l2_relative_error": float(phi_l2)
    })

    # Plot Corrector
    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, s_true, 'k-', label=r"True $\lambda u(1-u) - \lambda \cos(u)$")
    plt.plot(sol.t, s_pred, 'g--', label=r"Corrector $s_\psi$")
    plt.title(f"Discrepancy Correction, L2={s_l2:.2e}")
    plt.legend()
    plt.savefig(os.path.join(save_root, "prediction_corrector.png"), dpi=300)
    plt.close()

    # Plot phi reconstruction
    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, phi_true, 'k-', label=r"True $\phi$: $\lambda u(1-u)$")
    plt.plot(sol.t, phi_corrected, 'r--', label=r"Identified $\phi$")
    plt.title(f"Phi Reconstruction, L2={phi_l2:.2e}")
    plt.legend()
    plt.savefig(os.path.join(save_root, "phi_reconstruction.png"), dpi=300)
    plt.close()

# ============================================================
# 4. Entry Point
# ============================================================
def evaluate(config: ml_collections.ConfigDict, workdir: str):
    print("\n========== Evaluation ==========")
    evaluate_pretrained_pinns(config, workdir)
    evaluate_finetuned_pinns(config, workdir)
    if config.use_corrector:
        evaluate_corrector(config, workdir)
    print("========== Done ==========\n")