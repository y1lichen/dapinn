import os
import torch
import ml_collections
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
import shutil

from .models import PedagogicalBaselineComaprison, Corrector
from .utils import generate_reaction_ode_dataset
from dapinns.samplers import RandomSampler


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
    pretrained_dir = os.path.join(workdir, config.saving.save_dir, "pretrained")

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
    t = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1).to(device)
    t.requires_grad = True

    u_true = sol.y[0]

    model = PedagogicalBaselineComaprison(config).to(device)
    save_root = os.path.join(workdir, config.saving.save_dir)
    model_path = os.path.join(save_root, config.saving.finetune_path, "final_model.pt")
    model.load_finetuned_model(model_path)
    model.eval()

    # ----- u prediction -----
    u_pred = model(t)
    du_pred = torch.autograd.grad(u_pred, t, torch.ones_like(u_pred), create_graph=False)[0]
    f_pred = model.f_function(t, params['lambda'], u_pred) - params['lambda']*torch.cos(u_pred)

    u_pred = u_pred.detach().cpu().numpy().ravel()
    du_pred = du_pred.detach().cpu().numpy().ravel()
    f_pred = f_pred.detach().cpu().numpy().ravel()

    # ----- plot u -----
    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, u_true, 'k-', label="Truth")
    plt.plot(sol.t, u_pred, 'r--', label="Finetuned PINN")
    plt.title("Prediction of u(t)")
    plt.legend()
    plt.savefig(os.path.join(save_root, "prediction_u.png"), dpi=300)
    plt.close()

    # ----- plot f -----
    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, np.sin(3 * np.pi * sol.t), 'k-', label="True f(t)")
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
    model.load_finetuned_model(os.path.join(save_root, config.saving.finetune_path, "final_model.pt"))
    corrector.load_corrector_model(os.path.join(save_root, config.saving.corrector_path, "final_corrector.pt"))

    model.eval()
    corrector.eval()

    _, _, _, sol = generate_reaction_ode_dataset(params, T=params['T'], u0=params['u0'], n_t=params['n_t'])
    t = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1).to(device)
    
    # 預測解 u_theta
    u_pred_torch = model(t)
    u_pred = u_pred_torch.detach().cpu().numpy().ravel()
    
    # 預測校正項 s_psi
    s_pred = corrector(t).detach().cpu().numpy().ravel()
    
    # 真實解 u_true
    u_true = sol.y[0]
    
    # --- 計算反應項 phi ---
    # 真實反應項: lambda * u * (1 - u)
    phi_true = params['lambda'] * (u_true * (1 - u_true))
    
    # 預測(修正後)反應項: lambda_tilde * cos(u_theta) + s_psi
    # 注意：這裡的 params['lambda'] 若為 0.2 則對應論文中的固定 lambda_tilde
    phi_corrected = params['lambda'] * np.cos(u_pred) + s_pred

    # --- 第一張圖：原本的 Corrector 差異對比 ---
    s_true = params['lambda'] * (u_true * (1 - u_true) - np.cos(u_true))
    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, s_true, 'k-', label=r"True $\lambda u(1-u) - \lambda \cos(u)$")
    plt.plot(sol.t, s_pred, 'g--', label=r"Corrector $s_\psi$")
    plt.title("Discrepancy Correction ($s_\psi$)")
    plt.legend()
    plt.savefig(os.path.join(save_root, "prediction_corrector.png"), dpi=300)
    plt.close()

    # --- 第二張圖：完整物理反應項 phi 的重建對比 ---
    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, phi_true, 'k-', label=r"True $\phi$: $\lambda u(1-u)$")
    plt.plot(sol.t, phi_corrected, 'r--', label=r"Identified $\phi$: $\tilde{\lambda} \cos(u_\theta) + s_\psi$")
    plt.title("Reconstruction of Governing Reaction Term ($\phi$)")
    plt.xlabel("t")
    plt.ylabel(r"$\phi$")
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
