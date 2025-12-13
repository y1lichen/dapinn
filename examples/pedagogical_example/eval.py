import os
import torch
import ml_collections
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from examples.pedagogical_example.models import Pedagogical, Corrector
from examples.pedagogical_example.utils import generate_reaction_ode_dataset

def evaluate(config: ml_collections.ConfigDict, workdir: str):
    sns.set_style("whitegrid")
    device = config.device

    # 1. Load Ground Truth Data
    params = config.system_pedagogical.system_params
    T, u0, n_t = params['T'], params['u0'], params['n_t']
    _, _, _, sol = generate_reaction_ode_dataset(params, T=T, u0=u0, n_t=n_t)
    
    t_test = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1).to(device)
    u_true = sol.y[0]

    # 2. Load Finetuned Model
    model = Pedagogical(config).to(device)
    finetune_path = os.path.join(workdir, config.saving.save_dir, config.saving.finetune_path, "final_model.pt")
    
    if not os.path.exists(finetune_path):
        # Try best model if final not found
        finetune_path = os.path.join(workdir, config.saving.save_dir, config.saving.finetune_path, "best_finetuned_model.pt")

    try:
        model.load_finetuned_model(finetune_path)
        model.eval()
    except Exception as e:
        print(f"[Error] Cannot load model for evaluation: {e}")
        return

    # 3. Prediction Plot
    with torch.no_grad():
        u_pred = model(t_test).cpu().numpy().ravel()

    plt.figure(figsize=(10, 6))
    plt.plot(sol.t, u_true, 'k-', linewidth=2, label="Ground Truth")
    plt.plot(sol.t, u_pred, 'r--', linewidth=2, label="Prediction")
    
    title_str = "DAPINN" if config.use_corrector else "Standard PINN (Wrong Physics)"
    plt.title(f"{title_str}: Prediction vs Truth")
    plt.legend()
    plt.savefig(os.path.join(workdir, config.saving.save_dir, "prediction_comparison.png"), dpi=300)
    plt.close()

    # 4. Corrector Analysis (Only if DAPINN)
    if config.use_corrector:
        corrector = Corrector(config).to(device)
        corr_path = os.path.join(workdir, config.saving.save_dir, config.saving.corrector_path, "final_corrector.pt")
        
        try:
            corrector.load_corrector_model(corr_path)
            corrector.eval()
            
            # Prepare inputs: u and du/dt from the MODEL (not ground truth)
            t_test.requires_grad = True
            u_model = model(t_test)
            du_model = torch.autograd.grad(u_model, t_test, torch.ones_like(u_model), create_graph=False)[0]
            
            # Input to corrector
            inputs = torch.cat([u_model, du_model], dim=1)
            s_pred = corrector(inputs).cpu().detach().numpy().ravel()
            
            # Ground Truth Missing Term: lambda * u * (1-u)
            # NOTE: Calculated using the TRUE u, or Model u? 
            # Ideally, corrector should match the term based on the state it sees.
            # Let's plot based on u_true for reference
            lam = params['lambda']
            s_true = lam * u_true * (1 - u_true)
            
            plt.figure(figsize=(10, 6))
            plt.plot(sol.t, s_true, 'k-', alpha=0.6, label=r"True Missing Term $\lambda u(1-u)$")
            plt.plot(sol.t, s_pred, 'b--', linewidth=2, label=r"Corrector Output $s_\psi$")
            plt.title("Corrector Discovery Performance")
            plt.legend()
            plt.savefig(os.path.join(workdir, config.saving.save_dir, "corrector_analysis.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"[Warn] Corrector evaluation skipped: {e}")