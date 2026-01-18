import os
import time
import wandb
import torch
import numpy as np

from .models import Pedagogical, Corrector
from .utils import generate_reaction_ode_dataset
from dapinns.samplers import RandomSampler
from dapinns.utils import save_checkpoint

# ==========================================
# Stage 1: Pre-training (Eq 3.1)
# ==========================================
def pretrain(config, workdir):
    wandb.init(
        config=dict(config.wandb),
        project=config.wandb.project,
        name=config.wandb.name,
        tags=["pretrain"]
    )
    """
    Pre-trains the model on the INCOMPLETE physics only.
    Target: du/dt = f(t), with NO reaction term.
    NO observational data is used here.
    """
    print("\n=== Start Pre-training (Physics Only) ===")
    
    # Setup Model
    model = Pedagogical(config).to(config.device)
    model.train()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.pretraining.lr)
    
    # Save paths
    save_root = os.path.join(workdir, config.saving.save_dir, "pretrained")
    os.makedirs(save_root, exist_ok=True)

    max_epochs = config.pretraining.max_epochs
    best_loss = float("inf")

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        
        # Only Physics Loss (Eq 3.1: L_f = L_ic + L_res)
        # No corrector is used in pretraining
        loss, _, _, _ = model.f_loss(corrector=None)
        
        loss.backward()
        optimizer.step()

        # Logging
        if epoch % 100 == 0:
            print(f"[Pretrain] Epoch {epoch:04d} | Loss: {loss.item():.3e}")
        
        wandb.log({"pretrain_loss": loss.item()})

        # Save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(), # deep copy
                "loss": loss.item()
            }, save_root, epoch, keep=1, name="best_model.pt")

    print(f"=== Pre-training Done. Best Loss: {best_loss:.3e} ===\n")


# ==========================================
# Stage 2: Fine-tuning (Eq 3.2 - 3.4)
# ==========================================
def finetune(config, workdir):

    wandb.init(
        config=dict(config.wandb),
        project=config.wandb.project,
        name=config.wandb.name,
        tags=["finetune"]
    )
    # — 1. Data Generation (Ground Truth) —
    p = config.system_pedagogical.system_params
    T, u0, n_t = p["T"], p["u0"], p["n_t"]
    
    # 生成 Ground Truth: du/dt = f(t) + lambda*u(1-u)
    t, u, f, sol = generate_reaction_ode_dataset(params=p, T=T, u0=u0, n_t=n_t)
    
    # Sampling measurements
    
    sampler = RandomSampler(config, sample_size=config.sample_size)
    t_train, u_train = sampler.generate_data(t, u)
    t_train = t_train.to(config.device)
    u_train = u_train.to(config.device)

    # — 2. Load Models —
    model = Pedagogical(config).to(config.device)
    
    # 如果是 DAPINN 模式，先加載 Pretrained weights (Learned on incomplete physics)
    # 如果是 Standard PINN (baseline)，通常從隨機初始化開始，或者也可加載
    if config.use_corrector and config.load_pretrained:
        pretrained_path = os.path.join(workdir, config.saving.save_dir, "pretrained", "best_model.pt")
        model.load_pretrained_model(pretrained_path)

    model.train()
    # If I to freeze the model......
    # for p in model.parameters():
    #     p.requires_grad = False
    # Setup Corrector
    use_corrector = config.use_corrector
    if use_corrector:
        corrector = Corrector(config).to(config.device)
        corrector.train()
        print("[INFO] Running DAPINN (Model + Corrector)")
    else:
        corrector = None
        print("[INFO] Running Standard PINN (No Corrector)")

    # — 3. Optimizers —
    # Separate optimizers for alternating updates
    model_optimizer = torch.optim.Adam(model.parameters(), lr=config.finetune_pinns_optim.lr)
    model.optimizer = model_optimizer # attach for save_checkpoint convenience

    if use_corrector:
        corrector_optimizer = torch.optim.Adam(corrector.parameters(), lr=config.finetune_correction_optim.lr)
        corrector.optimizer = corrector_optimizer

    # — 4. Training Hyperparams —
    max_epochs = config.finetuning.max_epochs
    alt_steps = config.finetuning.alt_steps  # m epochs
    u_w, f_w, ic_w = config.finetuning.u_w, config.finetuning.f_w, config.finetuning.ic_w
    
    save_root = os.path.join(workdir, config.saving.save_dir)
    finetune_dir = os.path.join(save_root, config.saving.finetune_path)
    corrector_dir = os.path.join(save_root, config.saving.corrector_path)
    os.makedirs(finetune_dir, exist_ok=True)
    if use_corrector: os.makedirs(corrector_dir, exist_ok=True)

    best_total = float("inf")
    
    # — 5. Main Loop (Alternating) —
    print("=== Start Fine-tuning ===")
    
    for epoch in range(1, max_epochs + 1):
        
        # logic for alternating: 
        # If no corrector: always update model
        # If corrector: switch every 'alt_steps' epochs
        
        update_model = True
        if use_corrector:
            # Cycle: [0, m-1] -> Model, [m, 2m-1] -> Corrector
            cycle_idx = (epoch - 1) // alt_steps
            if cycle_idx % 2 == 1:
                update_model = False # Update Corrector
        
        # --- Step A: Update Model (Theta) ---
        if update_model:
            model_optimizer.zero_grad()
            
            # Data Loss (L_u)
            u_loss = model.u_loss(t_train, u_train)
            
            # Physics Loss (L_res)
            # If corrector exists, it is used in calculation but NOT updated here
            f_loss, ode_loss, ic_loss, corr_in = model.f_loss(corrector)
            
            # Eq 3.2: L_total = w_u * L_u + w_f * L_res
            total_loss = u_w * u_loss + (f_w * ode_loss) + (ic_w * ic_loss)
            
            total_loss.backward()
            model_optimizer.step()
            
        # --- Step B: Update Corrector (Psi) ---
        else:
            corrector_optimizer.zero_grad()
            
            # Physics Loss ONLY (Eq 3.3)
            # We want Corrector to minimize the PDE residual given fixed u_theta
            f_loss, ode_loss, ic_loss, corr_in = model.f_loss(corrector)
            
            # Note: Data loss does not depend on corrector directly in this formulation,
            # so we usually only backprop f_loss or total_loss (u_loss gradient will be 0 for corrector)
            total_loss = (f_w * ode_loss) + (ic_w * ic_loss)
            
            total_loss.backward()
            corrector_optimizer.step()
            
            # Recalculate u_loss just for logging
            with torch.no_grad():
                u_loss = model.u_loss(t_train, u_train)
                total_loss = u_w * u_loss + f_w * f_loss

        # --- Logging & Saving ---
        if getattr(config.wandb, "use_wandb", True):
            wandb.log({
                "u_loss": u_loss.item(), 
                "f_loss": f_loss.item(), 
                "total_loss": total_loss.item(),
                "mode": "Model" if update_model else "Corrector",
                "ic_loss": ic_loss.item(),
                "ode_loss": ode_loss.item()
            })

        if epoch % 500 == 0 or epoch == 1:
            mode_str = "Model" if update_model else "Corrector"
            print(f"Epoch {epoch:04d} [{mode_str}] | u: {u_loss.item():.3e}, f: {f_loss.item():.3e}, Total: {total_loss.item():.3e}")

        # Save Best
        if total_loss.item() < best_total:
            best_total = total_loss.item()
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": total_loss.item()
            }, finetune_dir, epoch, keep=1, name="best_finetuned_model.pt")
            
            if use_corrector:
                save_checkpoint({
                    "epoch": epoch,
                    "model_state_dict":
                    corrector.state_dict(),
                    "loss": total_loss.item(),
                    "corrector_inputs": corr_in.detach().cpu()
                }, corrector_dir, epoch, keep=1, name="best_corrector.pt")

    # — 6. LBFGS Final Polish (Optional but recommended) —
    print("\n=== Running LBFGS Polish ===")
    # Usually we polish both together or just the model. Let's polish both.
    params = list(model.parameters())
    if use_corrector:
        params += list(corrector.parameters())
    
    lbfgs = torch.optim.LBFGS(params, lr=0.1, max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        u_loss = model.u_loss(t_train, u_train)
        f_loss, ode_loss, ic_loss, _ = model.f_loss(corrector)
        loss = u_w * u_loss + f_w * ode_loss + ic_w * ic_loss
        loss.backward()
        return loss

    lbfgs.step(closure)
    final_loss = closure()
    print(f"LBFGS Final Loss: {final_loss.item():.3e}")

    # Save Final
    save_checkpoint({"model_state_dict": model.state_dict()}, finetune_dir, max_epochs, keep=1, name="final_model.pt")
    if use_corrector:
        _, _, _, final_corr_in = model.f_loss(corrector)
        save_checkpoint({
            "model_state_dict": corrector.state_dict(),
            "corrector_inputs": final_corr_in.detach().cpu()
            },
            corrector_dir, max_epochs, keep=1, name="final_corrector.pt")


def train(config, workdir):
    # Execute based on stage logic
    # DAPINN pipeline: Pretrain -> Finetune
    # Standard PINN: Just Finetune (usually)
    
    if config.run_pretrain:
        pretrain(config, workdir)
    
    if config.run_finetune:
        finetune(config, workdir)