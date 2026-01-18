import os
import time
import wandb
import torch
import numpy as np

from .models import Corrector, PedagogicalBaselineComaprison
from .utils import generate_reaction_ode_dataset
from dapinns.samplers import RandomSampler, UniformSampler
from dapinns.utils import save_checkpoint



# ==========================================
# Stage 2: Fine-tuning (Eq 3.2 - 3.4)
# ==========================================
def finetune(config, workdir):
    
    # — WandB init —
    wandb.init(
        config=dict(config.wandb),
        project=config.wandb.project,
        name=config.wandb.name,
        tags=[config.wandb.tag] if hasattr(config.wandb, "tag") else None
    )

    # — 1. Data Generation (Ground Truth) —
    p = config.system_pedagogical.system_params
    T, u0, n_t = p["T"], p["u0"], p["n_t"]
    
    # 生成 Ground Truth: du/dt = f(t) + lambda*u(1-u)
    t, u, f, sol = generate_reaction_ode_dataset(params=p, T=T, u0=u0, n_t=n_t)
    
    # Sampling measurements
    # sampler = RandomSampler(config, sample_size=config.sample_size)
    sampler = UniformSampler(sample_size=config.sample_size)
    t_train, u_train = sampler.generate_data(t, u)
    
    t_train = t_train.to(config.device)
    u_train = u_train.to(config.device)

    # — 2. Load Models —
    model = PedagogicalBaselineComaprison(config).to(config.device)
    

    # Setup Corrector
    corrector = Corrector(config).to(config.device)

    model.train()
    corrector.train()
    # — 3. Optimizers —
    # Separate optimizers for alternating updates
    all_params = list(model.parameters()) + list(corrector.parameters())
    optimizer = torch.optim.Adam(all_params, lr=config.finetune_pinns_optim.lr)
    # model.optimizer = optimizer # attach for save_checkpoint convenience


    # — 4. Training Hyperparams —
    max_epochs = config.finetuning.max_epochs
    u_w, f_w, ic_w = config.finetuning.u_w, config.finetuning.f_w, config.finetuning.ic_w
    
    save_root = os.path.join(workdir, config.saving.save_dir)
    finetune_dir = os.path.join(save_root, config.saving.finetune_path)
    corrector_dir = os.path.join(save_root, config.saving.corrector_path)
    os.makedirs(finetune_dir, exist_ok=True)
    os.makedirs(corrector_dir, exist_ok=True)

    best_total = float("inf")
    
    # — 5. Main Loop —
    print("=== Start Fine-tuning ===")
    
    for epoch in range(1, max_epochs + 1):

        optimizer.zero_grad()
        
        # Data Loss (L_u)
        u_loss = model.u_loss(t_train, u_train)
        
        # Physics Loss (L_res)
        # If corrector exists, it is used in calculation but NOT updated here
        ode_loss, ic_loss, corr_in = model.f_loss(corrector)
        
        # Eq 3.2: L_total = w_u * L_u + w_f * L_res
        total_loss = u_w * u_loss + f_w * ode_loss + ic_w * ic_loss

        total_loss.backward()
        optimizer.step()        

        # --- Logging & Saving ---
        wandb.log({
            "u_loss": u_loss.item(), 
            "ode_loss": ode_loss.item(), 
            "ic_loss": ic_loss.item(),
            "total_loss": total_loss.item(),
        })

        # Save Best
        if total_loss.item() < best_total:
            best_total = total_loss.item()
            save_checkpoint({
                "epoch": epoch, "model_state_dict": model.state_dict(), "loss": total_loss.item()
            }, finetune_dir, epoch, keep=1, name="best_finetuned_model.pt")
            
            save_checkpoint({
                "epoch": epoch, "model_state_dict": corrector.state_dict(), "loss": total_loss.item()
            }, corrector_dir, epoch, keep=1, name="best_corrector.pt")

    # — 6. LBFGS Final Polish (Optional but recommended) —
    # print("\n=== Running LBFGS Polish ===")
    # # Usually we polish both together or just the model. Let's polish both.
    # params = list(model.parameters())
    # params += list(corrector.parameters())
    
    # lbfgs = torch.optim.LBFGS(params, lr=0.1, max_iter=500, line_search_fn="strong_wolfe")

    # def closure():
    #     lbfgs.zero_grad()
    #     u_loss = model.u_loss(t_train, u_train)
    #     f_loss, _ = model.f_loss(corrector)
    #     loss = u_w * u_loss + f_w * f_loss
    #     loss.backward()
    #     return loss

    # lbfgs.step(closure)
    # final_loss = closure()
    # print(f"LBFGS Final Loss: {final_loss.item():.3e}")

    # Save Final
    save_checkpoint({"model_state_dict": model.state_dict()}, finetune_dir, max_epochs, keep=1, name="final_model.pt")
    save_checkpoint({"model_state_dict": corrector.state_dict()}, corrector_dir, max_epochs, keep=1, name="final_corrector.pt")


def train(config, workdir):
    if config.run_pretrain:
        print("Cmpinn has only one stage.")
    if config.run_finetune:
        finetune(config, workdir)