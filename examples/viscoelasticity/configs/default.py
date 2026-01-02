# configs/default.py
import ml_collections
import torch
import os

def get_config():
    config = ml_collections.ConfigDict()

    # ============================================================
    # Basic
    # ============================================================
    config.name = "memory_diffusion"
    config.is_pretrained = True      # True: only diffusion, False: add corrector
    config.mode = "train"             # train / eval
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.seed = 42

    # ============================================================
    # System: Diffusion with Memory
    # PDE:
    #   ∂_t u = D ∂_xx u + ∫_0^t K(t-s) u(x,s) ds
    # ============================================================
    config.system_memory = ml_collections.ConfigDict({
        "system_name": "MemoryDiffusion",
        "system_params": {
            "D": 0.1,                 # diffusion coefficient
            "alpha": 5.0,    # memory decay rate (for generating ground truth data)

            "T": 1.0,                 # final time
            "L": 1.0,                 # spatial domain [0, L]
            "nt": 200,               # temporal resolution
            "nx": 100,               # spatial resolution

            # Initial condition u(x,0)
            "ic_type": "sin",         # sin(pi x)
            
            # Memory kernel (ground truth, PINN does NOT know this)
            "kernel_type": "exp",     # exp(-(t-s))
            "kernel_decay": 5.0,      # alpha in exp(-alpha (t-s))

            # Noise level (optional)
            "noise": 0.0,
        }
    })

    # ============================================================
    # Dataset sampling
    # ============================================================
    config.pretrained_sample_size = 0      # physics-only
    config.finetune_sample_size = 2000     # sparse observation points

    # ============================================================
    # PINNs architecture: u(x,t)
    # ============================================================
    config.pinns_arch = ml_collections.ConfigDict({
        "arch_name": "Mlp",
        "num_layers": 4,
        "hidden_dim": 64,
        "input_dim": 2,        # (x, t)
        "output_dim": 1,       # u
        "activation": "Tanh",
        "with_fourier": False,
        "fourier_emb": None,
    })

    # ============================================================
    # Corrector architecture
    # sψ(u, u_t, u_xx)
    # ============================================================
    config.corrector_arch = ml_collections.ConfigDict({
        "arch_name": "Mlp",
        "num_layers": 2,
        "hidden_dim": 64,
        "input_dim": 4,        # (u, u_t, u_x, u_xx)
        "output_dim": 1,
        "activation": "Tanh",
        "with_fourier": False,
        "fourier_emb": None,
    })

    # ============================================================
    # Optimizers
    # ============================================================
    def optimizer_config(lr, optimizer, scheduler, gamma):
        return ml_collections.ConfigDict({
            "lr": lr,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "gamma": gamma,
        })

    config.pretrain_optim = optimizer_config(
        lr=1e-3, optimizer="Adam", scheduler="Exp", gamma=0.9
    )
    config.finetune_pinns_optim = optimizer_config(
        lr=1e-3, optimizer="Adam", scheduler="Exp", gamma=0.9
    )
    config.finetune_correction_optim = optimizer_config(
        lr=1e-3, optimizer="Adam", scheduler="Exp", gamma=0.9
    )

    # ============================================================
    # Training hyperparameters
    # ============================================================
    def training_config(max_epochs, u_w, f_w, alt_steps=None):
        return ml_collections.ConfigDict({
            "max_epochs": max_epochs,
            "u_w": u_w,
            "f_w": f_w,
            "alt_steps": alt_steps,
        })

    # Pretrain: pure diffusion
    config.pretraining = training_config(
        max_epochs=6000,
        u_w=1.0,
        f_w=1.0,
    )

    # Finetune: add corrector
    config.finetuning = training_config(
        max_epochs=12000,
        u_w=1.0,
        f_w=1e-5,
        alt_steps=200,
    )

    # ============================================================
    # Saving
    # ============================================================
    config.pretrained_model_name = "pretrained_model.pt"
    config.finetuned_model_name = "lbfgs_finetuned_model.pt"
    config.corrector_model_name = "lbfgs_finetuned_corrector.pt"

    config.saving = ml_collections.ConfigDict({
        "epsilon": 1e-8,
        "pretrain_path": "pretrained",
        "finetune_path": "finetuned",
        "corrector_path": "corrector",
        "save_interval": 5000,
        "save_dir": f"results{os.sep}memory_diffusion",
        "keep": 3,
        "early_stopping_patience": 8000,
    })

    # ============================================================
    # Weights & Biases
    # ============================================================
    config.wandb = None

    return config


if __name__ == "__main__":
    pass
