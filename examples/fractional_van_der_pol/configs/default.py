import ml_collections
import torch
import os

def get_config():
    config = ml_collections.ConfigDict()

    config.name = "default"
    config.is_pretrained = True  # True for pretraining, False for finetuning
    config.mode = "train"  # train, eval

    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # System parameters for Fractional Van der Pol
    config.system_fvdp = ml_collections.ConfigDict({
        "system_name": "FractionalVanderPol",
        "system_params": {
            # System parameters
            "mu": 1.0,              # Damping parameter
            "alpha": 0.9,           # Fractional order (0 < alpha <= 1)
            "T": 50.0,              # Total time interval
            "x1_0": 2.0,            # Initial displacement
            "x2_0": 0.0,            # Initial velocity
            "n_t": 500,             # Number of time points
        }
    })
    
    # Dataset sample size
    config.pretrained_sample_size = 0
    config.finetune_sample_size = 10
    
    # Architectures
    config.pinns_arch = ml_collections.ConfigDict({
        "arch_name": "Mlp",
        "num_layers": 2,
        "hidden_dim": 128,
        "input_dim": 1,
        "output_dim": 2,
        "activation": "Tanh",
        "with_fourier": False,
        "fourier_emb": None,
    })    
    config.corrector_arch = ml_collections.ConfigDict({
        "arch_name": "Mlp",
        "num_layers": 1,
        "hidden_dim": 64,
        "input_dim": 4,
        "output_dim": 2,
        "activation": "Tanh",
        "with_fourier": False,
        "fourier_emb": None,
    })

    # Optimizer
    def optimizer_config(lr, optimizer, scheduler, gamma):
        return ml_collections.ConfigDict({
            "lr": lr,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "gamma": gamma
        })
    
    config.pretrain_optim = optimizer_config(lr=0.01, optimizer="Adam", scheduler="Exp", gamma=0.9)
    config.finetune_pinns_optim = optimizer_config(lr=0.01, optimizer="Adam", scheduler="Exp", gamma=0.9)
    config.finetune_correction_optim = optimizer_config(lr=0.001, optimizer="Adam", scheduler="Exp", gamma=0.9)

    # Hyperparameter
    def training_config(max_epochs, u_w, f_w, alt_steps=None):
        return ml_collections.ConfigDict({
            "max_epochs": max_epochs,
            "u_w": u_w,
            "f_w": f_w,
            "alt_steps": alt_steps,
        })

    config.pretraining = training_config(max_epochs=300, u_w=1.0, f_w=1.0)
    config.finetuning = training_config(max_epochs=6000, u_w=1.0, f_w=1e-5, alt_steps=150)
    
    # Model paths
    config.pretrained_model_name = "pretrained_model.pt" 
    config.finetuned_model_name = "lbfgs_finetuned_model.pt"
    config.corrector_model_name = "lbfgs_finetuned_corrector.pt"

    config.saving = ml_collections.ConfigDict({
        "epsilon": 1e-8,
        "pretrain_path": "pretrained",
        "finetune_path": "finetuned",
        "corrector_path": "corrector",
        "save_interval": 100,
        "save_dir": "results",
        "keep": 1,
        "early_stopping_patience": None,
    })
    
    # W&B config
    config.wandb = ml_collections.ConfigDict({
        "project": "DAPINN-FractionalVDP",
        "name": config.name,
        "tag": "pretrain" if config.is_pretrained else "finetune",
        "sample_size": config.pretrained_sample_size if config.is_pretrained else config.finetune_sample_size,
        "lr": config.pretrain_optim.lr if config.is_pretrained else config.finetune_pinns_optim.lr,
        "u_w": config.pretraining.u_w if config.is_pretrained else config.finetuning.u_w,
        "f_w": config.pretraining.f_w if config.is_pretrained else config.finetuning.f_w,
        "scheduler": config.pretrain_optim.scheduler if config.is_pretrained else config.finetune_pinns_optim.scheduler,
        "alt_steps": None if config.is_pretrained else config.finetuning.alt_steps,
    })

    config.seed = 42
    config.lhs_sampling = False
    
    return config
