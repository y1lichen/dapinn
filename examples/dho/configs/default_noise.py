import ml_collections
import torch
import os
def get_config():
    config = ml_collections.ConfigDict()

    config.name = "default_noise"
    config.is_pretrained = True # True for pretraining, False for finetuning
    config.mode = "train" # train, eval

    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # System parameters
    config.system_uho = ml_collections.ConfigDict({
        "system_name": "UHO",
        "system_params": {
            # System parameters setting
            "m": 1.0,  # mass
            "k": 800.0,  # spring constant
            "T": 1.0,
            "x0": 1.0,
            "v0": 0.0,
            "n_t": 1000
        }
    })
    config.system_dho = ml_collections.ConfigDict({
        "system_name": "DHO",
        "system_params": {
            # System parameters setting
            "m": 1.0,  # mass
            "k": 800.0,  # spring constant
            "c": 4.0,  # damping coefficient
            "T": 1.0,
            "x0": 1.0,
            "v0": 0.0,
            "n_t": 1000,
            "noise": 0.01,  # noise level [0.01, 0.03, 0.05, 0.1]
        }
    })
    
    # Dataset sample size
    config.pretrained_sample_size = 0
    config.finetune_sample_size = 10000 # 100
    
    # Architectures
    config.pinns_arch = ml_collections.ConfigDict({
        "arch_name": "Mlp",  # "Mlp", "ModifiedMlp"
        "num_layers": 2,
        "hidden_dim": 50,
        "input_dim": 1,
        "output_dim": 1,
        "activation": "Tanh",
        "with_fourier": False,
        "fourier_emb": None,
    })    
    config.corrector_arch = ml_collections.ConfigDict({
        "arch_name": "Mlp",  # "Mlp", "ModifiedMlp"
        "num_layers": 1,
        "hidden_dim": 50,
        "input_dim": 3,
        "output_dim": 1,
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
    
    config.pretrain_optim = optimizer_config(lr=1e-3, optimizer="Adam", scheduler="Exp", gamma=0.9)
    config.finetune_pinns_optim = optimizer_config(lr=1e-3, optimizer="Adam", scheduler="Exp", gamma=0.9)
    config.finetune_correction_optim = optimizer_config(lr=1e-3, optimizer="Adam", scheduler="Exp", gamma=0.9)

    # Hyperparameter
    def training_config(max_epochs, u_w, f_w, alt_steps=None):
        return ml_collections.ConfigDict({
            "max_epochs": max_epochs,
            "u_w": u_w,
            "f_w": f_w,
            "alt_steps": alt_steps,
        })

    config.pretraining = training_config(max_epochs=100000, u_w=1, f_w=1e-6)
    config.finetuning = training_config(max_epochs=60000, u_w=1.0, f_w=1e-5, alt_steps=150)
    
    # Model paths
    config.pretrained_model_name = "pretrained_model.pt" 
    config.finetuned_model_name = "lbfgs_finetuned_model.pt"
    config.corrector_model_name = "lbfgs_finetuned_corrector.pt"

    config.saving = ml_collections.ConfigDict({
        "epsilon": 1e-8,
        "pretrain_path": "pretrained",
        "finetune_path": "finetuned",
        "corrector_path": "corrector",
        "save_interval": 6000,
        "save_dir": f"results" + os.sep + "dho",
        "keep": 5,
        "early_stopping_patience": 6000,
    })
    config.seed = 42

    # Weights & Biases
    config.wandb = None # Set function in main.py

    return config

if __name__ == "__main__":
    pass