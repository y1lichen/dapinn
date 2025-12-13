import ml_collections
import torch
import os

def get_config():
    config = ml_collections.ConfigDict()

    config.name = "pedagogical_default"
    config.mode = "train"  # train, eval

    # -----------------------------
    # Flow Control (Defaults)
    # -----------------------------
    config.run_pretrain = False
    config.run_finetune = True
    config.load_pretrained = False
    config.use_corrector = False
    
    # [CRITICAL] Legacy flag required by BasePinns
    config.is_pretrained = False 

    # -----------------------------
    # Device: CUDA > MPS > CPU
    # -----------------------------
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    config.device = device
    print(f"[CONFIG] Using device: {device}")

    # -----------------------------
    # System parameters for Pedagogical ODE
    # -----------------------------
    config.system_pedagogical = ml_collections.ConfigDict({
        "system_name": "PedagogicalODE",
        "system_params": {
            # "lambda": 1.0,
            "lambda": 0.2,
            "u0": 0.0,
            "T": 1.0,
            "n_t": 101,
        }
    })

    # -----------------------------
    # Dataset sample size
    # -----------------------------
    config.sample_size = 50
    config.noise = 0.0

    # -----------------------------
    # Model architectures
    # -----------------------------
    config.pinns_arch = ml_collections.ConfigDict({
        "arch_name": "Mlp",
        "num_layers": 2,
        "hidden_dim": 50,
        "input_dim": 1,
        "output_dim": 1,
        "activation": "Tanh",
        "with_fourier": False,
        "fourier_emb": None,
    })

    config.corrector_arch = ml_collections.ConfigDict({
        "arch_name": "Mlp",
        "num_layers": 2, 
        "hidden_dim": 50,
        "input_dim": 2,
        "output_dim": 1,
        "activation": "Tanh",
        "with_fourier": False,
        "fourier_emb": None,
    })

    # -----------------------------
    # Optimizer [CRITICAL FIX]
    # -----------------------------
    # 必須將 optimizer config 實際寫入 config 物件中
    # BasePinns 會讀取 config.finetune_pinns_optim
    
    def optimizer_config(lr, optimizer="Adam", scheduler="Exp", gamma=0.9):
        return ml_collections.ConfigDict({
            "lr": lr,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "gamma": gamma
        })

    # 補上這兩行：
    config.finetune_pinns_optim = optimizer_config(lr=1e-3)
    config.finetune_correction_optim = optimizer_config(lr=1e-3)

    # -----------------------------
    # Training Hyperparameters
    # -----------------------------
    config.pretraining = ml_collections.ConfigDict({
        "max_epochs": 2000,
        "lr": 1e-3,
    })

    config.finetuning = ml_collections.ConfigDict({
        "max_epochs": 10000, 
        "lr": 1e-3,
        "u_w": 100.0,       
        "f_w": 1.0,         
        "alt_steps": 200,   
    })

    # -----------------------------
    # Model paths & saving
    # -----------------------------
    config.finetuned_model_name = "lbfgs_finetuned_model.pt"
    config.corrector_model_name = "lbfgs_finetuned_corrector.pt"
    
    config.saving = ml_collections.ConfigDict({
        "epsilon": 1e-8,
        "finetune_path": "finetuned",
        "corrector_path": "corrector",
        "save_interval": 1000,
        "save_dir": "results", 
        "keep": 1,
        "early_stopping_patience": 2000,
        "warmup_epochs": 500,
    })

    config.seed = 42
    config.wandb = None 

    return config