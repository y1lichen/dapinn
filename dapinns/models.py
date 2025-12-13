from dapinns import archs
import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod

class ExponentialLRWithMin(optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, min_lr=1e-6, last_epoch=-1):
        self.min_lr = min_lr
        super().__init__(optimizer, gamma, last_epoch)

    def get_lr(self):
        return [
            max(lr, self.min_lr)
            for lr in super().get_lr()
        ]
    
def _get_activation_function(name):
    activations = {
        "ReLU": nn.ReLU,
        "Sigmoid": nn.Sigmoid,
        "Tanh": nn.Tanh,
        "LeakyReLU": nn.LeakyReLU,
        "ELU": nn.ELU,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    return activations[name]

def _initialize_pinns_archs(config):
    if config.arch_name == "Mlp":
        return archs.Mlp(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            activation=_get_activation_function(config.activation),  # Map string to function
            with_fourier=config.with_fourier,
            fourier_emb=config.fourier_emb,  # Pass the Fourier embedding config
        )
    
    elif config.arch_name == "ModifiedMlp":
        pass

    else:
        raise ValueError(f"Unknown architecture: {config.arch_name}")
    
def _initialize_corrector_archs(config):
    if config.arch_name == "Mlp":
        return archs.Mlp(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            activation=_get_activation_function(config.activation),  # Map string to function
            with_fourier=False,
            fourier_emb=None,  # No Fourier embedding for the corrector
        )
    
    elif config.arch_name == "ModifiedMlp":
        pass

    else:
        raise ValueError(f"Unknown architecture: {config.arch_name}")

def _initialize_optimizer(config, model_params, is_pretrained=False):
    return optim.Adam(model_params, lr=config.lr)

def _initialize_scheduler(config, optimizer):
    if config.scheduler == "Exp":
        return ExponentialLRWithMin(optimizer, gamma=config.gamma, min_lr=1e-6)
    elif config.scheduler == "Cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)
    elif config.scheduler == "Reduce":
        return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',              # 根據 loss 是否減少判定是否調整 learning rate
        factor=0.1,              # 將學習率乘上 0.1
        patience=3000,           # 超過 3000 步（或 epoch）沒有改善就降 lr
        min_lr=1e-6,             # 最低學習率限制
    )
    else:
        raise NotImplementedError(f"Scheduler {config.scheduler} not implemented.")
    
class BasePinns(nn.Module, ABC):  # Inherit from torch.nn.Module and ABC
    def __init__(self, config):
        super(BasePinns, self).__init__()  # Initialize torch.nn.Module
        self.config = config
        self.model = _initialize_pinns_archs(config.pinns_arch)

        if config.is_pretrained:
            self.optimizer = _initialize_optimizer(config.pretrain_optim, self.model.parameters(), is_pretrained=config.is_pretrained)
            self.scheduler = _initialize_scheduler(config.pretrain_optim, self.optimizer)
        else:
            self.optimizer = _initialize_optimizer(config.finetune_pinns_optim, self.model.parameters())
            self.scheduler = _initialize_scheduler(config.finetune_pinns_optim, self.optimizer)

    @abstractmethod
    def u_loss(self):
        raise NotImplementedError
    
    @abstractmethod
    def f_loss(self):
        raise NotImplementedError
        
    @abstractmethod
    def load_pretrained_model(self, checkpoint_dir):
        raise NotImplementedError

    @abstractmethod
    def load_finetuned_model(self, checkpoint_dir):
        raise NotImplementedError
    
    def forward(self, x):
        return self.model(x)
    
class BaseCorrector(nn.Module, ABC):  # Inherit from torch.nn.Module and ABC
    def __init__(self, config):
        super(BaseCorrector, self).__init__()  # Initialize torch.nn.Module
        self.config = config
        self.model = _initialize_corrector_archs(config.corrector_arch)
        self.optimizer = _initialize_optimizer(config.finetune_correction_optim, self.model.parameters())
        self.scheduler = _initialize_scheduler(config.finetune_correction_optim, self.optimizer)

    def forward(self, x):
        return self.model(x)  # Ensure the forward method is defined

