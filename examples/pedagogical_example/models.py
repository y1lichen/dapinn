import torch
import torch.nn as nn
from dapinns.models import BasePinns, BaseCorrector
import os
import math

class Pedagogical(BasePinns):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        p = config.system_pedagogical.system_params
        self.lam = p["lambda"]
        self.u0 = torch.tensor([[p["u0"]]], dtype=torch.float32).to(config.device)

        # Collocation points for Physics Loss
        # 使用更多的點來確保物理約束在整個域內成立
        T = p["T"]
        self.t_col = torch.linspace(0, T, 1000).reshape(-1, 1).to(config.device)

    def f_function(self, t, lambda_param, u):
        # Incomplete Physics Model: du/dt = f(t)
        # 這裡只返回 f(t)，不包含 lambda * u * (1-u)
        return torch.sin(2 * math.pi * t) 

    # ----------------------------------------------------------------------
    # Data Loss (Eq 3.2 first term)
    # ----------------------------------------------------------------------
    def u_loss(self, t, u_measurements):
        u_pred = self(t)
        return torch.mean((u_pred - u_measurements)**2)

    # ----------------------------------------------------------------------
    # Physics Loss (Eq 3.1 & 3.3)
    # ----------------------------------------------------------------------
    def f_loss(self, corrector=None):
        device = next(self.parameters()).device
        t = self.t_col
        t.requires_grad = True

        u = self(t)
        du = torch.autograd.grad(
            u, t, torch.ones_like(u).to(device),
            create_graph=True, retain_graph=True
        )[0]

        f_t = self.f_function(t, self.lam, u).to(device)

        # 根據是否使用 Corrector 決定 Residual 的定義
        if corrector is None:
            # Case 1: Standard PINN (Baseline) or Pre-training
            # Assumption: du/dt = f(t)
            # Residual = du/dt - f(t)
            residual = du - f_t
            corrections_inputs = None
        else:
            # Case 2: DAPINN Fine-tuning (Eq 3.3)
            # Assumption: du/dt = f(t) + s_psi
            # Residual = du/dt - f(t) - s_psi
            
            # Input to ADPC: s_psi(u, du, ...)
            corrections_inputs = torch.cat([u, du], dim=1)
            s = corrector(corrections_inputs)
            
            residual = du - (f_t + s)

        # Eq 3.3: Mean Squared Error of Residual
        ode_loss = torch.mean(residual**2)
        
        # Eq 3.1: Initial Condition Loss
        ic_loss = torch.mean((u[0] - self.u0)**2)

        total_physics_loss = ode_loss + ic_loss

        return total_physics_loss, corrections_inputs

    def load_pretrained_model(self, checkpoint_dir=None):
        # Logic strictly for loading the model trained in the Pre-training stage
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(self.config.workdir, self.config.saving.save_dir, "pretrained", "best_model.pt")
        
        try:
            checkpoint = torch.load(checkpoint_dir, map_location=self.config.device, weights_only=True)
            self.load_state_dict(checkpoint["model_state_dict"])
            print(f"[INFO] Pretrained model loaded from {checkpoint_dir}")
        except Exception as e:
            print(f"[WARN] Could not load pretrained model: {e}")

    def load_finetuned_model(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=True)
            self.load_state_dict(checkpoint["model_state_dict"])
            print(f"[INFO] Finetuned model loaded from {checkpoint_path}")
        except Exception as e:
            print(f"[ERROR] Error loading finetuned model: {e}")

class Corrector(BaseCorrector):
    def __init__(self, config):
        super().__init__(config)
    
    def load_corrector_model(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=True)
            self.load_state_dict(checkpoint["model_state_dict"])
            print(f"[INFO] Corrector model loaded from {checkpoint_path}")
        except Exception as e:
            print(f"[ERROR] Error loading corrector model: {e}")