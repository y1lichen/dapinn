import torch
import torch.nn as nn
from dapinns.models import BasePinns, BaseCorrector
from examples.pedagogical_example.models import Pedagogical
import os
import math

class PedagogicalBaselineComaprison(Pedagogical):
    def __init__(self, config):
        super().__init__(config)
    
    def f_function(self, t, lambda_param, u):
        # 這裡定義的是物理模型中的「右式項」(f + misspecified reaction)
        # 根據 Case (B)/(C): f(t) + lambda * cos(u) [cite: 310, 311]
        return torch.sin(3 * math.pi * t) + lambda_param * torch.cos(u)
    
    def f_loss(self, corrector=None):
        device = next(self.parameters()).device
        t = self.t_col
        t.requires_grad = True

        u = self(t)
        # 自動微分得到 du/dt (對應物理算子中的微分部分) [cite: 110, 115]
        du = torch.autograd.grad(
            u, t, torch.ones_like(u).to(device),
            create_graph=True, retain_graph=True
        )[0]

        # 這裡得到 f(t) + lambda * cos(u)
        f_total_hypo = self.f_function(t, self.lam, u).to(device)

        if corrector is None:
            # 標準 PINN: 殘差 = du/dt - (f + lambda*cos(u)) [cite: 121]
            residual = du - f_total_hypo
            corrections_inputs = None
        else:
            # CMPINN: 根據 Eq (4)，將校正項 s 加入算子中 
            # 物理上這代表：du/dt = f(t)+ lambda*cos(u) + s 
            s = corrector(t) # Corrector 的輸入是座標 t [cite: 192, 235]
            residual = du - (f_total_hypo + s) 

        # Eq (4): 物理殘差的均方誤差 
        ode_loss = torch.mean(residual**2)
        
        # Eq (1b): 初期條件損失 (這部分對應論文中的 b_theta 或邊界數據) [cite: 113, 117]
        ic_loss = torch.mean((u[0] - self.u0)**2)

        total_loss = ode_loss + ic_loss
        # return ode_loss, ic_loss, t
        return total_loss, t

class Corrector(BaseCorrector):
    def __init__(self, config):
        super().__init__(config)
    
    def load_corrector_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=True)
        self.load_state_dict(checkpoint["model_state_dict"])