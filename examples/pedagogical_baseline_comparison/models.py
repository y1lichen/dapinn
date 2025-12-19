import torch
import torch.nn as nn
from dapinns.models import BasePinns, BaseCorrector
import os
from examples.pedagogical_example.models import Pedagogical
import math


class PedagogicalBaselineComaprison(Pedagogical):
    def __init__(self, config):
        super().__init__(config)
    
    def f_function(self, t, lambda_param, u):
        # Incomplete Physics Model: du/dt = f(t)
        # baseline 論文這裡使用 f(t)+lambda cos(u)
        return torch.sin(2 * math.pi * t) + lambda_param * torch.cos(u)        

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