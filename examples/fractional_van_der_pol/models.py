from dapinns.models import BasePinns, BaseCorrector
from dapinns.evaluators import BaseEvaluator
import torch
import os


class Corrector(BaseCorrector):
    def __init__(self, config):
        super().__init__(config)

    def load_corrector_model(self, checkpoint_dir=None, corrector_model_name=None):
        if corrector_model_name:
            checkpoint_dir = os.path.join(checkpoint_dir, corrector_model_name)
        else: 
            checkpoint_dir = os.path.join(checkpoint_dir, self.config.corrector_model_name)
        try:
            checkpoint = torch.load(checkpoint_dir, map_location=self.config.device, weights_only=True)
            self.load_state_dict(checkpoint["model_state_dict"])
            print(f"Corrector model loaded from {checkpoint_dir}")
        except FileNotFoundError:
            print(f"Checkpoint file not found at {checkpoint_dir}. Please check the path.")
        except KeyError as e:
            print(f"Key error while loading checkpoint: {e}. Ensure the checkpoint contains 'model_state_dict'.")
        except Exception as e:
            print(f"Error loading corrector model: {e}")


class FractionalVanderPol(BasePinns):
    """
    Physics-informed neural network for fractional Van der Pol equation:
    d^α x1/dt^α = x2(t)
    d^α x2/dt^α = -x1(t) - μ(x1^2 - 1)x2(t)
    where 0 < α ≤ 1
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        params = config.system_fvdp.system_params

        # System parameters
        self.mu = params['mu']
        self.alpha = params.get('alpha', 0.9)
        
        # Initial conditions
        x1_0 = params['x1_0']
        x2_0 = params['x2_0']
        self.x1_0 = torch.tensor([[x1_0]], dtype=torch.float32).to(config.device)
        self.x2_0 = torch.tensor([[x2_0]], dtype=torch.float32).to(config.device)
        
        self.T = params['T']

        # Collocation points
        self.t = torch.linspace(0, self.T, 10000).reshape(-1, 1).to(config.device)

    def u_loss(self, t, x1, x2):
        """Data loss on observed points for both x1 and x2"""
        x1_pred, x2_pred = self(t)[:, [0]], self(t)[:, [1]]
        
        loss = torch.mean((x1_pred - x1)**2) + torch.mean((x2_pred - x2)**2)
        return loss

    def f_loss(self, corrector=None):
        """Physics-informed loss"""
        self.t.requires_grad = True
        device = next(self.parameters()).device

        # Forward pass - get [x1, x2]
        x = self(self.t)
        x1 = x[:, [0]]
        x2 = x[:, [1]]
        
        # First derivatives
        dx1_dt = torch.autograd.grad(
            x1, self.t, torch.ones_like(x1),
            create_graph=True, retain_graph=True
        )[0]
        
        dx2_dt = torch.autograd.grad(
            x2, self.t, torch.ones_like(x2),
            create_graph=True, retain_graph=True
        )[0]
        
        # Physics residuals
        # d^α x1/dt^α = x2
        residual1 = dx1_dt - x2
        
        # d^α x2/dt^α = -x1 - μ(x1^2 - 1)x2
        residual2 = dx2_dt + x1 + self.mu * (x1**2 - 1) * x2
        
        if self.config.is_pretrained:
            f_loss = torch.mean(residual1**2 + residual2**2)
        else:
            # With corrector
            corrections_inputs = torch.cat([x1, x2, dx1_dt, dx2_dt], dim=1)
            s1, s2 = corrector(corrections_inputs)
            residual1 = dx1_dt - x2 - s1
            residual2 = dx2_dt + x1 + self.mu * (x1**2 - 1) * x2 - s2
            f_loss = torch.mean(residual1**2 + residual2**2)
        
        # Initial condition losses
        x1_ic = self(torch.tensor([[0.0]], device=device))[:, [0]]
        x2_ic = self(torch.tensor([[0.0]], device=device))[:, [1]]
        
        ic_loss = torch.mean((x1_ic - self.x1_0)**2 + (x2_ic - self.x2_0)**2)
        
        total_loss = f_loss + ic_loss * 1e4
        
        if self.config.is_pretrained:
            return total_loss
        else:
            return total_loss, corrections_inputs

    def load_pretrained_model(self, checkpoint_dir=None):
        checkpoint_dir = os.path.join(checkpoint_dir)
        try:
            checkpoint = torch.load(checkpoint_dir, map_location=self.config.device, weights_only=True)
            self.load_state_dict(checkpoint["model_state_dict"])
        except FileNotFoundError:
            print(f"Checkpoint file not found at {checkpoint_dir}. Please check the path.")
        except KeyError as e:
            print(f"Key error while loading checkpoint: {e}. Ensure the checkpoint contains 'model_state_dict'.")
        except Exception as e:
            print(f"Error loading pretrained model: {e}")

    def load_finetuned_model(self, checkpoint_dir=None):
        checkpoint_dir = os.path.join(checkpoint_dir)
        try:
            checkpoint = torch.load(checkpoint_dir, map_location=self.config.device, weights_only=True)
            self.load_state_dict(checkpoint["model_state_dict"])
        except FileNotFoundError:
            print(f"Checkpoint file not found at {checkpoint_dir}. Please check the path.")
        except KeyError as e:
            print(f"Key error while loading checkpoint: {e}. Ensure the checkpoint contains 'model_state_dict'.")
        except Exception as e:
            print(f"Error loading finetuned model: {e}")
