from dapinns.models import BasePinns, BaseCorrector
from dapinns.evaluators import BaseEvaluator
import torch
import os

class Corrector(BaseCorrector):
    def __init__(self, config):
        super().__init__(config)  # Ensure the parent class is initialized properly

    def load_corrector_model(self, checkpoint_dir=None, corrector_model_name=None):
        if corrector_model_name:
            checkpoint_dir = os.path.join(checkpoint_dir, corrector_model_name)
        else:
            checkpoint_dir = os.path.join(checkpoint_dir, self.config.corrector_model_name)

        try:
            # Load the corrector model checkpoint
            checkpoint = torch.load(checkpoint_dir, map_location=self.config.device, weights_only=True)
            self.load_state_dict(checkpoint["model_state_dict"])
            print(f"Corrector model loaded from {checkpoint_dir}")
        except FileNotFoundError:
            print(f"Checkpoint file not found at {checkpoint_dir}. Please check the path.")
        except KeyError as e:
            print(f"Key error while loading checkpoint: {e}. Ensure the checkpoint contains 'model_state_dict'.")
        except Exception as e:
            print(f"Error loading corrector model: {e}")

class Qdho(BasePinns):
    def __init__(self, config):
        super().__init__(config)  # Initialize BasePinns
        self.config = config

        params = config.system_qdho.system_params  # Extract system parameters

        # System parameters
        self.m = params['m']
        self.k = params['k']

        # Initial condition
        self.x0 = params['x0']
        self.v0 = params['v0']
        self.T = params['T']
        self.t = torch.linspace(0, self.T, 10000).reshape(-1, 1).to(config.device)  # Collocation points

        self.val_t = torch.linspace(0, self.T, 1000).reshape(-1, 1).to(config.device)  # Collocation points
    def u_loss(self, x, y):
        y_pred = self(x).reshape(-1, 1)  # Calls the forward method
        residual = y_pred - y

        return torch.mean(residual**2)

    def f_loss(self, corrector=None):

        self.t.requires_grad = True
        device = next(self.parameters()).device  # Get the device of the model

        y = self(self.t)  # Calls the forward method
        dy = torch.autograd.grad(y, self.t, torch.ones_like(y).to(device), create_graph=True, retain_graph=True)[0]
        dy2 = torch.autograd.grad(dy, self.t, torch.ones_like(dy).to(device), create_graph=True)[0]
 
        if self.config.is_pretrained:  # Pre-training
            residual = self.m * dy2 + self.k * y
            
        else:  # Fine-tuning
            corrections_inputs = torch.cat([y, dy, dy2], dim=1)
            s = corrector(corrections_inputs)
            residual = self.m * dy2 + self.k * y + s

        ode_loss = torch.mean(residual**2)
        x0 = torch.tensor([[self.x0]], dtype=torch.float32).to(device)
        v0 = torch.tensor([[self.v0]], dtype=torch.float32).to(device)

        ic_loss_x = torch.mean((y[0] - x0)**2)
        ic_loss_v = torch.mean((dy[0] - v0)**2)
        total = ode_loss + ic_loss_x * 1e4 + ic_loss_v

        return (total, corrections_inputs) if corrector else total
    
    def val_f_loss(self, corrector=None):

        self.val_t.requires_grad = True
        device = next(self.parameters()).device  # Get the device of the model

        y = self(self.val_t)  # Calls the forward method
        dy = torch.autograd.grad(y, self.val_t, torch.ones_like(y).to(device), create_graph=True, retain_graph=True)[0]
        dy2 = torch.autograd.grad(dy, self.val_t, torch.ones_like(dy).to(device), create_graph=True)[0]
 
        if self.config.is_pretrained:  # Pre-training
            residual = self.m * dy2 + self.k * y
            
        else:  # Fine-tuning
            corrections_inputs = torch.cat([y, dy, dy2], dim=1)
            s = corrector(corrections_inputs)
            residual = self.m * dy2 + self.k * y + s

        ode_loss = torch.mean(residual**2)
        x0 = torch.tensor([[self.x0]], dtype=torch.float32).to(device)
        v0 = torch.tensor([[self.v0]], dtype=torch.float32).to(device)

        ic_loss_x = torch.mean((y[0] - x0)**2)
        ic_loss_v = torch.mean((dy[0] - v0)**2)
        total = ode_loss + ic_loss_x * 1e4 + ic_loss_v

        return total
        
    def load_pretrained_model(self, checkpoint_dir=None):

        checkpoint_dir = os.path.join(checkpoint_dir)
        try:
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_dir, map_location=self.config.device, weights_only=True)
            self.load_state_dict(checkpoint["model_state_dict"])
            # print(f"Pretrained model loaded from {checkpoint_dir}")
        except FileNotFoundError:
            print(f"Checkpoint file not found at {checkpoint_dir}. Please check the path.")
        except KeyError as e:
            print(f"Key error while loading checkpoint: {e}. Ensure the checkpoint contains 'model_state_dict'.")
        except Exception as e:
            print(f"Error loading pretrained model: {e}")

    def load_finetuned_model(self, checkpoint_dir=None):
            
        checkpoint_dir = os.path.join(checkpoint_dir)
        try:
            # Load the finetuned model checkpoint
            checkpoint = torch.load(checkpoint_dir, map_location=self.config.device, weights_only=True)
            self.load_state_dict(checkpoint["model_state_dict"])
            print(f"Finetuned model loaded from {checkpoint_dir}")
        except FileNotFoundError:
            print(f"Checkpoint file not found at {checkpoint_dir}. Please check the path.")
        except KeyError as e:
            print(f"Key error while loading checkpoint: {e}. Ensure the checkpoint contains 'model_state_dict'.")
        except Exception as e:
            print(f"Error loading finetuned model: {e}")

if __name__ == '__main__':

    pass