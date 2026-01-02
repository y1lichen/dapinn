from dapinns.models import BasePinns, BaseCorrector
import torch
import os
from pyDOE import lhs


# -------------------------------------------------
# Corrector (Pointwise, memoryless)
# -------------------------------------------------
class Corrector(BaseCorrector):
    def __init__(self, config):
        super().__init__(config)

    def load_corrector_model(self, checkpoint_dir=None, corrector_model_name=None):
        if corrector_model_name:
            checkpoint_dir = os.path.join(checkpoint_dir, corrector_model_name)
        else:
            checkpoint_dir = os.path.join(checkpoint_dir, self.config.corrector_model_name)

        checkpoint = torch.load(
            checkpoint_dir,
            map_location=self.config.device,
            weights_only=True
        )
        self.load_state_dict(checkpoint["model_state_dict"])


# -------------------------------------------------
# Memory Diffusion PINN (misspecified on purpose)
# -------------------------------------------------
class MemoryDiffusionPINN(BasePinns):
    """
    True PDE:
        u_t = D u_xx + ∫_0^t K(t-s) u(x,s) ds

    Encoded PDE (misspecified):
        u_t - D u_xx + s_ψ = 0
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        params = config.system_memory.system_params

        self.D = params["D"]
        self.T = params["T"]

        # Initial condition u(x,0) = u0(x)
        ic_type = params.get("ic_type", "sin")
        if ic_type == "sin":
            self.u0 = lambda x: torch.sin(torch.pi * x)
        else:
            raise ValueError(f"Unknown ic_type: {ic_type}")
        # Collocation points
        N_f = params.get("N_f", 20000)
        lhs_sample = lhs(n=2, samples=N_f)

        x = lhs_sample[:, 0:1]
        t = lhs_sample[:, 1:2]

        self.x = torch.tensor(x, dtype=torch.float32).to(config.device)
        self.t = torch.tensor(t * self.T, dtype=torch.float32).to(config.device)

    # -------------------------------------------------
    # Data loss (optional, scarce observations)
    # -------------------------------------------------
    def u_loss(self, X, y_true):
        y_pred = self(X)
        return torch.mean((y_pred - y_true) ** 2)

    # -------------------------------------------------
    # Physics loss
    # -------------------------------------------------
    def f_loss(self, corrector=None):

        self.x.requires_grad = True
        self.t.requires_grad = True

        X = torch.cat([self.x, self.t], dim=1)
        u = self(X)

        # First derivatives
        u_t = torch.autograd.grad(
            u, self.t, torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]

        u_x = torch.autograd.grad(
            u, self.x, torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]

        # Second derivative
        u_xx = torch.autograd.grad(
            u_x, self.x, torch.ones_like(u_x),
            create_graph=True
        )[0]

        # ------------------------------
        # Pretraining: no correction
        # ------------------------------
        correction_input = None
        if self.config.is_pretrained:
            residual = u_t - self.D * u_xx

        # ------------------------------
        # Fine-tuning: pointwise correction
        # ------------------------------
        else:
            correction_input = torch.cat(
                [u, u_t, u_x, u_xx],
                dim=1
            )
            s = corrector(correction_input)
            residual = u_t - self.D * u_xx + s

        physics_loss = torch.mean(residual ** 2)

        # ------------------------------
        # Initial condition loss
        # ------------------------------
        t0_mask = (self.t == 0).flatten()
        if t0_mask.any():
            u0_pred = u[t0_mask]
            x0 = self.x[t0_mask]
            u0_true = self.u0(x0)
            ic_loss = torch.mean((u0_pred - u0_true) ** 2)
        else:
            ic_loss = 0.0

        total_loss = physics_loss + 1e3 * ic_loss
        return (total_loss, correction_input)
    
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