from dapinns.models import BasePinns, BaseCorrector
from dapinns.evaluators import BaseEvaluator
import torch
import os
from scipy.stats import qmc
import numpy as np

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
            # print(f"Corrector model loaded from {checkpoint_dir}")
        except FileNotFoundError:
            print(f"Checkpoint file not found at {checkpoint_dir}. Please check the path.")
        except KeyError as e:
            print(f"Key error while loading checkpoint: {e}. Ensure the checkpoint contains 'model_state_dict'.")
        except Exception as e:
            print(f"Error loading corrector model: {e}")

class Burgers(BasePinns):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        params = config.system_viscous_burgers.system_params
        self.device = config.device

        # System parameters
        self.nu = params['nu']
        self.T = params['T']
        self.a = params['a']
        self.b = params['b']

        self.u0_func = lambda x: -torch.sin(torch.pi * x)  # Initial condition
        self.u_L = lambda t: self.u0_func(t * 0 + self.a)
        self.u_R = lambda t: self.u0_func(t * 0 + self.b)


        # --- latin ---
        # self.n_collocation = 10000

        # sampler_f = qmc.LatinHypercube(d=2, seed=config.seed)
        # lhs_f = sampler_f.random(n=self.n_collocation)
        # x_f = lhs_f[:, 0] * (self.b - self.a) + self.a
        # t_f = lhs_f[:, 1] * self.T
        # self.collocations = torch.tensor(np.stack([x_f, t_f], axis=1), dtype=torch.float32).to(self.device)

        # sampler_ic = qmc.LatinHypercube(d=1, seed=config.seed)
        # x_ic = sampler_ic.random(n=self.n_x).flatten() * (self.b - self.a) + self.a
        # t_ic = np.zeros_like(x_ic)
        # self.ic_points = torch.tensor(np.stack([x_ic, t_ic], axis=1), dtype=torch.float32).to(self.device)
        # self.u0_true = self.u0_func(self.ic_points[:, 0:1])  # shape [n_x, 1]

        # sampler_bc = qmc.LatinHypercube(d=1, seed=config.seed)
        # t_bc = sampler_bc.random(n=self.n_t).flatten() * self.T
        # x_left = np.full_like(t_bc, self.a)
        # x_right = np.full_like(t_bc, self.b)

        # self.bc_left = torch.tensor(np.stack([x_left, t_bc], axis=1), dtype=torch.float32).to(self.device)
        # self.bc_right = torch.tensor(np.stack([x_right, t_bc], axis=1), dtype=torch.float32).to(self.device)

        # self.ul_true = self.u_L(self.bc_left[:, 1:2])
        # self.ur_true = self.u_R(self.bc_right[:, 1:2])
        # --- latin ---

        # --- grid ---
        # x = torch.linspace(self.a, self.b, self.n_x)
        # t = torch.linspace(0, self.T, self.n_t)
        self.n_x = 100
        self.n_t = 100
        x = torch.linspace(self.a, self.b, self.n_x)
        t = torch.linspace(0, self.T, self.n_t)
        xx, tt = torch.meshgrid(x, t, indexing="ij")  # shape: [n_t, n_x]
        self.collocations = torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1)  # [n_x * n_t, 2]

        self.ic_x = x.to(self.device)
        self.u0_true = self.u0_func(self.ic_x).unsqueeze(1)  # [n_x, 1]
        self.t_bc = t.to(self.device)  # already on device
        self.ul_true = self.u_L(self.t_bc).unsqueeze(1)  # [n_t, 1]
        self.ur_true = self.u_R(self.t_bc).unsqueeze(1)  # [n_t, 1]        
        # --- grid ---

        # --- val grid ---
        self.val_n_x = 256
        self.val_n_t = 100
        val_x = torch.linspace(self.a, self.b, self.val_n_x)
        val_t = torch.linspace(0, self.T, self.val_n_t)
        val_xx, val_tt = torch.meshgrid(val_x, val_t, indexing="ij")  # shape: [n_t, n_x]
        self.val_collocations = torch.stack([val_xx.reshape(-1), val_tt.reshape(-1)], dim=1)  # [n_x * n_t, 2]
        self.val_ic_x = val_x.to(self.device)
        self.val_u0_true = self.u0_func(self.val_ic_x).unsqueeze(1)  # [n_x, 1]
        self.val_t_bc = val_t.to(self.device)  # already on device
        self.val_ul_true = self.u_L(self.val_t_bc).unsqueeze(1)  # [n_t, 1]
        self.val_ur_true = self.u_R(self.val_t_bc).unsqueeze(1)  # [n_t, 1]
        # --- val grid ---

    def u_loss(self, xyt, u_true):
        """
        Compute the supervised loss on observed data.
        Inputs:
            xyt: [N, 2] tensor of (x, t) pairs
            u_true: [N, 1] ground truth solution
        """
        u_pred = self(xyt)
        loss = torch.mean((u_pred - u_true) ** 2)

        return loss

    def val_f_loss(self, corrector=None):
 
        xyt = self.val_collocations.to(next(self.parameters()).device)
        xyt.requires_grad_(True)
        u = self(xyt)
        ux_ut = torch.autograd.grad(u, xyt, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = ux_ut[:, [0]]
        u_t = ux_ut[:, [1]]
        u_xx = torch.autograd.grad(u_x, xyt, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0][:, [0]]
        
        residual = u_t + u*u_x

        if self.config.is_pretrained:
            loss_f = torch.mean(residual ** 2)
        else:
            correction_inputs = torch.cat([u, u_x, u_xx], dim=1)
            s = corrector(correction_inputs)
            corrected_residual = residual - s
            loss_f = torch.mean(corrected_residual ** 2)

        # initial condition loss
        u = u.reshape(self.val_n_x, self.val_n_t)
        u0_pred = u[:, 0].reshape(-1, 1)  # u(x, t=0)
        loss_ic = torch.mean((u0_pred - self.val_u0_true) ** 2)
        
        # boundary condition loss
        ul_pred = u[0,:].reshape(-1, 1) # u(t, x=a)
        ur_pred = u[-1,:].reshape(-1, 1) # u(t, x=b)
        loss_bc = torch.mean((ul_pred - self.val_ul_true) ** 2) + torch.mean((ur_pred - self.val_ur_true) ** 2)

        val_loss = loss_f + loss_ic + loss_bc
        return val_loss

    def f_loss(self, corrector=None):
        """
        Compute the physics-informed residual loss for Burgers’ equation:
            u_t + u * u_x - nu * u_xx = 0, s for u * u_x
        Assumes Dirichlet boundary conditions u(x, 0) = sin(pi * x).
        Optionally includes correction model output.
        """
        xyt = self.collocations.to(next(self.parameters()).device)
        xyt.requires_grad_(True)

        u = self(xyt)  # forward prediction (n_t, n_x)
        # First-order derivatives
        ux_ut = torch.autograd.grad(u, xyt, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_x = ux_ut[:, [0]]
        u_t = ux_ut[:, [1]]

        # Second-order derivatives
        # uxx_utt = torch.autograd.grad(ux_ut, xyt, grad_outputs=torch.ones_like(ux_ut), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, xyt, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0][:, [0]]
        # PDE residual
        residual = u_t + u*u_x

        if self.config.is_pretrained:
            loss_f = torch.mean(residual ** 2)
        else:
            correction_inputs = torch.cat([u, u_x, u_xx], dim=1)
            s = corrector(correction_inputs)
            corrected_residual = residual + s
            loss_f = torch.mean(corrected_residual ** 2)
        
        # initial condition loss
        u = u.reshape(self.n_x, self.n_t)
        
        u0_pred = u[:, 0].reshape(-1, 1)  # u(x, t=0)
        loss_ic = torch.mean((u0_pred - self.u0_true) ** 2)
        
        # boundary condition loss
        
        ul_pred = u[0,:].reshape(-1, 1)  # u(t, x=a)
        ur_pred = u[-1,:].reshape(-1, 1)  # u(t, x=b)
        loss_bc = torch.mean((ul_pred - self.ul_true) ** 2) + torch.mean((ur_pred - self.ur_true) ** 2)

        if self.config.is_pretrained:
            return loss_f + loss_ic + loss_bc
        else:
            return loss_f + loss_ic + loss_bc, correction_inputs

    # def f_loss(self, corrector=None):
    #     """
    #     Compute the physics-informed residual loss for Burgers’ equation:
    #         u_t + u * u_x - nu * u_xx = 0
    #     with correction model support.
    #     Initial and boundary losses are computed on LHS-sampled points.
    #     """

    #     # === Latin Hypercube Collocation points (x, t) ===
    #     xyt = self.collocations.clone().detach().requires_grad_(True).to(self.device)
    #     x = xyt[:, [0]]
    #     t = xyt[:, [1]]

    #     u = self(xyt)  # shape: [N, 1]

    #     grads = torch.autograd.grad(
    #         outputs=u, inputs=xyt, grad_outputs=torch.ones_like(u),
    #         retain_graph=True, create_graph=True
    #     )[0]
    #     u_x = grads[:, [0]]
    #     u_t = grads[:, [1]]

    #     u_xx = torch.autograd.grad(
    #         outputs=u_x, inputs=xyt, grad_outputs=torch.ones_like(u_x),
    #         create_graph=True
    #     )[0][:, [0]]
    #     residual = u_t - self.nu * u_xx

    #     # === Grid Collocation points (x, t) ===
    #     g_xyt = self.collocations.to(next(self.parameters()).device)
    #     g_xyt.requires_grad_(True)

    #     g_x = xyt[:, [0]]
    #     g_t = xyt[:, [1]]

    #     g_u = self(g_xyt)  # forward prediction (n_t, n_x)
    #     # First-order derivatives
    #     g_ux_ut = torch.autograd.grad(g_u, g_xyt, grad_outputs=torch.ones_like(g_u), retain_graph=True, create_graph=True)[0]
    #     g_u_x = g_ux_ut[:, [0]]
    #     g_u_t = g_ux_ut[:, [1]]

    #     # Second-order derivatives
    #     g_uxx_utt = torch.autograd.grad(g_ux_ut, g_xyt, grad_outputs=torch.ones_like(g_ux_ut), create_graph=True)[0]
    #     g_u_xx = g_uxx_utt[:, [0]]

    #     if self.config.is_pretrained:
    #         loss_f = torch.mean(residual ** 2)
    #     else:
    #         correction_inputs = torch.cat([g_u, g_u_x, g_u_t, g_u_xx], dim=1)
    #         s = corrector(correction_inputs)
    #         corrected_residual = residual + s
    #         loss_f = torch.mean(corrected_residual ** 2)

    #     # === Initial condition loss: u(x, 0) ≈ u0_func(x) ===
    #     u0_pred = self(self.ic_points)  # model(x, t=0)
    #     loss_ic = torch.mean((u0_pred - self.u0_true) ** 2)

    #     # === Boundary condition loss: u(a,t), u(b,t) ===
    #     ul_pred = self(self.bc_left)
    #     ur_pred = self(self.bc_right)
    #     loss_bc = torch.mean((ul_pred - self.ul_true) ** 2) + torch.mean((ur_pred - self.ur_true) ** 2)

    #     if self.config.is_pretrained:
    #         return loss_f + loss_ic + loss_bc
    #     else:
    #         return loss_f + loss_ic + loss_bc, correction_inputs
        
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
            # print(f"Finetuned model loaded from {checkpoint_dir}")
        except FileNotFoundError:
            print(f"Checkpoint file not found at {checkpoint_dir}. Please check the path.")
        except KeyError as e:
            print(f"Key error while loading checkpoint: {e}. Ensure the checkpoint contains 'model_state_dict'.")
        except Exception as e:
            print(f"Error loading finetuned model: {e}")

if __name__ == '__main__':

    pass