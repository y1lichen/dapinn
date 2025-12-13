import numpy as np
import torch
from scipy.integrate import solve_ivp

# Inviscid Burgers' Equation Dataset Generator

def apply_ic(u):
    return -np.sin(np.pi * u)  # Example initial condition

def apply_bc(u, u_L, u_R, t):
    u[0] = u_L(t)
    u[-1] = u_R(t)
    return u

def generate_inviscid_burgers_dataset(
    params, T=1.0, n_x=100, n_t=100, 
    u0_func=apply_ic, u_L=lambda t: 0.0, u_R=lambda t: 0.0
):
    a, b = params['a'], params['b']
    x = np.linspace(a, b, n_x)
    dx = x[1] - x[0]
    u0 = u0_func(x)

    u_L = lambda t: apply_ic(t * 0 + a)
    u_R = lambda t: apply_ic(t * 0 + b)

    def inviscid_burgers(t, u):
        u = apply_bc(u.copy(), u_L, u_R, t)
        dudx = np.zeros_like(u)
        for i in range(1, len(u) - 1):
            if u[i] > 0:
                dudx[i] = (u[i] - u[i - 1]) / dx  # backward
            else:
                dudx[i] = (u[i + 1] - u[i]) / dx  # forward
        dudt = -u * dudx
        return dudt

    t_eval = np.linspace(0, T, n_t)
    sol = solve_ivp(inviscid_burgers, [0, T], u0, t_eval=t_eval, method='RK45')

    t_tensor = torch.tensor(sol.t).float().reshape(-1, 1)       # [n_t, 1]
    x_tensor = torch.tensor(x).float().reshape(-1, 1)           # [n_x, 1]
    u_tensor = torch.tensor(sol.y).float()                      # [n_x, n_t] 

    return x_tensor, t_tensor, u_tensor

# Viscous Burgers' Equation Dataset Generator
def generate_viscous_burgers_dataset(
    params, T=1.0, n_x=100, n_t=100, 
    nu=0.01, u0_func=apply_ic, u_L=lambda t: 0.0, u_R=lambda t: 0.0
):
    a, b = params['a'], params['b'], 
    mode = params['mode']
    x = np.linspace(a, b, n_x)
    dx = x[1] - x[0]
    u0 = u0_func(x)

    u_L = lambda t: apply_ic(t * 0 + a)
    u_R = lambda t: apply_ic(t * 0 + b)
    
    def viscous_burgers(t, u):
        u = apply_bc(u.copy(), u_L, u_R, t)
        dudx = np.zeros_like(u)
        d2udx2 = np.zeros_like(u)

        # Central difference for first derivative (advection)
        dudx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)

        # Second-order central difference for second derivative (diffusion)
        d2udx2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2

        dudt = -u * dudx + nu * d2udx2
        return dudt

    t_eval = np.linspace(0, T, n_t)
    sol = solve_ivp(viscous_burgers, [0, T], u0, t_eval=t_eval, method='RK45')

    t_tensor = torch.tensor(sol.t).float().reshape(-1, 1)       # [n_t, 1]
    x_tensor = torch.tensor(x).float().reshape(-1, 1)           # [n_x, 1]
    u_tensor = torch.tensor(sol.y).float()                      # [n_x, n_t]
    
    # Add Gaussian noise if 'noise' is specified in params
    if 'noise' in params and mode == "train":
        noise_level = params['noise']
        noise = noise_level * torch.randn_like(u_tensor)
        u_tensor = u_tensor + noise

    u_x = np.zeros_like(u_tensor)
    u_x[1:-1, :] = (u_tensor[2:, :] - u_tensor[:-2, :]) / (2 * dx)
    u_x_tensor = torch.tensor(u_x).float()

    u_xx = np.zeros_like(u_tensor)
    u_xx[1:-1, :] = (u_tensor[2:, :] - 2 * u_tensor[1:-1, :] + u_tensor[:-2, :]) / dx**2
    u_xx_tensor = torch.tensor(u_xx).float()

    return x_tensor, t_tensor, u_tensor, u_x_tensor, u_xx_tensor               
