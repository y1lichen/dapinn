import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from scipy.integrate import solve_ivp
import torch
import numpy as np

# ----------------------------
# Original ODE with reaction term
# du/dt = f(t) + λ * u * (1 - u)
# f(t) = sin(2π t)
# ----------------------------
def generate_reaction_ode_dataset(params, T=1.0, u0=0.0, n_t=101):

    lam = params['lambda']
    f_func = lambda t: np.sin(2 * np.pi * t)

    def ode_func(t, u):
        return f_func(t) + lam * u * (1 - u)

    t_vals = torch.linspace(0, T, n_t).reshape(-1, 1)
    sol = solve_ivp(ode_func, [t_vals[0].item(), t_vals[-1].item()],
                    [u0], t_eval=t_vals.ravel())

    t = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1)
    u = torch.tensor(sol.y[0], dtype=torch.float32).reshape(-1, 1)
    f = torch.tensor(f_func(sol.t), dtype=torch.float32).reshape(-1, 1)

    return t, u, f, sol