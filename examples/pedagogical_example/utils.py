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
    f_func = lambda t: np.sin(3 * np.pi * t)

    def ode_func(t, u):
        return f_func(t) + lam * u * (1 - u)

    t_vals = torch.linspace(0, T, n_t).reshape(-1, 1)
    sol = solve_ivp(ode_func, [t_vals[0].item(), t_vals[-1].item()],
                    [u0], t_eval=t_vals.ravel())

    t = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1)
    u = torch.tensor(sol.y[0], dtype=torch.float32).reshape(-1, 1)
    f = torch.tensor(f_func(sol.t), dtype=torch.float32).reshape(-1, 1)

    return t, u, f, sol


# ----------------------------
# Modified ODE without zero-order term
# du/dt = f(t)
# ----------------------------
def generate_no_reaction_ode_dataset(params, T=1.0, u0=0.0, n_t=101):

    f_func = lambda t: np.sin(3 * np.pi * t)

    def ode_func(t, u):
        return f_func(t)  # no λu(1-u) term

    t_vals = torch.linspace(0, T, n_t).reshape(-1, 1)
    sol = solve_ivp(ode_func, [t_vals[0].item(), t_vals[-1].item()],
                    [u0], t_eval=t_vals.ravel())

    t = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1)
    u = torch.tensor(sol.y[0], dtype=torch.float32).reshape(-1, 1)
    f = torch.tensor(f_func(sol.t), dtype=torch.float32).reshape(-1, 1)

    return t, u, f, sol


if __name__ == '__main__':
    # Parameters
    params = {'lambda': 1.0}
    T = 1.0
    u0 = 0.0
    n_t = 101

    # Baseline (with reaction term)
    t1, u1, f1, _ = generate_reaction_ode_dataset(params, T=T, u0=u0, n_t=n_t)

    # Experiment (without reaction term)
    t2, u2, f2, _ = generate_no_reaction_ode_dataset(params, T=T, u0=u0, n_t=n_t)

    print('With reaction term:')
    print(t1.shape, u1.shape, f1.shape)
    print(u1[:5])

    print('\nWithout reaction term:')
    print(t2.shape, u2.shape, f2.shape)
    print(u2[:5])
