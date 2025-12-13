import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from scipy.integrate import solve_ivp
import torch
import numpy as np

# Undamped Harmonic Oscillator
def generate_uho_dataset(params, T=1.0, x0=1, v0=0, n_t=1000):

    m = params['m']
    k = params['k']

    def simple_oscillator(t,y):
        x, v = y
        dxdt = v
        dvdt = -(k / m) * x
        return [dxdt, dvdt]
    
    t_vals = torch.linspace(0, T, n_t).reshape(-1,1) # x
    sol = solve_ivp(simple_oscillator,[t_vals[0], t_vals[-1]], [x0, v0], t_eval=t_vals.ravel())

    x = t_vals.reshape(-1,1)
    y = torch.tensor(sol.y[0]).reshape(-1,1)

    return x, y, sol

# Damped Harmonic Oscillator
def generate_dho_dataset(params, T=1.0, x0=1, v0=0, n_t=1000):

    m = params['m']
    k = params['k']
    c = params['c']

    def damped_oscillator(t, y):
        x, v = y
        dxdt = v
        dvdt = -(c / m) * v - (k / m) * x
        return [dxdt, dvdt]
    
    t_vals = torch.linspace(0, T, n_t).reshape(-1,1) # x
    sol = solve_ivp(damped_oscillator, [t_vals[0], t_vals[-1]], [x0, v0], t_eval=t_vals.ravel())
    
    x = t_vals.reshape(-1,1)
    y = torch.tensor(sol.y[0]).reshape(-1,1)

    # Add Gaussian noise if 'noise' is specified in params
    if 'noise' in params:
        noise_level = params['noise']
        y += noise_level * torch.randn_like(y)
        
    return x, y, sol

if __name__ == '__main__':

    # System parameters
    params = {'m': 1.0, 'k': 800.0, 'c': 4.0}

    # Initial Condition
    x0 = 1.0   # initial displacement
    v0 = 0.0   # initial velocity

    p_x, p_y, _  = generate_uho_dataset(params, T=1.0, x0=1.0, v0=0.0, n_t=1000)
    f_x, f_y, _ = generate_dho_dataset(params, T=1.0, x0=1.0, v0=0.0, n_t=1000)
    print(p_x.shape, p_y.shape, f_x.shape, f_y.shape)
    print(type(p_x), type(p_y), type(f_x), type(f_y) )