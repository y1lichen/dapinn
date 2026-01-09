import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from scipy.integrate import solve_ivp
from scipy.special import gamma as gamma_func
import torch
import numpy as np


def generate_fractional_vanderpol_dataset(params, T=50.0, x1_0=2.0, x2_0=0.0, n_t=500, alpha=0.9):
    """
    Generate dataset for fractional Van der Pol equation using numerical integration.
    
    d^α x1/dt^α = x2(t)
    d^α x2/dt^α = -x1(t) - μ(x1^2 - 1)x2(t)
    
    Args:
        params: dict with 'mu' (damping parameter)
        T: total time (default 50.0)
        x1_0, x2_0: initial conditions (default 2.0, 0.0)
        n_t: number of time points (default 500)
        alpha: fractional order (default 0.9, 0 < alpha <= 1)
    
    Returns:
        t: time points [n_t, 1]
        x1: displacement [n_t, 1]
        x2: velocity [n_t, 1]
        sol: solution object
    """
    mu = params.get('mu', 1.0)
    
    # Time discretization
    t_vals = np.linspace(0, T, n_t)
    h = t_vals[1] - t_vals[0]
    
    # Solution arrays
    x1 = np.zeros(n_t)
    x2 = np.zeros(n_t)
    
    # Initial conditions
    x1[0] = x1_0
    x2[0] = x2_0
    
    # Caputo fractional derivative approximation using Grünwald-Letnikov formula
    def caputo_fractional_derivative(y, h, alpha, n):
        """
        Compute Caputo fractional derivative at step n
        using Grünwald-Letnikov approximation
        """
        if n == 0:
            return 0.0
        
        # Compute binomial coefficients
        coeff = np.zeros(n + 1)
        coeff[0] = 1.0
        for k in range(1, n + 1):
            coeff[k] = coeff[k - 1] * (alpha - k + 1) / k
        
        # Compute fractional derivative
        result = 0.0
        for k in range(n + 1):
            result += coeff[k] * y[n - k]
        
        return result / (h ** alpha)
    
    # Numerical integration using predictor-corrector method
    for n in range(1, n_t):
        # Predictor using fractional derivative
        dx1_alpha = caputo_fractional_derivative(x1, h, alpha, n - 1)
        dx2_alpha = caputo_fractional_derivative(x2, h, alpha, n - 1)
        
        x1_pred = x1[n - 1] + h ** alpha * x2[n - 1]
        x2_pred = x2[n - 1] + h ** alpha * (-x1[n - 1] - mu * (x1[n - 1] ** 2 - 1) * x2[n - 1])
        
        # Corrector (simple average)
        x1[n] = (x1[n - 1] + x1_pred) / 2.0 + 0.5 * h ** alpha * x2[n - 1]
        x2[n] = (x2[n - 1] + x2_pred) / 2.0 + 0.5 * h ** alpha * (-x1[n - 1] - mu * (x1[n - 1] ** 2 - 1) * x2[n - 1])
    
    # Convert to torch tensors
    t = torch.tensor(t_vals, dtype=torch.float32).reshape(-1, 1)
    x1_tensor = torch.tensor(x1, dtype=torch.float32).reshape(-1, 1)
    x2_tensor = torch.tensor(x2, dtype=torch.float32).reshape(-1, 1)
    
    # Create a solution-like object
    class Solution:
        def __init__(self, t, x1, x2):
            self.t = t
            self.y = np.array([x1, x2])
    
    sol = Solution(t_vals, x1, x2)
    
    return t, x1_tensor, x2_tensor, sol


if __name__ == '__main__':
    params = {'mu': 1.0}
    t, x1, x2, sol = generate_fractional_vanderpol_dataset(
        params, T=50.0, x1_0=2.0, x2_0=0.0, n_t=500, alpha=0.9
    )
    print(f"Time shape: {t.shape}")
    print(f"x1 shape: {x1.shape}")
    print(f"x2 shape: {x2.shape}")
    print(f"First few x1 values:\n{x1[:5]}")
    print(f"First few x2 values:\n{x2[:5]}")
