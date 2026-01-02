import torch
import numpy as np
from scipy.integrate import solve_ivp


def generate_memory_diffusion_dataset(
    params,
    nx=50,
    nt=200,
    noise=0.0
):
    """
    Ground truth:
        u_t = D u_xx + w
        w_t = -α w + α u
    """

    D = params["D"]
    alpha = params["alpha"]
    T = params["T"]

    x = np.linspace(0, 1, nx)
    t = np.linspace(0, T, nt)

    dx = x[1] - x[0]

    def laplacian(u):
        u_xx = np.zeros_like(u)
        u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        return u_xx

    def rhs(t, y):
        u = y[:nx]
        w = y[nx:]
        du = D * laplacian(u) + w
        dw = -alpha * w + alpha * u
        return np.concatenate([du, dw])

    # Initial condition
    u0 = np.sin(np.pi * x)
    w0 = np.zeros_like(u0)
    y0 = np.concatenate([u0, w0])

    sol = solve_ivp(rhs, [0, T], y0, t_eval=t)

    U = sol.y[:nx, :].T  # (nt, nx)

    if noise > 0:
        U += noise * np.random.randn(*U.shape)

    W = sol.y[nx:, :].T  # (nt, nx)
    W_flat = W.flatten()[:, None]
    # Prepare training pairs (x,t) → u
    X, T_grid = np.meshgrid(x, t)
    X_flat = np.stack([X.flatten(), T_grid.flatten()], axis=1)
    U_flat = U.flatten()[:, None]

    return (
        torch.tensor(X_flat, dtype=torch.float32),
        torch.tensor(U_flat, dtype=torch.float32),
        torch.tensor(W_flat, dtype=torch.float32),
        sol
    )


def generate_pure_diffusion_dataset(params, nx=50, nt=200):
    """
    生成不含積分項的數據 (純擴散): u_t = D u_xx
    用於預訓練 model，讓它先學會基礎擴散行為。
    """
    D = params["D"]
    T = params["T"]
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, T, nt)
    dx = x[1] - x[0]

    def laplacian(u):
        u_xx = np.zeros_like(u)
        u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        return u_xx

    def rhs(t, u):
        return D * laplacian(u)

    u0 = np.sin(np.pi * x)
    sol = solve_ivp(rhs, [0, T], u0, t_eval=t)
    U = sol.y.T # (nt, nx)

    X, T_grid = np.meshgrid(x, t)
    X_flat = np.stack([X.flatten(), T_grid.flatten()], axis=1)
    return torch.tensor(X_flat, dtype=torch.float32), torch.tensor(U.flatten()[:, None], dtype=torch.float32)