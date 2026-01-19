import torch
import numpy as np
from scipy.integrate import solve_ivp


def generate_viscoelastic_dataset(
    params,
    nx=50,
    nt=200,
    noise=0.0
):
    """
    正確物理 (Ground Truth): Boltzmann 遺傳模型
    方程形式: 
        u_t = D * u_xx + ∫[0,t] β * exp(-α(t-τ)) * u_xx(x,τ) dτ
    
    轉化為 ODE 系統 (透過輔助變量 w):
        u_t = D * u_xx + w
        w_t = β * u_xx - α * w
    
    這裡:
        D: 瞬時彈性/擴散係數 (G(0))
        β: 記憶項強度 (dot{G}(0))
        α: 鬆弛速率 (Relaxation rate)
    """

    D = params["D"]
    alpha = params["alpha"]
    beta = params["beta"]
    T = params["T"]

    x = np.linspace(0, 1, nx)
    t = np.linspace(0, T, nt)
    dx = x[1] - x[0]

    def laplacian(u):
        u_xx = np.zeros_like(u)
        # 使用二階中央差分，邊界條件假設為 Dirichlet (u=0)
        u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2
        return u_xx

    def rhs(t, y):
        u = y[:nx]
        w = y[nx:]
        
        u_xx = laplacian(u)
        
        du = D * u_xx + w
        dw = beta * u_xx - alpha * w
        
        return np.concatenate([du, dw])

    # 初始條件: u(x,0) = sin(πx), w(x,0) = 0 (假設過去無受力歷史)
    u0 = np.sin(np.pi * x)
    w0 = np.zeros_like(u0)
    y0 = np.concatenate([u0, w0])

    # 求解系統
    sol = solve_ivp(rhs, [0, T], y0, t_eval=t, method='RK45')

    U = sol.y[:nx, :].T  # (nt, nx)
    W = sol.y[nx:, :].T  # (nt, nx)

    # 添加噪聲
    if noise > 0:
        U += noise * np.random.randn(*U.shape)

    # 準備訓練資料格式
    X, T_grid = np.meshgrid(x, t)
    X_flat = np.stack([X.flatten(), T_grid.flatten()], axis=1)
    U_flat = U.flatten()[:, None]
    W_flat = W.flatten()[:, None]

    return (
        torch.tensor(X_flat, dtype=torch.float32),
        torch.tensor(U_flat, dtype=torch.float32),
        torch.tensor(W_flat, dtype=torch.float32),
        sol
    )


def generate_elastic_dataset(params, nx=50, nt=200):
    """
    錯誤假設 (Misspecified): 理想彈性體 / 純擴散模型
    方程: u_t = D * u_xx
    用於預訓練，使模型僅學習基本的擴散行為。
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
    sol = solve_ivp(rhs, [0, T], u0, t_eval=t, method='RK45')
    U = sol.y.T # (nt, nx)

    X, T_grid = np.meshgrid(x, t)
    X_flat = np.stack([X.flatten(), T_grid.flatten()], axis=1)
    
    return torch.tensor(X_flat, dtype=torch.float32), torch.tensor(U.flatten()[:, None], dtype=torch.float32)