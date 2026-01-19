import os
import torch
import ml_collections
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import glob
import numpy as np
import re
import shutil

from examples.viscoelasticity.models import MemoryDiffusionPINN, Corrector
from examples.viscoelasticity.utils import generate_elastic_dataset, generate_viscoelastic_dataset

# ===============plot=============================================
# Helper: 3D Plotting
# ============================================================
def plot_3d_comparison(X_flat, Y_true, Y_pred, title, save_path, z_label="u(x,t)", nx=100, nt=200):
    """
    繪製 3D 對照圖：左邊為真值，中間為模型預測，右邊為誤差分布
    """
    # 重新整理數據形狀為 (nt, nx)
    # X_flat 的形狀是 (nt*nx, 2)，包含 (x, t)
    x = X_flat[:, 0].reshape(nt, nx)
    t = X_flat[:, 1].reshape(nt, nx)
    z_true = Y_true.reshape(nt, nx)
    z_pred = Y_pred.reshape(nt, nx)
    z_error = np.abs(z_true - z_pred)

    fig = plt.figure(figsize=(20, 6))
    
    # 1. 真值曲面
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(x, t, z_true, cmap=cm.viridis, antialiased=False, alpha=0.8)
    ax1.set_title(f"Reference ({title})")
    ax1.set_xlabel('Space (x)')
    ax1.set_ylabel('Time (t)')
    ax1.set_zlabel(z_label)

    # 2. 預測曲面
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(x, t, z_pred, cmap=cm.magma, antialiased=False, alpha=0.8)
    ax2.set_title(f"DAPINN Prediction")
    ax2.set_xlabel('Space (x)')
    ax2.set_ylabel('Time (t)')
    ax2.set_zlabel(z_label)

    # 3. 絕對誤差曲面
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(x, t, z_error, cmap=cm.coolwarm, antialiased=False)
    ax3.set_title("Absolute Error")
    ax3.set_xlabel('Space (x)')
    ax3.set_ylabel('Time (t)')
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ============================================================
# Dataset Helpers
# ============================================================
def _get_viscoelastic_test_data(config: ml_collections.ConfigDict):
    params = config.system_memory.system_params
    X, y, w_true, sol = generate_viscoelastic_dataset(
        params, nx=params["nx"], nt=params["nt"], noise=0.0
    )
    return X, y, w_true, sol

def _get_elastic_test_data(config: ml_collections.ConfigDict):
    params = config.system_memory.system_params
    X, y = generate_elastic_dataset(
        params, nx=params["nx"], nt=params["nt"]
    )
    return X, y

# ============================================================
# Update Evaluations
# ============================================================
def evaluate_pretrained_pinns(config, workdir):
    params = config.system_memory.system_params
    nx, nt = params["nx"], params["nt"]
    
    # Pretrain 目標：學習錯誤的純物理 (Elastic)
    X_test, y_pure = _get_elastic_test_data(config)
    model = MemoryDiffusionPINN(config).to(config.device)
    pretrained_dir = os.path.join(workdir, config.saving.save_dir, config.saving.pretrain_path)

    ckpts = sorted(glob.glob(os.path.join(pretrained_dir, "checkpoint*.pt")), 
                   key=lambda x: int(re.search(r"epoch_(\d+)", x).group(1)) if re.search(r"epoch_(\d+)", x) else -1)

    if not ckpts: return
    
    # 載入最後一個或最佳的權重
    model.load_pretrained_model(ckpts[-1]) 
    model.eval()
    with torch.no_grad():
        u_pred = model(X_test.to(config.device)).cpu().numpy().ravel()

    # 繪製 3D 圖：比較 PINN 輸出的 $u$ 與純彈性解的 $u$
    plot_3d_comparison(
        X_test.numpy(), y_pure.numpy().ravel(), u_pred, 
        "Elastic/Pure Diffusion", 
        os.path.join(pretrained_dir, "pretrain_3d_comparison.png"),
        nx=nx, nt=nt
    )

def evaluate_finetuned_pinns(config, workdir):
    params = config.system_memory.system_params
    nx, nt = params["nx"], params["nt"]
    
    # Finetune 目標：學習真實粘彈性物理 (Viscoelastic)
    X_test, y_real, _, _ = _get_viscoelastic_test_data(config)
    model = MemoryDiffusionPINN(config).to(config.device)
    
    finetuned_ckpt_path = os.path.join(workdir, config.saving.save_dir, config.saving.finetune_path, config.finetuned_model_name)
    save_img_dir = os.path.dirname(finetuned_ckpt_path)

    model.load_finetuned_model(finetuned_ckpt_path)
    model.eval()
    with torch.no_grad():
        u_pred = model(X_test.to(config.device)).cpu().numpy().ravel()

    plot_3d_comparison(
        X_test.numpy(), y_real.numpy().ravel(), u_pred, 
        "Real Viscoelastic System", 
        os.path.join(save_img_dir, "finetune_3d_comparison.png"),
        nx=nx, nt=nt
    )

def evaluate_corrector(config, workdir):
    params = config.system_memory.system_params
    nx, nt = params["nx"], params["nt"]
    
    X_test, _, w_real, _ = _get_viscoelastic_test_data(config)
    model = MemoryDiffusionPINN(config).to(config.device)
    corrector = Corrector(config).to(config.device)

    finetuned_ckpt_path = os.path.join(workdir, config.saving.save_dir, config.saving.finetune_path, config.finetuned_model_name)
    corrector_ckpt_path = os.path.join(workdir, config.saving.save_dir, config.saving.corrector_path, config.corrector_model_name)
    
    model.load_finetuned_model(finetuned_ckpt_path)
    checkpoint_c = torch.load(corrector_ckpt_path, map_location=config.device, weights_only=True)
    corrector.load_state_dict(checkpoint_c["model_state_dict"])
    
    model.eval(); corrector.eval()
    
    X_test_dev = X_test.to(config.device); X_test_dev.requires_grad = True
    u = model(X_test_dev)
    grads = torch.autograd.grad(u, X_test_dev, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, X_test_dev, grad_outputs=torch.ones_like(u_x))[0][:, 0:1]

    with torch.no_grad():
        corr_input = torch.cat([u, u_t, u_x, u_xx], dim=1)
        s_pred = corrector(corr_input).cpu().numpy().ravel()

    # 繪製 3D 圖：比較 Corrector 輸出的 $s_\psi$ 與真正的記憶項 $w$
    plot_3d_comparison(
        X_test.numpy(),
        -w_real.numpy().ravel(), s_pred, 
        "Memory Term w", 
        os.path.join(os.path.dirname(corrector_ckpt_path), "corrector_3d_comparison.png"),
        z_label="Correction s_psi", nx=nx, nt=nt
    )

def evaluate(config, workdir):
    print("Starting 3D visualization evaluation...")
    evaluate_pretrained_pinns(config, workdir)
    evaluate_finetuned_pinns(config, workdir)
    evaluate_corrector(config, workdir)