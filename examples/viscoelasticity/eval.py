import os
import torch
import ml_collections
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import numpy as np
import re
import shutil

from examples.viscoelasticity.models import MemoryDiffusionPINN, Corrector
from examples.viscoelasticity.utils import generate_memory_diffusion_dataset
from dapinns.samplers import UniformSampler, RandomSampler


# ============================================================
# Dataset
# ============================================================
def _get_test_data(config: ml_collections.ConfigDict):
    params = config.system_memory.system_params

    # 注意：generate_memory_diffusion_dataset 現在回傳 4 個值
    X, y, w_true_val, sol = generate_memory_diffusion_dataset(
        params,
        nx=params["nx"],
        nt=params["nt"],
        noise=params.get("noise", 0.0),
    )

    sampler = UniformSampler(sample_size=X.shape[0])
    X_test, y_test = sampler.generate_data(X, y)

    return (
        X_test.to(config.device),
        y_test.to(config.device),
        sol
    )


# ============================================================
# Pretrained PINN
# ============================================================
def evaluate_pretrained_pinns(config, workdir):
    sns.set_style("white")

    X_test, y_test, sol = _get_test_data(config)
    nx = config.system_memory.system_params["nx"]

    model = MemoryDiffusionPINN(config).to(config.device)

    pretrained_dir = os.path.join(
        workdir,
        config.saving.save_dir,
        config.saving.pretrain_path,
    )

    def extract_epoch(filename):
        m = re.search(r"epoch_(\d+)", filename)
        return int(m.group(1)) if m else -1

    ckpts = sorted(
        glob.glob(os.path.join(pretrained_dir, "checkpoint*.pt")),
        key=extract_epoch
    )

    if not ckpts:
        print("No pretrained checkpoints found. Skipping.")
        return

    best_l2 = float("inf")
    best_pred = None
    best_ckpt = None

    u_true = sol.y[:nx, :].T.flatten()

    for ckpt in ckpts:
        model.load_pretrained_model(ckpt)
        model.eval()

        with torch.no_grad():
            u_pred = model(X_test).cpu().numpy().ravel()

        l2 = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
        print(f"Pretrain Checkpoint: {os.path.basename(ckpt)} | L2 = {l2:.6e}")

        if l2 < best_l2:
            best_l2 = l2
            best_pred = u_pred
            best_ckpt = ckpt

    # 保存最佳模型
    shutil.copy(best_ckpt, os.path.join(pretrained_dir, "pretrained_model.pt"))

    plt.figure(figsize=(7, 4))
    plt.plot(u_true, label="Reference (Memory)")
    plt.plot(best_pred, "--", label="Pretrained PINN (Pure Diffusion)")
    plt.legend()
    plt.title("Pretrained PINN vs Memory Ground Truth")
    plt.savefig(os.path.join(pretrained_dir, "best_pretrain.png"), dpi=300)
    plt.close()
    print("-" * 80)


# ============================================================
# Finetuned PINN
# ============================================================
def evaluate_finetuned_pinns(config, workdir):
    sns.set_style("white")

    X_test, y_test, sol = _get_test_data(config)
    nx = config.system_memory.system_params["nx"]

    model = MemoryDiffusionPINN(config).to(config.device)

    # 修正路徑問題：區分文件路徑與保存路徑
    finetuned_ckpt_path = os.path.join(
        workdir, config.saving.save_dir,
        config.saving.finetune_path,
        config.finetuned_model_name,
    )
    save_img_dir = os.path.dirname(finetuned_ckpt_path)
    os.makedirs(save_img_dir, exist_ok=True)

    model.load_finetuned_model(finetuned_ckpt_path)
    model.eval()

    with torch.no_grad():
        u_pred = model(X_test).cpu().numpy().ravel()

    u_true = sol.y[:nx, :].T.flatten()

    plt.figure(figsize=(7, 4))
    plt.plot(u_true, label="Reference")
    plt.plot(u_pred, "--", label="DAPINN Prediction")
    plt.legend()
    plt.title("Finetuned DAPINN Results")
    plt.savefig(os.path.join(save_img_dir, "finetune_results.png"), dpi=300)
    plt.close()
    print("-" * 80)


# ============================================================
# Corrector
# ============================================================
def evaluate_corrector(config, workdir):
    sns.set_style("white")

    X_test, _, sol = _get_test_data(config)
    nx = config.system_memory.system_params["nx"]

    # 初始化模型
    model = MemoryDiffusionPINN(config).to(config.device)
    corrector = Corrector(config).to(config.device)

    # 修正路徑問題
    corrector_ckpt_path = os.path.join(
        workdir, config.saving.save_dir,
        config.saving.corrector_path,
        config.corrector_model_name,
    )
    save_img_dir = os.path.dirname(corrector_ckpt_path)
    os.makedirs(save_img_dir, exist_ok=True)

    # 載入權重
    checkpoint = torch.load(corrector_ckpt_path, map_location=config.device, weights_only=True)
    corrector.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    corrector.eval()

    # 即時計算 Corrector 需要的輸入 (u, u_t, u_x, u_xx)
    X_test.requires_grad = True
    u = model(X_test)
    
    # 這裡我們需要計算導數，所以不能用 no_grad
    u_t = torch.autograd.grad(u, X_test, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1:2]
    u_x = torch.autograd.grad(u, X_test, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0:1]
    u_xx = torch.autograd.grad(u_x, X_test, grad_outputs=torch.ones_like(u_x))[0][:, 0:1]

    with torch.no_grad():
        # 注意這裡輸入維度必須與 config 一致 (u, u_t, u_xx) 或 (u, u_t, u_x, u_xx)
        # 根據你前面的代碼，應該是 4 個維度
        corr_input = torch.cat([u, u_t, u_x, u_xx], dim=1)
        s_pred = corrector(corr_input).cpu().numpy().ravel()

    # Ground truth memory variable w
    w_true = sol.y[nx:, :].T.flatten()

    plt.figure(figsize=(7, 4))
    plt.plot(w_true, label="True Memory Term (w)")
    plt.plot(s_pred, "--", label="Predicted Correction (s_psi)")
    plt.legend()
    plt.title("Comparison: Learned Correction vs True Memory")
    plt.savefig(os.path.join(save_img_dir, "corrector_comparison.png"), dpi=300)
    plt.close()
    print(f"Corrector plot saved to {save_img_dir}")


# ============================================================
def evaluate(config: ml_collections.ConfigDict, workdir: str):
    print("Starting evaluation...")
    evaluate_pretrained_pinns(config, workdir)
    evaluate_finetuned_pinns(config, workdir)
    evaluate_corrector(config, workdir)