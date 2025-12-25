import os
import torch
import ml_collections
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
import shutil
import re

from examples.pedagogical_example.models import Pedagogical, Corrector
from examples.pedagogical_example.utils import generate_reaction_ode_dataset
from dapinns.samplers import RandomSampler, UniformSampler

def evaluate_pretrained_pinns(config: ml_collections.ConfigDict, workdir: str):
    """評估預訓練階段的模型，並選出 L2 Error 最小的 Checkpoint 作為最終預訓練模型"""
    sns.set_style("whitegrid")
    device = config.device
    params = config.system_pedagogical.system_params
    T, u0, n_t = params['T'], params['u0'], params['n_t']
    
    # 1. 準備測試資料 (不完整物理的參考解)
    _, _, _, sol = generate_reaction_ode_dataset(params, T=T, u0=u0, n_t=n_t)
    t_test = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1).to(device)
    u_true = sol.y[0]

    model = Pedagogical(config).to(device)
    pretrained_dir = os.path.join(workdir, config.saving.save_dir, "pretrained")

    # 2. 自動搜尋所有的 checkpoint 檔案
    checkpoint_files = glob.glob(os.path.join(pretrained_dir, "checkpoint_*.pt"))
    if not checkpoint_files:
        print(f"[Warn] No pretrained checkpoints found in {pretrained_dir}")
        return

    best_l2_error = float('inf')
    best_ckpt = None
    best_pred = None

    # 3. 遍歷所有 checkpoint 找出最準的一個
    for ckpt in checkpoint_files:
        try:
            checkpoint = torch.load(ckpt, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            with torch.no_grad():
                pred = model(t_test).cpu().numpy().ravel()
            
            error = np.linalg.norm(pred - u_true) / np.linalg.norm(u_true)
            if error < best_l2_error:
                best_l2_error = error
                best_ckpt = ckpt
                best_pred = pred
        except: continue

    print(f"Best Pretrained Checkpoint: {os.path.basename(best_ckpt)} (L2: {best_l2_error:.4e})")

    # 4. 複製最佳模型為固定路徑，方便 Fine-tuning 加載
    shutil.copy(best_ckpt, os.path.join(pretrained_dir, "best_model.pt"))

    # 5. 繪圖
    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, u_true, 'k-', label="Incomplete Physics Truth")
    plt.plot(sol.t, best_pred, 'r--', label="Pretrained Prediction")
    plt.title(f"Best Pretrained Model (L2 Error: {best_l2_error:.2e})")
    plt.legend()
    plt.savefig(os.path.join(pretrained_dir, "best_pretrain.png"))
    plt.close()

def evaluate_finetuned_pinns(config: ml_collections.ConfigDict, workdir: str):
    """評估微調後的模型，並畫出訓練時的觀測點 (Measurements)"""
    sns.set_style("whitegrid")
    device = config.device
    params = config.system_pedagogical.system_params
    T, u0, n_t = params['T'], params['u0'], params['n_t']
    
    # 1. 載入真值與測試集
    _, _, _, sol = generate_reaction_ode_dataset(params, T=T, u0=u0, n_t=n_t)
    t_test = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1).to(device)
    u_true = sol.y[0]

    # 2. 獲取訓練時的隨機觀測點 (使用與訓練相同的種子)
    sampler = RandomSampler(config, sample_size=config.sample_size)
    # 我們需要重現訓練時抽出的點位索引
    t_full = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1)
    u_full = torch.tensor(sol.y[0], dtype=torch.float32).reshape(-1, 1)
    _, _, indices = sampler.generate_data(t_full, u_full, return_indices=True)

    # 3. 加載微調模型
    model = Pedagogical(config).to(device)
    save_root = os.path.join(workdir, config.saving.save_dir)
    finetune_path = os.path.join(save_root, config.saving.finetune_path, "final_model.pt")
    if not os.path.exists(finetune_path):
        finetune_path = os.path.join(save_root, config.saving.finetune_path, "best_finetuned_model.pt")

    model.load_finetuned_model(finetune_path)
    model.eval()

    with torch.no_grad():
        u_pred = model(t_test).cpu().numpy().ravel()

    l2_error = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)

    # 4. 繪圖 (包含真值、預測、觀測點、配點)
    plt.figure(figsize=(10, 6))
    plt.plot(sol.t, u_true, 'k-', linewidth=2, label="Ground Truth")
    plt.plot(sol.t, u_pred, 'r--', linewidth=2, label="DAPINN Prediction")
    
    # 畫出訓練用的觀測點 (Measurements)
    plt.scatter(sol.t[indices], u_true[indices], color='green', s=30, label="Measurements", zorder=3)
    
    # 畫出物理配點 (Collocation Points) - 放在最底端
    t_col = model.t_col.detach().cpu().numpy().ravel()
    plt.scatter(t_col, np.full_like(t_col, plt.gca().get_ylim()[0]), 
                color='gray', marker='|', s=10, alpha=0.3, label="Collocation Points")

    plt.title(f"Finetune Result (L2 Error: {l2_error:.2e})")
    plt.legend()
    plt.savefig(os.path.join(save_root, "prediction_comparison.png"), dpi=300)
    plt.close()

def evaluate_corrector(config: ml_collections.ConfigDict, workdir: str):
    """分析校正器學到的 Missing Term 是否符合 lambda * u * (1-u)"""
    device = config.device
    params = config.system_pedagogical.system_params
    
    # 1. 加載模型
    model = Pedagogical(config).to(device)
    corrector = Corrector(config).to(device)
    
    save_root = os.path.join(workdir, config.saving.save_dir)
    model_path = os.path.join(save_root, config.saving.finetune_path, "final_model.pt")
    corr_path = os.path.join(save_root, config.saving.corrector_path, "final_corrector.pt")
    
    try:
        model.load_finetuned_model(model_path)
        corrector.load_corrector_model(corr_path)
        model.eval()
        corrector.eval()
    except: return

    # 2. 獲取真值
    _, _, _, sol = generate_reaction_ode_dataset(params, T=params['T'], u0=params['u0'], n_t=params['n_t'])
    t_test = torch.tensor(sol.t, dtype=torch.float32).reshape(-1, 1).to(device)
    t_test.requires_grad = True
    
    # 3. 計算 Corrector 的預測輸出
    u_model = model(t_test)
    du_model = torch.autograd.grad(u_model, t_test, torch.ones_like(u_model), create_graph=False)[0]
    inputs = torch.cat([u_model, du_model], dim=1)
    s_pred = corrector(inputs).cpu().detach().numpy().ravel()

    # 4. 計算真實的 Missing Term: lambda * u * (1-u)
    lam = params['lambda']
    u_true = sol.y[0]
    s_true = lam * u_true * (1 - u_true)

    # 5. 繪圖
    plt.figure(figsize=(8, 5))
    plt.plot(sol.t, s_true, 'k-', alpha=0.6, label=r"True Missing Term $\lambda u(1-u)$")
    plt.plot(sol.t, s_pred, 'b--', linewidth=2, label=r"Corrector Output $s_\psi$")
    plt.title("Corrector: Missing Physics Discovery")
    plt.xlabel("Time (t)")
    plt.ylabel("Correction Value")
    plt.legend()
    plt.savefig(os.path.join(save_root, "corrector_analysis.png"), dpi=300)
    plt.close()

def evaluate(config: ml_collections.ConfigDict, workdir: str):
    """總進入點：執行預訓練、微調與校正器的全面評估"""
    print("\n" + "="*30 + " Starting Evaluation " + "="*30)
    evaluate_pretrained_pinns(config, workdir)
    evaluate_finetuned_pinns(config, workdir)
    if config.use_corrector:
        evaluate_corrector(config, workdir)
    print("="*30 + " Evaluation Finished " + "="*30 + "\n")