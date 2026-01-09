import os
import torch
import ml_collections
import seaborn as sns
from examples.fractional_van_der_pol.models import FractionalVanderPol, Corrector
from examples.fractional_van_der_pol.utils import generate_fractional_vanderpol_dataset
from dapinns.samplers import UniformSampler, RandomSampler
import matplotlib.pyplot as plt
import glob
import numpy as np
import re
import shutil


def _get_fvdp_test(config: ml_collections.ConfigDict, workdir: str):
    params = config.system_fvdp.system_params
    T = params['T']
    x1_0 = params['x1_0']
    x2_0 = params['x2_0']
    n_t = params['n_t']
    alpha = params.get('alpha', 0.9)

    # Generate dataset
    t, x1, x2, sol = generate_fractional_vanderpol_dataset(
        params, T=T, x1_0=x1_0, x2_0=x2_0, n_t=n_t, alpha=alpha
    )
    
    sampler = UniformSampler(sample_size=n_t)
    t_test, x1_test = sampler.generate_data(t, x1)
    _, x2_test = sampler.generate_data(t, x2)
    
    t_test = t_test.to(config.device)
    x1_test = x1_test.to(config.device)
    x2_test = x2_test.to(config.device)

    return t_test, x1_test, x2_test, sol


def evaluate_pretrained_pinns(config: ml_collections.ConfigDict, workdir: str):
    sns.set_style("white")

    t_test, x1_test, x2_test, sol = _get_fvdp_test(config, workdir)
    t = sol.t
    x1_ground_truth = sol.y[0]
    x2_ground_truth = sol.y[1]

    model = FractionalVanderPol(config).to(config.device)
    
    pretrained_dir = os.path.join(workdir, config.saving.save_dir, config.saving.pretrain_path)

    def extract_epoch(filename):
        match = re.search(r"epoch_(\d+)", filename)
        return int(match.group(1)) if match else -1

    checkpoint_files = glob.glob(os.path.join(pretrained_dir, "checkpoint*.pt"))
    checkpoint_files = sorted(checkpoint_files, key=extract_epoch)

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {pretrained_dir}")

    best_l2_error = float('inf')
    best_checkpoint = None
    best_prediction_x1 = None
    best_prediction_x2 = None

    for ckpt_file in checkpoint_files:
        model.load_pretrained_model(ckpt_file)
        model.eval()

        with torch.no_grad():
            pred = model(t_test).cpu().detach().numpy()
            pred_x1 = pred[:, 0]
            pred_x2 = pred[:, 1]

        l2_error_x1 = np.linalg.norm(pred_x1 - x1_ground_truth, 2) / np.linalg.norm(x1_ground_truth, 2)
        l2_error_x2 = np.linalg.norm(pred_x2 - x2_ground_truth, 2) / np.linalg.norm(x2_ground_truth, 2)
        l2_error = l2_error_x1 + l2_error_x2

        print(f"Checkpoint {os.path.basename(ckpt_file)}: L2 error (x1) = {l2_error_x1:.6f}, L2 error (x2) = {l2_error_x2:.6f}")

        if l2_error < best_l2_error:
            best_l2_error = l2_error
            best_checkpoint = ckpt_file
            best_prediction_x1 = pred_x1
            best_prediction_x2 = pred_x2

    print(f"\nBest checkpoint: {os.path.basename(best_checkpoint)}")
    print(f"Best L2 error: {best_l2_error:.6f}")
    
    # Plotting
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t, x1_ground_truth, 'b-', label='Ground Truth', linewidth=2)
    plt.plot(t, best_prediction_x1, 'r--', label='PINN Pretrained', linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('$x_1(t)$')
    plt.title('Fractional Van der Pol - $x_1$ (Displacement)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(t, x2_ground_truth, 'b-', label='Ground Truth', linewidth=2)
    plt.plot(t, best_prediction_x2, 'r--', label='PINN Pretrained', linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('$x_2(t)$')
    plt.title('Fractional Van der Pol - $x_2$ (Velocity)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    fig_dir = os.path.join(workdir, config.saving.save_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "pretrained_predictions.png"), dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.join(fig_dir, 'pretrained_predictions.png')}")
    plt.close()


def evaluate_finetuned_pinns(config: ml_collections.ConfigDict, workdir: str):
    sns.set_style("white")

    t_test, x1_test, x2_test, sol = _get_fvdp_test(config, workdir)
    t = sol.t
    x1_ground_truth = sol.y[0]
    x2_ground_truth = sol.y[1]

    model = FractionalVanderPol(config).to(config.device)
    
    finetune_dir = os.path.join(workdir, config.saving.save_dir, config.saving.finetune_path)

    def extract_epoch(filename):
        match = re.search(r"epoch_(\d+)", filename)
        return int(match.group(1)) if match else -1

    checkpoint_files = glob.glob(os.path.join(finetune_dir, "checkpoint*.pt"))
    checkpoint_files = sorted(checkpoint_files, key=extract_epoch)

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {finetune_dir}")

    best_l2_error = float('inf')
    best_checkpoint = None
    best_prediction_x1 = None
    best_prediction_x2 = None

    for ckpt_file in checkpoint_files:
        model.load_finetuned_model(ckpt_file)
        model.eval()

        with torch.no_grad():
            pred = model(t_test).cpu().detach().numpy()
            pred_x1 = pred[:, 0]
            pred_x2 = pred[:, 1]

        l2_error_x1 = np.linalg.norm(pred_x1 - x1_ground_truth, 2) / np.linalg.norm(x1_ground_truth, 2)
        l2_error_x2 = np.linalg.norm(pred_x2 - x2_ground_truth, 2) / np.linalg.norm(x2_ground_truth, 2)
        l2_error = l2_error_x1 + l2_error_x2

        if l2_error < best_l2_error:
            best_l2_error = l2_error
            best_checkpoint = ckpt_file
            best_prediction_x1 = pred_x1
            best_prediction_x2 = pred_x2

    print(f"Best checkpoint: {os.path.basename(best_checkpoint)}")
    print(f"Best L2 error: {best_l2_error:.6f}")
    
    # Plotting
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t, x1_ground_truth, 'b-', label='Ground Truth', linewidth=2)
    plt.plot(t, best_prediction_x1, 'g--', label='PINN Finetuned', linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('$x_1(t)$')
    plt.title('Fractional Van der Pol - $x_1$ (Displacement)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(t, x2_ground_truth, 'b-', label='Ground Truth', linewidth=2)
    plt.plot(t, best_prediction_x2, 'g--', label='PINN Finetuned', linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('$x_2(t)$')
    plt.title('Fractional Van der Pol - $x_2$ (Velocity)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    fig_dir = os.path.join(workdir, config.saving.save_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "finetuned_predictions.png"), dpi=300, bbox_inches='tight')
    print(f"Figure saved to {os.path.join(fig_dir, 'finetuned_predictions.png')}")
    plt.close()


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    if config.is_pretrained:
        evaluate_pretrained_pinns(config, workdir)
    else:
        evaluate_finetuned_pinns(config, workdir)
