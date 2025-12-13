import os
import torch
import ml_collections
import seaborn as sns
from examples.dho.models import Dho, Corrector
from examples.dho.utils import generate_uho_dataset, generate_dho_dataset
from dapinns.samplers import UniformSampler, RandomSampler
import matplotlib.pyplot as plt
import glob
import numpy as np
import re
import shutil

def _get_uho_test(config: ml_collections.ConfigDict, workdir: str):
    params = config.system_uho.system_params
    T = params['T']
    x0 = params['x0']
    v0 = params['v0']
    n_t = params['n_t']

    # Generate dataset
    x, y, sol = generate_uho_dataset(params, T=T, x0=x0, v0=v0, n_t=n_t) # dho
    sampler = UniformSampler(sample_size=config.system_uho.system_params['n_t'])
    x_test, y_test = sampler.generate_data(x, y)
    x_test = x_test.to(config.device)
    y_test = y_test.to(config.device)

    return x_test, y_test, sol

def _get_dho_test(config: ml_collections.ConfigDict, workdir: str):
    params = config.system_dho.system_params
    T = params['T']
    x0 = params['x0']
    v0 = params['v0']
    n_t = params['n_t']

    # Generate dataset
    x, y, sol = generate_dho_dataset(params, T=T, x0=x0, v0=v0, n_t=n_t) # dho
    sampler = UniformSampler(sample_size=config.system_dho.system_params['n_t'])
    x_test, y_test = sampler.generate_data(x, y)
    x_test = x_test.to(config.device)
    y_test = y_test.to(config.device)

    return x_test, y_test, sol

def evaluate_pretrained_pinns(config: ml_collections.ConfigDict, workdir: str):
    sns.set_style("white")  # No grid

    x_test, y_test, sol = _get_uho_test(config, workdir)
    t = sol.t
    ground_truth = sol.y[0]

    model = Dho(config).to(config.device)
    
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
    best_prediction = None

    for ckpt_file in checkpoint_files:
        model.load_pretrained_model(ckpt_file)
        model.eval()

        with torch.no_grad():
            pred = model(x_test).cpu().detach().numpy().ravel()

        l2_error = np.linalg.norm(pred - ground_truth, 2) / np.linalg.norm(ground_truth, 2)

        print(f"Checkpoint {os.path.basename(ckpt_file)}: L2 error = {l2_error:.6f}")

        if l2_error < best_l2_error:
            best_l2_error = l2_error
            best_checkpoint = ckpt_file
            best_prediction = pred

    print("-" * 80)
    print(f"Best checkpoint: {os.path.basename(best_checkpoint)} with L2 error = {best_l2_error:.6f}")

    # Store the best checkpoint
    best_model_path = os.path.join(pretrained_dir, "pretrained_model.pt")
    shutil.copy(best_checkpoint, best_model_path)
    print(f"Best checkpoint copied and saved as {best_model_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(t, ground_truth, label="Reference")
    plt.plot(t, best_prediction, label="Predictions (Best)", color='orange', linestyle="--")
    plt.xlabel("Time (t)")
    plt.ylabel("Displacement x(t)")
    plt.legend()
    plt.title("Best Pretrained Model on Undamped Harmonic Oscillator")
    best_plot_path = os.path.join(pretrained_dir, "best_pretrain.png")
    plt.savefig(best_plot_path)
    print(f"Figure best_pretrain.png saved to {pretrained_dir}")
    print("-" * 80)

def evaluate_finetuned_pinns(config: ml_collections.ConfigDict, workdir: str):
    sns.set_style("white")  # No grid

    x_test, y_test, sol = _get_dho_test(config, workdir)

    model = Dho(config).to(config.device)
    finetuned_dir = os.path.join(workdir, config.saving.save_dir, config.saving.finetune_path)
    finetuned_model_dir = os.path.join(finetuned_dir, config.finetuned_model_name)

    model.load_finetuned_model(finetuned_model_dir)
    model.eval()

    y_preds = model(x_test).cpu().detach().numpy().ravel()
    t = sol.t
    ground_truth = sol.y[0]

    # Get mearsurements indices
    params = config.system_dho.system_params
    T, x0, v0, n_t = params['T'], params['x0'], params['v0'], params['n_t']
    if "noise" in config.system_dho.system_params:
        noise = config.system_dho.system_params['noise']
        params['noise'] = noise
        print(f"Noise level: {noise}")
    else:
        print("No noise level specified.")
    x, y, _ = generate_dho_dataset(params, T=T, x0=x0, v0=v0, n_t=n_t)
    sampler = RandomSampler(config, sample_size=config.finetune_sample_size)
    x_train, y_train, indices = sampler.generate_data(x, y, return_indices=True)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(t, ground_truth, label="Reference")
    plt.plot(t, y_preds, label="DAPINNs Predictions", color='orange', linestyle="--")
    plt.scatter(t[indices], y_train.cpu().detach().numpy().ravel(), color='green', label="Measurements", s=10)
    plt.xlabel("Time (t)")
    plt.ylabel("Displacement x(t)")
    plt.legend()
    plt.title("DAPINNs: Prediction on Damped Harmonic Oscillator")
    plt.savefig(finetuned_dir + os.sep + "finetune.png", dpi=300)
    print(f"Figure finetune.png saved to {finetuned_dir}")
    print("-"*80)
def evaluate_corrector(config: ml_collections.ConfigDict, workdir: str):
    sns.set_style("white")  # No grid

    x_test, y_test, sol = _get_dho_test(config, workdir)

    # Load the Corrector model
    corrector = Corrector(config).to(config.device)
    corrector_dir = os.path.join(workdir, config.saving.save_dir, config.saving.corrector_path)
    corrector.load_corrector_model(corrector_dir)

    # Load corrector_inputs if available
    corrector_inputs_dir = os.path.join(corrector_dir,config.corrector_model_name)
    corrector_checkpoint = torch.load(corrector_inputs_dir, map_location=config.device, weights_only=False)
    if 'corrector_inputs' in corrector_checkpoint:
        corrector_inputs = corrector_checkpoint['corrector_inputs']
        print("Corrector inputs loaded successfully.")
    else:
        corrector_inputs = None
        print("No corrector inputs found in the checkpoint.")

    corrector.eval()

    s = corrector(corrector_inputs)
    s_len = s.shape[0]

    t = sol.t
    dx = sol.y[1]
    c = config.system_dho.system_params['c']
    n_t = config.system_dho.system_params['n_t']
    ground_truth = c * dx

    # detach and convert to numpy for plotting
    s = s.cpu().detach().numpy().ravel()[::s_len//n_t] # align

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(t, ground_truth, label="Ground truth misspecified term")
    plt.plot(t, s, label="Predicted Correction", color='orange', linestyle="--")
    plt.xlabel("Time (t)")
    plt.ylabel(r'Correction $s_{\psi}$')
    plt.legend()
    plt.title("Predicted Correction (Damping Term)")
    plt.savefig(corrector_dir + os.sep + "corrector.png")
    print(f"Figure corrector.png saved to {corrector_dir}")

def evaluate(config: ml_collections.ConfigDict, workdir: str):
    
    evaluate_pretrained_pinns(config, workdir)
    evaluate_finetuned_pinns(config, workdir)
    evaluate_corrector(config, workdir)