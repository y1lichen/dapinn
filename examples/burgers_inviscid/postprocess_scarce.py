import os
import torch
import ml_collections
import seaborn as sns
from examples.burgers_inviscid.models import Burgers, Corrector
from examples.burgers_inviscid.utils import generate_viscous_burgers_dataset
from dapinns.samplers import UniformSampler, RandomSampler, TimeSpaceUniformSampler
import matplotlib.pyplot as plt
import numpy as np
from ml_collections import config_flags
from absl import app
from absl import flags
from kneed import KneeLocator

def _get_viscous_burgers_test(config: ml_collections.ConfigDict, workdir: str):
    # System parameters
    params = config.system_viscous_burgers.system_params
    T = params['T']
    n_x = params['n_x']
    n_t = params['n_t']

    # Data generation
    x, t, u, dudx_test, d2udx2_test = generate_viscous_burgers_dataset(params, T=T, n_x=n_x, n_t=n_t)

    # Sample
    sampler = TimeSpaceUniformSampler(sample_size=n_x * n_t)
    xyt_test, u_test = sampler.generate_data(x, t, u)
    xyt_test, u_test = xyt_test.to(config.device), u_test.to(config.device)

    return xyt_test, u_test, x, t, u, dudx_test, d2udx2_test

# def load_finetuned_models(config: ml_collections.ConfigDict, workdir: str, case: str, target: str):
#     lower_case = case.lower()
#     if target == "Scarce-level-result":
#         SUBFOLDERS = [15, 50, 100, 500, 1000, 5000, 10000, 30000]      
#     elif target == "Noise-level-result":
#         SUBFOLDERS = [0.01, 0.02, 0.03, 0.05, str(0.10), 0.15]
#         case = case + "-scarce-data"

#     models = dict()
#     for sf in SUBFOLDERS:
#         model = Burgers(config).to(config.device)
#         finetuned_model_dir = os.path.join(workdir, target, case, str(sf), lower_case, "finetuned", config.finetuned_model_name)
#         model.load_finetuned_model(finetuned_model_dir)
#         models[sf] = model

#     return models, SUBFOLDERS

# def evaluate_finetune_by_scarce_levels(config, workdir, models, case):

#     # System parameters
#     params = config.system_viscous_burgers.system_params
#     n_x = params['n_x']
#     n_t = params['n_t']

#     postprocess_dir = os.path.join(workdir, "scarce_postprocess")
#     os.makedirs(postprocess_dir, exist_ok=True)
    
#     if case == "BURGERS":
#         case_dir = os.path.join(postprocess_dir, "BURGERS")
#         xyt_test, u_test, x, t, u, dudx_test, d2udx2_test = _get_viscous_burgers_test(config, workdir)
#         ground_truth = u_test.cpu().detach().numpy().reshape(n_x, n_t)
#         ground_truth = ground_truth.T
#         X, T = np.meshgrid(x.squeeze(), t.squeeze())  

#     os.makedirs(case_dir, exist_ok=True)

#     y_plot_dict = dict()
#     y_error_dict = dict()
#     for level, model in models.items():
#         print(f"Evaluating model with scarce level: {level}")

#         # store the y prediction for each level
#         model.eval()
#         y = model(xyt_test).cpu().detach().numpy().reshape(n_x, n_t)
#         y = y.T
#         y_plot_dict[level] = y

#         # Relative L2 error
#         y_error = np.linalg.norm(y - ground_truth) / np.linalg.norm(ground_truth)
#         y_error_dict[level] = y_error
#         print(f"Relative L2 error: {y_error:.4e}")

#     finetuned_dir = os.path.join(case_dir, "finetuned")
#     os.makedirs(finetuned_dir, exist_ok=True)

#     sns.set_style("white")  # No grid

#     title_map = {
#         "BURGERS": "Burgers' Equation"
#     }
#     case_title = title_map.get(case, case)

#     level_ints = sorted([int(lvl) for lvl in y_plot_dict.keys()])
#     min_level, max_level = min(level_ints), max(level_ints)

#     colors = plt.cm.viridis(np.linspace(0, 1, len(level_ints)))
#     min_alpha, max_alpha = 0.3, 1.0

#     time_fractions = [0.25, 0.5, 0.75]  # Fraction of time to plot
#     ground_truth = ground_truth.T # (n_x, n_t)
#     n_t = ground_truth.shape[1]
#     plt.figure(figsize=(15, 5))
#     plt.suptitle(f"DAPINNs Prediction on {case_title}")
#     for i, frac in enumerate(time_fractions, start=1):
#         t_idx = int(frac * n_t)
#         plt.subplot(1, 3, i)
#         plt.plot(x.squeeze(), ground_truth[:, t_idx], label='Ground Truth', color='black')
#         plt.title(f"t = {frac}")
#         plt.xlabel("x")
#         if i == 1:
#             plt.ylabel("u(x, t)")
#         plt.xlim(-1, 1)
#         plt.ylim(-1.1, 1.1)

#         for j, level in enumerate(level_ints):
#             y = y_plot_dict[level].T
#             l2_error = y_error_dict[level]
#             alpha = min_alpha + (level - min_level) / (max_level - min_level + 1e-8) * (max_alpha - min_alpha)

#             plt.plot(x.squeeze(), y[:, t_idx], label=f"{level} pts (L2: {l2_error:.2e})",
#                     color=colors[j], linestyle="--", alpha=alpha)

#         if i==3:
#             plt.legend(loc="upper right", fontsize=8, frameon=True)

#     save_path = os.path.join(finetuned_dir, f"{case.lower()}-finetune-scarce-level.png")
#     plt.savefig(save_path, dpi=300)
#     plt.close()

#     print(f"Figure saved to {save_path}")

#     # --------------------------------
#     # Plot L2 error vs. scarce data level (with equal spacing on x-axis)
#     level_array = np.array(level_ints)
#     error_array = np.array([y_error_dict[level] for level in level_ints])

#     # Use equal spacing for x-axis (0, 1, 2, ...)
#     x_pos = np.arange(len(level_array))  # e.g., [0, 1, 2, 3, 4, 5]

#     knee_locator = KneeLocator(
#         x_pos, error_array,
#         curve='convex',
#         direction='decreasing',
#         S=2.5,
#     )
#     knee_index = knee_locator.knee

#     plt.figure(figsize=(8, 5))
#     plt.plot(x_pos, error_array, marker='o', label="L2 Error")
#     if knee_index is not None:
#         plt.axvline(x=knee_index, color='red', linestyle='--', label=f'Knee Point')
#         plt.scatter([knee_index], [error_array[knee_index]], color='red', zorder=5)

#     plt.xticks(ticks=x_pos, labels=level_array)  # 使用原始 level 當 label
#     plt.xlabel("Number of Measurement Points")
#     plt.ylabel("Relative L2 Error")
#     plt.title(f"L2 Error vs. Scarce Data Level ({case_title})")
#     plt.legend()
#     plt.tight_layout()

#     error_plot_path = os.path.join(finetuned_dir, f"{case.lower()}-l2error-vs-scarce.png")
#     plt.savefig(error_plot_path, dpi=300)
#     plt.close()

#     print(f"L2 error plot saved to {error_plot_path}")
#     print("-" * 80)

# def load_correctors_models(config: ml_collections.ConfigDict, workdir: str, case: str, target: str):
#     lower_case = case.lower()
#     if target == "Scarce-level-result":
#         SUBFOLDERS = [15, 50, 100, 500, 1000, 5000, 10000]   
#     elif target == "Noise-level-result":
#         SUBFOLDERS = [0.01, 0.02, 0.03, 0.05, str(0.10), 0.15]
#         case = case + "-scarce-data"
#         case = case + "-sufficient-data"

#     correctors = dict()
#     correctors_inputs = dict()
#     for sf in SUBFOLDERS:
#         corrector = Corrector(config).to(config.device)
#         corrector_dir = os.path.join(workdir, target, case, str(sf), lower_case, "corrector")
#         corrector.load_corrector_model(corrector_dir)
#         correctors[sf] = corrector

#         corrector_inputs_dir = os.path.join(corrector_dir, config.corrector_model_name)
#         corrector_checkpoint = torch.load(corrector_inputs_dir, weights_only=False)
#         corrector_inputs = corrector_checkpoint['corrector_inputs']
#         correctors_inputs[sf] = corrector_inputs

#     return correctors, correctors_inputs, SUBFOLDERS    

# def evaluate_corrector_by_scarce_levels(config, workdir, models, inputs, case):

#     postprocess_dir = os.path.join(workdir, "scarce_postprocess")
#     os.makedirs(postprocess_dir, exist_ok=True)
    
#     # System parameters
#     params = config.system_viscous_burgers.system_params
#     n_x = params['n_x']
#     n_t = params['n_t']
    
#     if case == "BURGERS":
#         case_dir = os.path.join(postprocess_dir, "BURGERS")
#         xyt_test, u_test, x, t, u, dudx_test, d2udx2_test = _get_viscous_burgers_test(config, workdir)
#         ground_truth = u.cpu().detach().numpy() * dudx_test.cpu().detach().numpy()
#         ground_truth = ground_truth[::10, ::10]  # Downsample for plotting
#         ground_truth = ground_truth.T
#         X, T = np.meshgrid(x.squeeze()[::10], t.squeeze()[::10])  # Create a meshgrid for plotting

#     os.makedirs(case_dir, exist_ok=True)
#     y_plot_dict = dict()
#     y_error_dict = dict()
#     for level, model in models.items():
#         print(f"Evaluating model with scarce level: {level}")

#         input_tensor = inputs[level]
#         # store the y prediction for each level
#         model.eval()
#         y = model(input_tensor).cpu().detach().numpy().reshape(100, 100)
#         y = y.T
#         y_plot_dict[level] = y

#         # Relative L2 error
#         y_error = np.linalg.norm(y - ground_truth) / np.linalg.norm(ground_truth)
#         y_error_dict[level] = y_error
#         print(f"Relative L2 error: {y_error:.4e}")

#     corrector_dir = os.path.join(case_dir, "corrector")
#     os.makedirs(corrector_dir, exist_ok=True)

#     sns.set_style("white")  # No grid

#     title_map = {
#         "BURGERS": "Burgers' Equation"
#     }
#     case_title = title_map.get(case, case)

#     level_ints = sorted([int(lvl) for lvl in y_plot_dict.keys()])
#     min_level, max_level = min(level_ints), max(level_ints)

#     colors = plt.cm.viridis(np.linspace(0, 1, len(level_ints)))
#     min_alpha, max_alpha = 0.3, 1.0

#     time_fractions = [0.25, 0.5, 0.75]  # Fraction of time to plot
#     ground_truth = ground_truth.T # (n_x, n_t)
#     n_t = ground_truth.shape[1]
#     plt.figure(figsize=(15, 5))
#     plt.suptitle(f"DAPINNs Prediction on {case_title}")
#     for i, frac in enumerate(time_fractions, start=1):
#         t_idx = int(frac * n_t)
#         plt.subplot(1, 3, i)
#         plt.plot(x.squeeze(), ground_truth[:, t_idx], label='Ground Truth', color='black')
#         plt.title(f"t = {frac}")
#         plt.xlabel("x")
#         if i == 1:
#             plt.ylabel("u(x, t)")

#         for j, level in enumerate(level_ints):
#             y = y_plot_dict[level].T
#             l2_error = y_error_dict[level]
#             alpha = min_alpha + (level - min_level) / (max_level - min_level + 1e-8) * (max_alpha - min_alpha)

#             plt.plot(x.squeeze(), y[:, t_idx], label=f"{level} pts (L2: {l2_error:.2e})",
#                     color=colors[j], linestyle="--", alpha=alpha)

#         if i==3:
#             plt.legend(loc="upper right", fontsize=8, frameon=True)

#     save_path = os.path.join(corrector_dir, f"{case.lower()}-corrector-scarce-level.png")
#     plt.savefig(save_path, dpi=300)
#     plt.close()

#     print(f"Figure saved to {save_path}")

#     # --------------------------------
#     # Plot L2 error vs. scarce data level (with equal spacing on x-axis)
#     level_array = np.array(level_ints)
#     error_array = np.array([y_error_dict[level] for level in level_ints])

#     # Use equal spacing for x-axis (0, 1, 2, ...)
#     x_pos = np.arange(len(level_array))  # e.g., [0, 1, 2, 3, 4, 5]

#     knee_locator = KneeLocator(
#         x_pos, error_array,
#         curve='convex',
#         direction='decreasing',
#         S=2.5,
#     )
#     knee_index = knee_locator.knee

#     plt.figure(figsize=(8, 5))
#     plt.plot(x_pos, error_array, marker='o', label="L2 Error")
#     if knee_index is not None:
#         plt.axvline(x=knee_index, color='red', linestyle='--', label=f'Knee Point')
#         plt.scatter([knee_index], [error_array[knee_index]], color='red', zorder=5)

#     plt.xticks(ticks=x_pos, labels=level_array)  # 使用原始 level 當 label
#     plt.xlabel("Number of Measurement Points")
#     plt.ylabel("Relative L2 Error")
#     plt.title(f"L2 Error vs. Scarce Data Level ({case_title})")
#     plt.legend()
#     plt.tight_layout()

#     error_plot_path = os.path.join(corrector_dir, f"{case.lower()}-l2error-vs-scarce.png")
#     plt.savefig(error_plot_path, dpi=300)
#     plt.close()

#     print(f"L2 error plot saved to {error_plot_path}")
#     print("-" * 80)

def load_seeds_models(config, workdir: str, case: str, target: str, SUBFOLDERS=None):
    lower_case = case.lower()
    
    if SUBFOLDERS is None:
        SUBFOLDERS = [15, 50, 100, 500, 1000, 5000, 10000, 30000]   

    models = dict()

    for sf in SUBFOLDERS:
        models[sf] = dict()

        scarce_dir = os.path.join(workdir, target, lower_case, str(sf))

        for seed_folder in os.listdir(scarce_dir):
            seed_path = os.path.join(scarce_dir, seed_folder, lower_case)
            model = Burgers(config).to(config.device)
            finetuned_model_dir = os.path.join(seed_path, "finetuned", config.finetuned_model_name)
            model.load_finetuned_model(finetuned_model_dir)

            models[sf][seed_folder] = model

    return models

def load_seeds_correctors(config, workdir: str, case: str, target: str, SUBFOLDERS=None):
    lower_case = case.lower()
    
    if SUBFOLDERS is None:
        SUBFOLDERS = [15, 50, 100, 500, 1000, 5000, 10000, 30000]   

    correctors = dict()
    correctors_inputs = dict()

    for sf in SUBFOLDERS:
        correctors[sf] = dict()
        correctors_inputs[sf] = dict()

        scarce_dir = os.path.join(workdir, target, lower_case, str(sf))

        for seed_folder in os.listdir(scarce_dir):
            seed_path = os.path.join(scarce_dir, seed_folder, lower_case)
            corrector = Corrector(config).to(config.device)
            corrector_dir = os.path.join(seed_path, "corrector")
            corrector.load_corrector_model(corrector_dir)

            correctors[sf][seed_folder] = corrector

            corrector_inputs_dir = os.path.join(corrector_dir, config.corrector_model_name)
            corrector_checkpoint = torch.load(corrector_inputs_dir, weights_only=False)
            corrector_inputs = corrector_checkpoint['corrector_inputs']
            correctors_inputs[sf][seed_folder] = corrector_inputs

    return correctors, correctors_inputs
def evaluate_finetune_by_seeds(config, workdir, models, case):
    postprocess_dir = os.path.join(workdir, "scarce_postprocess")
    os.makedirs(postprocess_dir, exist_ok=True)

    # System parameters
    params = config.system_viscous_burgers.system_params
    n_x = params['n_x']
    n_t = params['n_t']
    if case == "BURGERS":
        case_dir = os.path.join(postprocess_dir, "BURGERS")
        xyt_test, u_test, x, t, u, dudx_test, d2udx2_test = _get_viscous_burgers_test(config, workdir)
        ground_truth = u_test.cpu().detach().numpy().reshape(n_x, n_t)
        ground_truth = ground_truth.T
        X, T = np.meshgrid(x.squeeze(), t.squeeze())  

    os.makedirs(case_dir, exist_ok=True)
    
    y_plot_dict = dict()
    y_std_dict = dict()
    y_error_plot_dict = dict()
    y_error_std_dict = dict()
    check_list = []
    for level in models.keys():
        seeds_models = models[level]

        # store the y prediction for each level
        y_list = []
        y_error_list = []
        for seed in seeds_models.keys():
            model = seeds_models[seed]
            model.eval()
            y = model(xyt_test).cpu().detach().numpy().ravel()  # align
            y_list.append(y)

            y_tx = y.reshape(n_x,n_t).T
            error = np.linalg.norm(y_tx - ground_truth) / np.linalg.norm(ground_truth)
            y_error_list.append(error)
            if np.isnan(error):
                check_list.append((level, seed))

        y_array = np.stack(y_list, axis=0)  # shape: (num_seeds, num_points)
        y_mean = np.mean(y_array, axis=0)
        y_std = np.std(y_array, axis=0)

        # Relative L2 error of mean prediction
        y_mean = y_mean.reshape(n_x, n_t)  # Reshape to match the ground truth shape
        y_mean = y_mean.T  # Transpose to match the ground truth shape

        y_error_array = np.stack(y_error_list, axis=0)
        y_error_mean = np.mean(y_error_array)
        y_error_std = np.std(y_error_array)

        # Store
        y_plot_dict[level] = y_mean
        y_std_dict[level] = y_std

        y_error_plot_dict[level] = y_error_mean
        y_error_std_dict[level] = y_error_std
        print(f"Evaluating DAPINNs with scarce level: {level}, Relative L2 error (mean prediction): {y_error_mean:.4e}")
    print(check_list)

    finetuned_dir = os.path.join(case_dir, "finetuned")
    os.makedirs(finetuned_dir, exist_ok=True)
    sns.set_style("white")  # No grid

    time_fractions = [0.25, 0.50, 0.75]
    for level in models.keys():
        plt.figure(figsize=(9, 3), dpi=300)
        level_dir = os.path.join(finetuned_dir, f"{level}")
        os.makedirs(level_dir, exist_ok=True)
        n_t = ground_truth.shape[0]
        reshape_ground_truth = ground_truth.T # n_x n_t
        reshape_y_mean = y_plot_dict[level].T # n_x n_t
        reshape_y_std = y_std_dict[level].reshape(n_x,n_t) # n_x n_t
        for i, frac in enumerate(time_fractions, start=1):
            t_idx = int(frac * n_t)
            plt.subplot(1, 3, i)
            plt.plot(x.squeeze(), reshape_ground_truth[:, t_idx], label='Ground Truth', color='black')
            plt.plot(x.squeeze(), reshape_y_mean[:, t_idx], label='Prediction', linestyle='--', color='tab:red')
            plt.fill_between(
                x.squeeze(), 
                reshape_y_mean[:, t_idx] - 2*reshape_y_std[:, t_idx],
                reshape_y_mean[:, t_idx] + 2*reshape_y_std[:, t_idx],
                label=f'±2 Std Dev ({level} pts)',
                alpha=0.2, 
                color='red'
            )
            plt.title(f"t = {frac}")
            plt.xlabel("x")
            if i == 1:
                plt.ylabel("u(x, t)")
            if i == 3:
                plt.legend(loc='lower right')
        plt.tight_layout()
        save_path = os.path.join(level_dir, "u_vs_x_slices.png")
        plt.savefig(save_path)
        print(f"Saved {level}-3-slice plot to {save_path}")
    print("-"*80)

    plt.figure(figsize=(9, 3), dpi=300)
    level_1, level_2 = 100, 400
    reshape_y_mean_1 = y_plot_dict[level_1].T # n_x n_t
    reshape_y_mean_2 = y_plot_dict[level_2].T # n_x n_t
    reshape_y_std_1 = y_std_dict[level_1].reshape(n_x,n_t) # n_x n_t
    reshape_y_std_2 = y_std_dict[level_2].reshape(n_x,n_t) # n_x n_t
    for i, frac in enumerate(time_fractions, start=1):
        t_idx = int(frac * n_t)
        plt.subplot(1, 3, i)
        plt.plot(x.squeeze(), reshape_ground_truth[:, t_idx], label='Ground Truth', color='black')
        plt.plot(x.squeeze(), reshape_y_mean_1[:, t_idx], label=f'{level_1} pts Prediction', linestyle='--', color='tab:red')
        plt.plot(x.squeeze(), reshape_y_mean_2[:, t_idx], label=f'{level_2} pts Prediction', linestyle='--', color='tab:blue')
        plt.fill_between(
            x.squeeze(), 
            reshape_y_mean_1[:, t_idx] - 2*reshape_y_std_1[:, t_idx],
            reshape_y_mean_1[:, t_idx] + 2*reshape_y_std_1[:, t_idx],
            label=f'±2 Std Dev ({level} pts)',
            alpha=0.2, 
            color='red'
        )
        plt.fill_between(
            x.squeeze(), 
            reshape_y_mean_2[:, t_idx] - 2*reshape_y_std_2[:, t_idx],
            reshape_y_mean_2[:, t_idx] + 2*reshape_y_std_2[:, t_idx],
            label=f'±2 Std Dev ({level} pts)',
            alpha=0.2, 
            color='blue'
        )
        plt.title(f"t = {frac}")
        plt.xlabel("x")
        if i == 1:
            plt.ylabel("u(x, t)")
        if i == 3:
            plt.legend(loc='lower right')
    plt.tight_layout()
    save_path = os.path.join(finetuned_dir, f"comparison_{level_1}_{level_2}_u_vs_x_slices.png")
    plt.savefig(save_path)
    print(f"Saved comparison-3-slice plot to {save_path}")
    print("-"*80)
    title_map = {
        "DHO": "Damped Harmonic Oscillator",
        "QDHO": "Quadratic Damped Harmonic Oscillator",
        "BURGERS": "Burgers' Equation"
    }
    case_title = title_map.get(case, case)

    level_ints = sorted([int(lvl) for lvl in y_plot_dict.keys()])
    min_level, max_level = min(level_ints), max(level_ints)
    # --------------------------------
    # Plot mean L2 error ± std vs. scarce data level (with equal spacing on x-axis)   
    level_array = np.array(level_ints)
    error_array = np.array([y_error_plot_dict[level] for level in level_ints])
    std_array = np.array([y_error_std_dict[level] for level in level_ints])

    x_pos = np.arange(len(level_array))  # Equal spacing on x-axis

    plt.figure(figsize=(8, 5))
    plt.fill_between(
        x_pos,
        error_array - 2 * std_array,
        error_array + 2 * std_array,
        color='blue',
        alpha=0.15,
        label='±2 Std Dev',
        zorder=0
    )
    plt.plot(x_pos, error_array, marker='o', color='blue', label="L2 Error (mean)", zorder=2)
    plt.xticks(ticks=x_pos, labels=level_array)
    plt.xlabel("Number of Measurement Points")
    plt.ylabel("Relative L2 Error")
    plt.title(f"DAPINNs L2 Error vs. Scarce Data Level ({case_title})")
    plt.legend()
    plt.tight_layout()

    error_plot_path = os.path.join(finetuned_dir, f"{case.lower()}-seeds-l2error-mean-std.png")
    plt.savefig(error_plot_path, dpi=300)
    plt.close()

    print(f"L2 error plot with error bars saved to {error_plot_path}")
    print("-" * 80)

def evaluate_corrector_by_seeds(config, workdir, models, inputs, case):
    postprocess_dir = os.path.join(workdir, "scarce_postprocess")
    os.makedirs(postprocess_dir, exist_ok=True)
    # System parameters
    params = config.system_viscous_burgers.system_params
    n_x = params['n_x']
    n_t = params['n_t']

    if case == "BURGERS":
        case_dir = os.path.join(postprocess_dir, "BURGERS")
        xyt_test, u_test, x, t, u, dudx_test, d2udx2_test = _get_viscous_burgers_test(config, workdir)
        ground_truth = u.cpu().detach().numpy() * dudx_test.cpu().detach().numpy()
        # ground_truth = ground_truth[::10, ::10]  # Downsample for plotting
        ground_truth = ground_truth.T
        X, T = np.meshgrid(x.squeeze()[::10], t.squeeze()[::10])  # Create a meshgrid for plotting

    os.makedirs(case_dir, exist_ok=True)

    y_plot_dict = dict()
    y_std_dict = dict()
    y_error_plot_dict = dict()
    y_error_std_dict = dict()
    check_list = []
    for level in models.keys():
        seeds_models = models[level]
        seeds_inputs_tensor = inputs[level]

        y_list = []
        y_error_list = []
        for seed in seeds_models.keys():
            model = seeds_models[seed]
            input_tensor = seeds_inputs_tensor[seed]
            model.eval()
            y = model(input_tensor)
            y = y.cpu().detach().numpy().ravel()  # align
            y_list.append(y)

            y_tx = y.reshape(n_x, n_t).T
            error = np.linalg.norm(y_tx - ground_truth) / np.linalg.norm(ground_truth)
            y_error_list.append(error)
            if np.isnan(error):
                check_list.append((level, seed))

        y_array = np.stack(y_list, axis=0)  # shape: (num_seeds, num_points)
        y_mean = np.mean(y_array, axis=0)
        y_std = np.std(y_array, axis=0)

        # Relative L2 error of mean prediction
        y_mean = y_mean.reshape(n_x, n_t)
        y_mean = y_mean.T

        y_error_array = np.stack(y_error_list, axis=0)
        y_error_mean = np.mean(y_error_array)
        y_error_std = np.std(y_error_array)

        # Store
        y_plot_dict[level] = y_mean
        y_std_dict[level] = y_std

        y_error_plot_dict[level] = y_error_mean
        y_error_std_dict[level] = y_error_std
        print(f"Evaluating corrector with scarce level: {level}, Relative L2 error (mean prediction): {y_error_mean:.4e}")
    
    for level in y_error_plot_dict.keys():
        print(f"Mean: {y_error_plot_dict[level]}, std: {y_error_std_dict[level]}")

    sns.set_style("white")  # No grid
    corrector_dir = os.path.join(case_dir, "corrector")
    os.makedirs(corrector_dir, exist_ok=True)

    time_fractions = [0.25, 0.50, 0.75]
    for level in models.keys():
        plt.figure(figsize=(9, 3), dpi=300)
        level_dir = os.path.join(corrector_dir, f"{level}")
        os.makedirs(level_dir, exist_ok=True)
        n_t = ground_truth.shape[0]
        reshape_ground_truth = ground_truth.T # n_x n_t
        reshape_y_mean = y_plot_dict[level].T # n_x n_t
        reshape_y_std = y_std_dict[level].reshape(n_x,n_t) # n_x n_t
        for i, frac in enumerate(time_fractions, start=1):
            t_idx = int(frac * n_t)
            plt.subplot(1, 3, i)
            plt.plot(x.squeeze(), reshape_ground_truth[:, t_idx], label='Ground Truth', color='black')
            plt.plot(x.squeeze(), reshape_y_mean[:, t_idx], label='Prediction', linestyle='--', color='tab:red')
            plt.fill_between(
                x.squeeze(), 
                reshape_y_mean[:, t_idx] - 2*reshape_y_std[:, t_idx],
                reshape_y_mean[:, t_idx] + 2*reshape_y_std[:, t_idx],
                label=f'±2 Std Dev ({level} pts)',
                alpha=0.2, 
                color='red'
            )
            plt.title(f"t = {frac}")
            plt.xlabel("x")
            if i == 1:
                plt.ylabel("u(x, t)")
            if i == 3:
                plt.legend(loc='lower right')
        plt.tight_layout()
        save_path = os.path.join(level_dir, "u_vs_x_slices.png")
        plt.savefig(save_path)
        print(f"Saved {level}-3-slice plot to {save_path}")
    print("-"*80)
    
    plt.figure(figsize=(9, 3), dpi=300)
    level_1, level_2 = 400, 1000
    reshape_y_mean_1 = y_plot_dict[level_1].T # n_x n_t
    reshape_y_mean_2 = y_plot_dict[level_2].T # n_x n_t
    reshape_y_std_1 = y_std_dict[level_1].reshape(n_x,n_t) # n_x n_t
    reshape_y_std_2 = y_std_dict[level_2].reshape(n_x,n_t) # n_x n_t
    for i, frac in enumerate(time_fractions, start=1):
        t_idx = int(frac * n_t)
        plt.subplot(1, 3, i)
        plt.plot(x.squeeze(), reshape_ground_truth[:, t_idx], label='Ground Truth', color='black')
        plt.plot(x.squeeze(), reshape_y_mean_1[:, t_idx], label=f'{level_1} pts Prediction', linestyle='--', color='tab:red')
        plt.plot(x.squeeze(), reshape_y_mean_2[:, t_idx], label=f'{level_2} pts Prediction', linestyle='--', color='tab:blue')
        plt.fill_between(
            x.squeeze(), 
            reshape_y_mean_1[:, t_idx] - 2*reshape_y_std_1[:, t_idx],
            reshape_y_mean_1[:, t_idx] + 2*reshape_y_std_1[:, t_idx],
            label=f'±2 Std Dev ({level} pts)',
            alpha=0.2, 
            color='red'
        )
        plt.fill_between(
            x.squeeze(), 
            reshape_y_mean_2[:, t_idx] - 2*reshape_y_std_2[:, t_idx],
            reshape_y_mean_2[:, t_idx] + 2*reshape_y_std_2[:, t_idx],
            label=f'±2 Std Dev ({level} pts)',
            alpha=0.2, 
            color='blue'
        )
        plt.title(f"t = {frac}")
        plt.xlabel("x")
        if i == 1:
            plt.ylabel("u(x, t)")
        if i == 3:
            plt.legend(loc='lower right')
    plt.tight_layout()
    save_path = os.path.join(corrector_dir, f"comparison_{level_1}_{level_2}_u_vs_x_slices.png")
    plt.savefig(save_path)
    print(f"Saved comparison-3-slice plot to {save_path}")
    print("-"*80)
    
    title_map = {
        "DHO": "Damped Harmonic Oscillator",
        "QDHO": "Quadratic Damped Harmonic Oscillator",
        "BURGERS": "Burgers' Equation"
    }
    case_title = title_map.get(case, case)

    level_ints = sorted([int(lvl) for lvl in y_plot_dict.keys()])

    # --------------------------------
    # Plot mean L2 error ± std vs. scarce data level (with equal spacing on x-axis)   
    level_array = np.array(level_ints)
    error_array = np.array([y_error_plot_dict[level] for level in level_ints])
    std_array = np.array([y_error_std_dict[level] for level in level_ints])

    x_pos = np.arange(len(level_array))  # Equal spacing on x-axis
    print(error_array, std_array)

    plt.figure(figsize=(8, 5))
    plt.fill_between(
        x_pos,
        error_array - 2 * std_array,
        error_array + 2 * std_array,
        color='blue',
        alpha=0.15,
        label='±2 Std Dev',
        zorder=0
    )

    plt.plot(x_pos, error_array, marker='o', color='blue', label="L2 Error (mean)", zorder=2)

    plt.xticks(ticks=x_pos, labels=level_array)
    plt.xlabel("Number of Measurement Points")
    plt.ylabel("Relative L2 Error")
    plt.title(f"Corrector L2 Error vs. Scarce Data Level ({case_title})")
    plt.legend()
    plt.tight_layout()

    error_plot_path = os.path.join(corrector_dir, f"{case.lower()}-seeds-l2error-mean-std.png")
    plt.savefig(error_plot_path, dpi=300)
    plt.close()

    print(f"L2 error plot with error bars saved to {error_plot_path}")
    print("-" * 80)


def evaluate_corrector_dapinns_by_seeds(config, workdir, dapinns, models, case):
    postprocess_dir = os.path.join(workdir, "scarce_postprocess")
    os.makedirs(postprocess_dir, exist_ok=True)
    # System parameters
    params = config.system_viscous_burgers.system_params
    n_x = params['n_x']
    n_t = params['n_t']

    if case == "BURGERS":
        case_dir = os.path.join(postprocess_dir, "BURGERS")
        xyt_test, u_test, x, t, u, dudx_test, d2udx2_test = _get_viscous_burgers_test(config, workdir)
        ground_truth = u.cpu().detach().numpy() * dudx_test.cpu().detach().numpy()
        # ground_truth = ground_truth[::10, ::10]  # Downsample for plotting
        ground_truth = ground_truth.T
        X, T = np.meshgrid(x.squeeze()[::10], t.squeeze()[::10])  # Create a meshgrid for plotting
        xyt_test = xyt_test.to(config.device)

    os.makedirs(case_dir, exist_ok=True)

    y_plot_dict = dict()
    y_std_dict = dict()
    y_error_plot_dict = dict()
    y_error_std_dict = dict()
    check_list = []
    for level in models.keys():
        seeds_dapinns = dapinns[level]
        seeds_models = models[level]

        y_list = []
        y_error_list = []
        for seed in seeds_models.keys():
            dapinn = seeds_dapinns[seed]
            dapinn.eval()
            xyt_test.requires_grad_(True)
            u_pred = dapinn(xyt_test)
            ux_ut_pred = torch.autograd.grad(u_pred, xyt_test, grad_outputs=torch.ones_like(u_pred), retain_graph=True, create_graph=True)[0]
            ux_pred = ux_ut_pred[:, [0]]
            ut_pred = ux_ut_pred[:, [1]]
            input_tensor = torch.cat([u_pred, ux_pred, ut_pred], dim=1)

            model = seeds_models[seed]
            model.eval()
            y = model(input_tensor)
            y = y.cpu().detach().numpy().ravel()  # align
            y_list.append(y)

            y_tx = y.reshape(n_x, n_t).T
            error = np.linalg.norm(y_tx - ground_truth) / np.linalg.norm(ground_truth)
            y_error_list.append(error)
            if np.isnan(error):
                check_list.append((level, seed))

        y_array = np.stack(y_list, axis=0)  # shape: (num_seeds, num_points)
        y_mean = np.mean(y_array, axis=0)
        y_std = np.std(y_array, axis=0)

        # Relative L2 error of mean prediction
        y_mean = y_mean.reshape(n_x, n_t)
        y_mean = y_mean.T

        y_error_array = np.stack(y_error_list, axis=0)
        y_error_mean = np.mean(y_error_array)
        y_error_std = np.std(y_error_array)

        # Store
        y_plot_dict[level] = y_mean
        y_std_dict[level] = y_std

        y_error_plot_dict[level] = y_error_mean
        y_error_std_dict[level] = y_error_std
        print(f"Evaluating corrector with scarce level: {level}, Relative L2 error (mean prediction): {y_error_mean:.4e}")
    
    for level in y_error_plot_dict.keys():
        print(f"Mean: {y_error_plot_dict[level]}, std: {y_error_std_dict[level]}")

    sns.set_style("white")  # No grid
    corrector_dir = os.path.join(case_dir, "corrector")
    os.makedirs(corrector_dir, exist_ok=True)

    time_fractions = [0.25, 0.50, 0.75]
    for level in models.keys():
        plt.figure(figsize=(9, 3), dpi=300)
        level_dir = os.path.join(corrector_dir, f"{level}")
        os.makedirs(level_dir, exist_ok=True)
        n_t = ground_truth.shape[0]
        reshape_ground_truth = ground_truth.T # n_x n_t
        reshape_y_mean = y_plot_dict[level].T # n_x n_t
        reshape_y_std = y_std_dict[level].reshape(n_x,n_t) # n_x n_t
        for i, frac in enumerate(time_fractions, start=1):
            t_idx = int(frac * n_t)
            plt.subplot(1, 3, i)
            plt.plot(x.squeeze(), reshape_ground_truth[:, t_idx], label='Ground Truth', color='black')
            plt.plot(x.squeeze(), reshape_y_mean[:, t_idx], label='Prediction', linestyle='--', color='tab:red')
            plt.fill_between(
                x.squeeze(), 
                reshape_y_mean[:, t_idx] - 2*reshape_y_std[:, t_idx],
                reshape_y_mean[:, t_idx] + 2*reshape_y_std[:, t_idx],
                label=f'±2 Std Dev ({level} pts)',
                alpha=0.2, 
                color='red'
            )
            plt.title(f"t = {frac}")
            plt.xlabel("x")
            if i == 1:
                plt.ylabel("u(x, t)")
            if i == 3:
                plt.legend(loc='lower right')
        plt.tight_layout()
        save_path = os.path.join(level_dir, "u_vs_x_slices.png")
        plt.savefig(save_path)
        print(f"Saved {level}-3-slice plot to {save_path}")
    print("-"*80)
    
    plt.figure(figsize=(9, 3), dpi=300)
    level_1, level_2 = 100, 400
    reshape_y_mean_1 = y_plot_dict[level_1].T # n_x n_t
    reshape_y_mean_2 = y_plot_dict[level_2].T # n_x n_t
    reshape_y_std_1 = y_std_dict[level_1].reshape(n_x,n_t) # n_x n_t
    reshape_y_std_2 = y_std_dict[level_2].reshape(n_x,n_t) # n_x n_t
    for i, frac in enumerate(time_fractions, start=1):
        t_idx = int(frac * n_t)
        plt.subplot(1, 3, i)
        plt.plot(x.squeeze(), reshape_ground_truth[:, t_idx], label='Ground Truth', color='black')
        plt.plot(x.squeeze(), reshape_y_mean_1[:, t_idx], label=f'{level_1} pts Prediction', linestyle='--', color='tab:red')
        plt.plot(x.squeeze(), reshape_y_mean_2[:, t_idx], label=f'{level_2} pts Prediction', linestyle='--', color='tab:blue')
        plt.fill_between(
            x.squeeze(), 
            reshape_y_mean_1[:, t_idx] - 2*reshape_y_std_1[:, t_idx],
            reshape_y_mean_1[:, t_idx] + 2*reshape_y_std_1[:, t_idx],
            label=f'±2 Std Dev ({level_1} pts)',
            alpha=0.2, 
            color='red'
        )
        plt.fill_between(
            x.squeeze(), 
            reshape_y_mean_2[:, t_idx] - 2*reshape_y_std_2[:, t_idx],
            reshape_y_mean_2[:, t_idx] + 2*reshape_y_std_2[:, t_idx],
            label=f'±2 Std Dev ({level_2} pts)',
            alpha=0.2, 
            color='blue'
        )
        plt.title(f"t = {frac}")
        plt.xlabel("x")
        if i == 1:
            plt.ylabel("u(x, t)")
        if i == 3:
            plt.legend(loc='lower right')
    plt.tight_layout()
    save_path = os.path.join(corrector_dir, f"comparison_{level_1}_{level_2}_u_vs_x_slices.png")
    plt.savefig(save_path)
    print(f"Saved comparison-3-slice plot to {save_path}")
    print("-"*80)
    
    title_map = {
        "DHO": "Damped Harmonic Oscillator",
        "QDHO": "Quadratic Damped Harmonic Oscillator",
        "BURGERS": "Burgers' Equation"
    }
    case_title = title_map.get(case, case)

    level_ints = sorted([int(lvl) for lvl in y_plot_dict.keys()])

    # --------------------------------
    # Plot mean L2 error ± std vs. scarce data level (with equal spacing on x-axis)   
    level_array = np.array(level_ints)
    error_array = np.array([y_error_plot_dict[level] for level in level_ints])
    std_array = np.array([y_error_std_dict[level] for level in level_ints])

    x_pos = np.arange(len(level_array))  # Equal spacing on x-axis
    print(error_array, std_array)

    plt.figure(figsize=(8, 5))
    plt.fill_between(
        x_pos,
        error_array - 2 * std_array,
        error_array + 2 * std_array,
        color='blue',
        alpha=0.15,
        label='±2 Std Dev',
        zorder=0
    )

    plt.plot(x_pos, error_array, marker='o', color='blue', label="L2 Error (mean)", zorder=2)

    plt.xticks(ticks=x_pos, labels=level_array)
    plt.xlabel("Number of Measurement Points")
    plt.ylabel("Relative L2 Error")
    plt.title(f"Corrector L2 Error vs. Scarce Data Level ({case_title})")
    plt.legend()
    plt.tight_layout()

    error_plot_path = os.path.join(corrector_dir, f"{case.lower()}-seeds-l2error-mean-std.png")
    plt.savefig(error_plot_path, dpi=300)
    plt.close()

    print(f"L2 error plot with error bars saved to {error_plot_path}")
    print("-" * 80)


def main(argv):

    FLAGS = flags.FLAGS

    flags.DEFINE_string("workdir", "."+os.sep+'results_sys', "Directory to store model data.")
    config_flags.DEFINE_config_file(
        "config",
        "examples/burgers/configs/fourier_emb.py",
        "File path to the training hyperparameter configuration.",
        lock_config=True,
    )
    
    config = FLAGS.config
    # SUBFOLDERS = [500]   
    # SUBFOLDERS = [100, 200, 300, 400, 1000]
    SUBFOLDERS = [100, 200, 300, 400, 1000, 8000]   

    # scarce_models, SUBFOLDERS = load_finetuned_models(config, FLAGS.workdir, case="BURGERS", target="Scarce-level-result")
    # evaluate_finetune_by_scarce_levels(config, FLAGS.workdir, scarce_models, case="BURGERS")

    # scarce_correctors, scarce_correctors_inputs, SUBFOLDERS = load_correctors_models(config, FLAGS.workdir, case="BURGERS", target="Scarce-level-result")    
    # evaluate_corrector_by_scarce_levels(config, FLAGS.workdir, scarce_correctors, scarce_correctors_inputs, case="BURGERS")

    scarce_models = load_seeds_models(config, FLAGS.workdir, case="BURGERS", target="scarce_seeds", SUBFOLDERS=SUBFOLDERS)
    evaluate_finetune_by_seeds(config, FLAGS.workdir, scarce_models, case="BURGERS")

    scarce_correctors, scarce_correctors_inputs = load_seeds_correctors(config, FLAGS.workdir, case="BURGERS", target="scarce_seeds", SUBFOLDERS=SUBFOLDERS)
    # evaluate_corrector_by_seeds(config, FLAGS.workdir, scarce_correctors, scarce_correctors_inputs, case="BURGERS")
    evaluate_corrector_dapinns_by_seeds(config, FLAGS.workdir, scarce_models, scarce_correctors, case="BURGERS")

if __name__ == "__main__":
    app.run(main)
