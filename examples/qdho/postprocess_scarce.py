import os
import torch
import ml_collections
import seaborn as sns
from examples.qdho.models import Qdho, Corrector
from examples.qdho.utils import generate_qdho_dataset
from dapinns.samplers import UniformSampler, RandomSampler
import matplotlib.pyplot as plt
import numpy as np
from ml_collections import config_flags
from absl import app
from absl import flags
from kneed import KneeLocator

def _get_qdho_test(config: ml_collections.ConfigDict, workdir: str):

    params = config.system_qdho.system_params
    T = params['T']
    x0 = params['x0']
    v0 = params['v0']
    n_t = params['n_t']

    # Generate dataset
    x, y, sol = generate_qdho_dataset(params, T=T, x0=x0, v0=v0, n_t=n_t) # qdho
    sampler = UniformSampler(sample_size=config.system_qdho.system_params['n_t'])
    x_test, y_test = sampler.generate_data(x, y)
    x_test = x_test.to(config.device)
    y_test = y_test.to(config.device)

    return x_test, y_test, sol

# def load_finetuned_models(config: ml_collections.ConfigDict, workdir: str, case: str, target: str):
#     lower_case = case.lower()
#     if target == "Scarce-level-result":
#         SUBFOLDERS = [1, 3, 5, 10, 15, 100, 10000]   
#     elif target == "Noise-level-result":
#         SUBFOLDERS = [0.01, 0.02, 0.03, 0.05, str(0.10), 0.15]
#         case = case + "-scarce-data"

#     models = dict()
#     for sf in SUBFOLDERS:
#         model = Qdho(config).to(config.device)
#         finetuned_model_dir = os.path.join(workdir, target, case, str(sf), lower_case, "finetuned", "lbfgs_finetuned_model.pt")
#         model.load_finetuned_model(finetuned_model_dir)
#         models[sf] = model

#     return models, SUBFOLDERS

# def evaluate_finetune_by_scarce_levels(config, workdir, models, case):

#     postprocess_dir = os.path.join(workdir, "postprocess")
#     os.makedirs(postprocess_dir, exist_ok=True)
    
#     if case == "QDHO":
#         case_dir = os.path.join(postprocess_dir, "QDHO")
#         x_test, y_test, sol = _get_qdho_test(config, workdir)

#     os.makedirs(case_dir, exist_ok=True)
#     t = sol.t
#     ground_truth = sol.y[0]

#     y_plot_dict = dict()
#     y_error_dict = dict()
#     for level, model in models.items():
#         print(f"Evaluating model with scarce level: {level}")

#         # store the y prediction for each level
#         model.eval()
#         y = model(x_test).cpu().detach().numpy().ravel()
#         y_plot_dict[level] = y

#         # Relative L2 error
#         y_error = np.linalg.norm(y - ground_truth) / np.linalg.norm(ground_truth)
#         y_error_dict[level] = y_error
#         print(f"Relative L2 error: {y_error:.4e}")

#     finetuned_dir = os.path.join(case_dir, "finetuned")
#     os.makedirs(finetuned_dir, exist_ok=True)

#     sns.set_style("white")  # No grid

#     title_map = {
#         "DHO": "Damped Harmonic Oscillator",
#         "QDHO": "Quadratic Damped Harmonic Oscillator",
#         "BURGERS": "Burgers' Equation"
#     }
#     case_title = title_map.get(case, case)

#     plt.figure(figsize=(10, 6))
#     plt.plot(t, ground_truth, label="Reference", linewidth=2, color="black")

#     level_ints = sorted([int(lvl) for lvl in y_plot_dict.keys()])
#     min_level, max_level = min(level_ints), max(level_ints)

#     colors = plt.cm.viridis(np.linspace(0, 1, len(level_ints)))
#     min_alpha, max_alpha = 0.3, 1.0

#     for i, level in enumerate(level_ints):
#         y = y_plot_dict[level]
#         l2_error = y_error_dict[level]

#         alpha = min_alpha + (level - min_level) / (max_level - min_level + 1e-8) * (max_alpha - min_alpha)

#         plt.plot(t, y, label=f"{level} pts (L2: {l2_error:.2e})",
#             color=colors[i], linestyle="--", alpha=alpha)

#     plt.xlabel("Time (t)")
#     plt.ylabel("Displacement x(t)")
#     plt.title(f"DAPINNs Prediction on {case_title}")
#     plt.legend(loc="best", frameon=True)
#     plt.tight_layout()

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
#         S=1,
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
#         SUBFOLDERS = [1, 3, 5, 10, 15, 100, 10000]   
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

#     postprocess_dir = os.path.join(workdir, "postprocess")
#     os.makedirs(postprocess_dir, exist_ok=True)
    
#     if case == "QDHO":
#         case_dir = os.path.join(postprocess_dir, "QDHO")
#         x_test, y_test, sol = _get_qdho_test(config, workdir)
#         t = sol.t
#         dx = sol.y[1]
#         c = config.system_qdho.system_params['c']
#         ground_truth = c * dx

#     os.makedirs(case_dir, exist_ok=True)

#     y_plot_dict = dict()
#     y_error_dict = dict()
#     for level in models.keys():

#         model = models[level]
#         input_tensor = inputs[level]
#         print(f"Evaluating corrector with scarce level: {level}")

#         # store the y prediction for each level
#         model.eval()
#         y = model(input_tensor)
#         y = y.cpu().detach().numpy().ravel()[::10] # align
#         y_plot_dict[level] = y

#         # Relative L2 error
#         y_error = np.linalg.norm(y - ground_truth) / np.linalg.norm(ground_truth)
#         y_error_dict[level] = y_error
#         print(f"Relative L2 error: {y_error:.4e}")

#     finetuned_dir = os.path.join(case_dir, "corrector")
#     os.makedirs(finetuned_dir, exist_ok=True)

#     sns.set_style("white")  # No grid

#     title_map = {
#         "DHO": "Damped Harmonic Oscillator",
#         "QDHO": "Quadratic Damped Harmonic Oscillator",
#         "BURGERS": "Burgers' Equation"
#     }
#     case_title = title_map.get(case, case)

#     plt.figure(figsize=(10, 6))
#     plt.plot(t, ground_truth, label="Reference", linewidth=2, color="black")

#     level_ints = sorted([int(lvl) for lvl in y_plot_dict.keys()])
#     min_level, max_level = min(level_ints), max(level_ints)

#     colors = plt.cm.viridis(np.linspace(0, 1, len(level_ints)))
#     min_alpha, max_alpha = 0.3, 1.0

#     for i, level in enumerate(level_ints):
#         y = y_plot_dict[level]
#         l2_error = y_error_dict[level]

#         alpha = min_alpha + (level - min_level) / (max_level - min_level + 1e-8) * (max_alpha - min_alpha)

#         plt.plot(t, y, label=f"{level} pts (L2: {l2_error:.2e})",
#                 color=colors[i], linestyle="--", alpha=alpha)

#     plt.xlabel("Time (t)")
#     plt.ylabel("Displacement x(t)")
#     plt.title(f"DAPINNs Corrector Prediction on {case_title}")
#     plt.legend(loc="best", frameon=True)
#     plt.tight_layout()

#     save_path = os.path.join(finetuned_dir, f"{case.lower()}-corrector-scarce-level.png")
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
#         S=1
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
#     plt.title(f"Corrector L2 Error vs. Scarce Data Level ({case_title})")
#     plt.legend()
#     plt.tight_layout()

#     error_plot_path = os.path.join(finetuned_dir, f"{case.lower()}-l2error-vs-scarce.png")
#     plt.savefig(error_plot_path, dpi=300)
#     plt.close()

#     print(f"L2 error plot saved to {error_plot_path}")
#     print("-" * 80)

def load_seeds_models(config, workdir: str, case: str, target: str, SUBFOLDERS=None):
    lower_case = case.lower()
    
    if SUBFOLDERS is None:
        SUBFOLDERS = [1, 3, 5, 10, 15, 100, 10000]

    models = dict()

    for sf in SUBFOLDERS:
        models[sf] = dict()

        scarce_dir = os.path.join(workdir, target, lower_case, str(sf))

        for seed_folder in os.listdir(scarce_dir):
            seed_path = os.path.join(scarce_dir, seed_folder, lower_case)
            model = Qdho(config).to(config.device)
            finetuned_model_dir = os.path.join(seed_path, "finetuned", "lbfgs_finetuned_model.pt")
            model.load_finetuned_model(finetuned_model_dir)

            models[sf][seed_folder] = model

    return models, SUBFOLDERS

def load_seeds_correctors(config, workdir: str, case: str, target: str, SUBFOLDERS=None):
    lower_case = case.lower()
    
    if SUBFOLDERS is None:
        SUBFOLDERS = [1, 3, 5, 10, 15, 100, 10000]

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

    return correctors, correctors_inputs, SUBFOLDERS

def evaluate_finetune_by_seeds(config, workdir, models, case):
    postprocess_dir = os.path.join(workdir, "scarce_postprocess")
    os.makedirs(postprocess_dir, exist_ok=True)
    
    if case == "QDHO":
        case_dir = os.path.join(postprocess_dir, "QDHO")
        x_test, y_test, sol = _get_qdho_test(config, workdir)

    os.makedirs(case_dir, exist_ok=True)
    
    t = sol.t
    ground_truth = sol.y[0]

    y_plot_dict = dict()
    y_std_dict = dict()
    y_error_plot_dict = dict()
    y_error_std_dict = dict()
    for level in models.keys():
        seeds_models = models[level]
        print(f"Evaluating model with scarce level: {level}")

        # store the y prediction for each level
        y_list, y_error_list = [], []
        for seed in seeds_models.keys():
            model = seeds_models[seed]
            print(f"Evaluating model with seed: {seed}")
            model.eval()
            y = model(x_test).cpu().detach().numpy().ravel()
            y_list.append(y)

            error = np.linalg.norm(y - ground_truth) / np.linalg.norm(ground_truth)
            y_error_list.append(error)

        y_array = np.stack(y_list, axis=0)  # shape: (num_seeds, num_points)
        y_mean = np.mean(y_array, axis=0)
        y_std = np.std(y_array, axis=0)

        # Relative L2 error of mean prediction
        y_error_array = np.stack(y_error_list, axis=0)
        y_error_mean = np.mean(y_error_array)
        y_error_std = np.std(y_error_array)

        # Store
        y_plot_dict[level] = y_mean
        y_std_dict[level] = y_std
        
        y_error_plot_dict[level] = y_error_mean
        y_error_std_dict[level] = y_error_std
        print(f"Relative L2 error (mean prediction): {y_error_mean:.4e}")

    finetuned_dir = os.path.join(case_dir, "finetuned")
    os.makedirs(finetuned_dir, exist_ok=True)

    sns.set_style("white")  # No grid

    title_map = {
        "DHO": "Damped Harmonic Oscillator",
        "QDHO": "Quadratic Damped Harmonic Oscillator",
        "BURGERS": "Burgers' Equation"
    }
    case_title = title_map.get(case, case)

    for level, y_mean in y_plot_dict.items():
        l2_error = y_error_plot_dict[level]
        plt.figure(figsize=(10, 6))
        plt.plot(t, ground_truth, label="Reference", linewidth=2, color="dimgray")
        plt.plot(t, y_mean, label=f"{level} pts (L2: {l2_error:.2e})", linestyle="--", alpha=0.6, color="red")
        plt.fill_between(t, y_mean - 2*y_std_dict[level], y_mean + 2*y_std_dict[level],label=f'±2 Std Dev ({level} pts)', alpha=0.2, color='red')
        plt.xlabel("Time (t)")
        plt.ylabel("Displacement x(t)")
        plt.title(f"DAPINNs Prediction (mean) on {case_title}")
        plt.legend(loc="best", frameon=True)
        plt.savefig(os.path.join(finetuned_dir, f"{case.lower()}-{str(level)}pts-finetune-scarce-level.png"), dpi=300)
    plt.figure(figsize=(10, 6))
    plt.plot(t, ground_truth, label="Reference", linewidth=2, color="dimgray")
    plt.plot(t, y_plot_dict[10], label=f"10 pts (L2: {y_error_plot_dict[10]:.2e})", linestyle="--", alpha=0.6, color='red')
    plt.plot(t, y_plot_dict[30], label=f"30 pts (L2: {y_error_plot_dict[30]:.2e})", linestyle="--", alpha=0.6, color='blue')
    plt.fill_between(t, y_plot_dict[10] - 2*y_std_dict[10], y_mean + 2*y_std_dict[10], label='±2 Std Dev (10 pts)', alpha=0.2, color='red')
    plt.fill_between(t, y_plot_dict[30] - 2*y_std_dict[30], y_mean + 2*y_std_dict[30], label='±2 Std Dev (30 pts)', alpha=0.2, color='blue')
    plt.xlabel("Time (t)")
    plt.ylabel("Displacement x(t)")
    plt.title(f"DAPINNs Prediction (mean) on {case_title}")
    plt.legend(loc="best", frameon=True)
    plt.savefig(os.path.join(finetuned_dir, f"comparison-{case.lower()}-{str(level)}pts-finetune-scarce-level.png"), dpi=300)

    # --------------------------------
    # Plot mean L2 error ± std vs. scarce data level (with equal spacing on x-axis)   
    level_ints = sorted([int(lvl) for lvl in y_plot_dict.keys()])
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
    
    if case == "QDHO":
        case_dir = os.path.join(postprocess_dir, "QDHO")
        x_test, y_test, sol = _get_qdho_test(config, workdir)
        t = sol.t
        dx = sol.y[1]
        c = config.system_qdho.system_params['c']
        ground_truth = c * dx * abs(dx)

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
            # print(f"Evaluating model with seed: {seed}")
            model.eval()
            y = model(input_tensor)
            y = y.cpu().detach().numpy().ravel()[::10]  # align
            y_list.append(y)

            error = np.linalg.norm(y - ground_truth) / np.linalg.norm(ground_truth)
            y_error_list.append(error)
            if np.isnan(error):
                check_list.append((level, seed))

        y_array = np.stack(y_list, axis=0)  # shape: (num_seeds, num_points)
        y_mean = np.mean(y_array, axis=0)
        y_std = np.std(y_array, axis=0)

        # Relative L2 error of mean prediction
        y_error_array = np.stack(y_error_list, axis=0)
        y_error_mean = np.mean(y_error_array)
        y_error_std = np.std(y_error_array)

        # Store
        y_plot_dict[level] = y_mean
        y_std_dict[level] = y_std

        y_error_plot_dict[level] = y_error_mean
        y_error_std_dict[level] = y_error_std
        print(f"Evaluating corrector with scarce level: {level}, Relative L2 error (mean prediction): {y_error_mean:.4e}")
    print(check_list)
    corrector_dir = os.path.join(case_dir, "corrector")
    os.makedirs(corrector_dir, exist_ok=True)

    sns.set_style("white")  # No grid

    title_map = {
        "DHO": "Damped Harmonic Oscillator",
        "QDHO": "Quadratic Damped Harmonic Oscillator",
        "BURGERS": "Burgers' Equation"
    }
    case_title = title_map.get(case, case)

    for level, y_mean in y_plot_dict.items():
        l2_error = y_error_plot_dict[level]
        plt.figure(figsize=(10, 6))
        plt.plot(t, ground_truth, label="Reference", linewidth=2, color="dimgray")
        plt.plot(t, y_mean, label=f"{level} pts (L2: {l2_error:.2e})", linestyle="--", alpha=0.6, color="red")
        plt.fill_between(t, y_mean - 2*y_std_dict[level], y_mean + 2*y_std_dict[level],label=f'±2 Std Dev ({level} pts)', alpha=0.2, color='red')
        plt.xlabel("Time (t)")
        plt.ylabel("Displacement x(t)")
        plt.title(f"Corrector Prediction (mean) on {case_title}")
        plt.legend(loc="best", frameon=True)
        plt.savefig(os.path.join(corrector_dir, f"{case.lower()}-{str(level)}pts-corrector-scarce-level.png"), dpi=300)
    plt.figure(figsize=(10, 6))
    plt.plot(t, ground_truth, label="Reference", linewidth=2, color="dimgray")
    plt.plot(t, y_plot_dict[10], label=f"10 pts (L2: {y_error_plot_dict[10]:.2e})", linestyle="--", alpha=0.6, color='red')
    plt.plot(t, y_plot_dict[30], label=f"30 pts (L2: {y_error_plot_dict[30]:.2e})", linestyle="--", alpha=0.6, color='blue')
    plt.fill_between(t, y_plot_dict[10] - 2*y_std_dict[10], y_mean + 2*y_std_dict[10], label='±2 Std Dev (10 pts)', alpha=0.2, color='red')
    plt.fill_between(t, y_plot_dict[30] - 2*y_std_dict[30], y_mean + 2*y_std_dict[30], label='±2 Std Dev (30 pts)', alpha=0.2, color='blue')
    plt.xlabel("Time (t)")
    plt.ylabel("Displacement x(t)")
    plt.title(f"Corrector Prediction (mean) on {case_title}")
    plt.legend(loc="best", frameon=True)
    plt.savefig(os.path.join(corrector_dir, f"comparison-{case.lower()}-{str(level)}pts-corrector-scarce-level.png"), dpi=300)

    # --------------------------------
    # Plot mean L2 error ± std vs. scarce data level (with equal spacing on x-axis) 
    level_ints = sorted([int(lvl) for lvl in y_plot_dict.keys()])  
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
    SUBFOLDERS = [10, 15, 20, 30, 100, 1000]
    flags.DEFINE_string("workdir", "."+os.sep+"results_sys", "Directory to store model data.")
    config_flags.DEFINE_config_file(
        "config",
        "examples/qdho/configs/default.py",
        "File path to the training hyperparameter configuration.",
        lock_config=True,
    )
    
    config = FLAGS.config

    scarce_models, SUBFOLDERS = load_seeds_models(config, FLAGS.workdir, case="QDHO", target="scarce_seeds", SUBFOLDERS=SUBFOLDERS)
    evaluate_finetune_by_seeds(config, FLAGS.workdir, scarce_models, case="QDHO")
    scarce_correctors, scarce_correctors_inputs, SUBFOLDERS = load_seeds_correctors(config, FLAGS.workdir, case="QDHO", target="scarce_seeds", SUBFOLDERS=SUBFOLDERS)
    evaluate_corrector_by_seeds(config, FLAGS.workdir, scarce_correctors, scarce_correctors_inputs, case="QDHO")
    
    #---------------- noise levels -----------------
    # noise_models, SUBFOLDERS = load_finetuned_models(config, FLAGS.workdir, case="QDHO", target="Noise-level-result")
    # evaluate_finetune_by_noise_levels(config, FLAGS.workdir, noise_models, case="QDHO")
    # noise_correctors, noise_correctors_inputs, SUBFOLDERS = load_correctors_models(config, FLAGS.workdir, case="QDHO", target="Noise-level-result")
    # evaluate_corrector_by_noise_levels(config, FLAGS.workdir, scarce_correctors, scarce_correctors_inputs, case="QDHO")
    

if __name__ == "__main__":
    app.run(main)


