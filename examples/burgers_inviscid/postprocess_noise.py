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

def load_seeds_models(config, workdir: str, case: str, target: str, SUBFOLDERS: list):
    lower_case = case.lower()

    models = dict()

    for sf in SUBFOLDERS:
        models[sf] = dict()
        scarce_dir = os.path.join(workdir, target, lower_case, str(sf))
        for noise_folder in os.listdir(scarce_dir):
            models[sf][noise_folder] = dict()
            noise_dir = os.path.join(scarce_dir, noise_folder)
            for seed_folder in os.listdir(noise_dir):
                seed_path = os.path.join(noise_dir, seed_folder, lower_case)
                model = Burgers(config).to(config.device)
                finetuned_model_dir = os.path.join(seed_path, "finetuned", "lbfgs_finetuned_model.pt")
                model.load_finetuned_model(finetuned_model_dir)
     
                models[sf][noise_folder][seed_folder] = model

    return models, SUBFOLDERS

def load_seeds_correctors(config, workdir: str, case: str, target: str, SUBFOLDERS: list):
    lower_case = case.lower()
    
    correctors = dict()
    correctors_inputs = dict()

    for sf in SUBFOLDERS:
        correctors[sf] = dict()
        correctors_inputs[sf] = dict()
        scarce_dir = os.path.join(workdir, target, lower_case, str(sf))
        for noise_folder in os.listdir(scarce_dir):
            correctors[sf][noise_folder] = dict()
            correctors_inputs[sf][noise_folder] = dict()
            noise_dir = os.path.join(scarce_dir, noise_folder)
            for seed_folder in os.listdir(noise_dir):
                seed_path = os.path.join(noise_dir, seed_folder, lower_case)
                corrector_dir = os.path.join(seed_path, "corrector")
                corrector = Corrector(config).to(config.device)
                corrector.load_corrector_model(corrector_dir)
                correctors[sf][noise_folder][seed_folder] = corrector

                corrector_inputs_dir = os.path.join(corrector_dir, config.corrector_model_name)
                corrector_checkpoint = torch.load(corrector_inputs_dir, weights_only=False)
                corrector_inputs = corrector_checkpoint['corrector_inputs']
                correctors_inputs[sf][noise_folder][seed_folder] = corrector_inputs

    return correctors, correctors_inputs, SUBFOLDERS

def evaluate_finetune_by_seeds(config, workdir, models, case):
    postprocess_dir = os.path.join(workdir, "noise_postprocess")
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
    checklist = []
    for scarce_level in models.keys():
        y_plot_dict[scarce_level] = dict()
        y_std_dict[scarce_level] = dict()
        y_error_plot_dict[scarce_level] = dict()
        y_error_std_dict[scarce_level] = dict()
        for noise_level in models[scarce_level].keys():
            seeds_models = models[scarce_level][noise_level]
            y_list, y_error_list = [], [] # store the y prediction for each levels
            print(f"Evaluating model with scarce level: {scarce_level}, noise level: {noise_level}")
            for seed in seeds_models.keys():
                print(f"Evaluating model with seed: {seed}")
                model = seeds_models[seed]
                model.eval()
                y = model(xyt_test).cpu().detach().numpy().ravel()
                y_list.append(y)

                y_tx = y.reshape(n_x, n_t).T
                error = np.linalg.norm(y_tx - ground_truth) / np.linalg.norm(ground_truth)
                y_error_list.append(error)
                if np.isnan(error):
                    checklist.append([scarce_level, noise_level, seed])
            
            # prediction y
            y_array = np.stack(y_list, axis=0)  # shape: (num_seeds, num_points)
            y_mean = np.mean(y_array, axis=0)
            y_std = np.std(y_array, axis=0)

            y_plot_dict[scarce_level][noise_level] = y_mean
            y_std_dict[scarce_level][noise_level] = y_std

            # error 
            y_error_array = np.stack(y_error_list, axis=0)
            y_error_mean = np.mean(y_error_array)
            y_error_std = np.std(y_error_array)

            y_error_plot_dict[scarce_level][noise_level] = y_error_mean
            y_error_std_dict[scarce_level][noise_level] = y_error_std

            print(f"{noise_level}, Relative L2 error (mean prediction): {y_error_mean:.4e}")
    print(f"nan: {checklist}")
    finetuned_dir = os.path.join(case_dir, "finetuned")
    os.makedirs(finetuned_dir, exist_ok=True)

    sns.set_style("white")  # No grid

    title_map = {
        "QDHO": "Quadratic Damped Harmonic Oscillator",
        "BURGERS": "Burgers' Equation",
    }
    case_title = title_map.get(case, case)
    for scarce_level in y_plot_dict.keys():
        finetuned_scarce_level_dir = os.path.join(finetuned_dir, f"{scarce_level}")
        os.makedirs(finetuned_scarce_level_dir, exist_ok=True)
    
        time_fractions = [0.25, 0.50, 0.75]
        for noise_level, y_mean in y_plot_dict[scarce_level].items():
            noise_str = noise_level.split("_")[1]
            noise_percent = f"{float(noise_str) * 100:.2f}%"
            l2_error = y_error_plot_dict[scarce_level][noise_level]
            y_mean = y_mean.reshape(n_x, n_t)
            y_std = y_std_dict[scarce_level][noise_level]
            y_std = y_std.reshape(n_x, n_t)
            plt.figure(figsize=(9, 3), dpi=300)
            for i, frac in enumerate(time_fractions, start=1):
                t_idx = int(frac * n_t)
                plt.subplot(1, 3, i)
                plt.plot(x.squeeze(), ground_truth.T[:, t_idx], label='Ground Truth', color='black')
                plt.plot(x.squeeze(), y_mean[:, t_idx], label=f'{scarce_level} pts (L2: {l2_error:.2e})', linestyle='--', color='tab:red')
                plt.fill_between(x.squeeze(), y_mean[:, t_idx] - 2*y_std[:, t_idx], y_mean[:, t_idx] + 2*y_std[:, t_idx],label=f'±2 Std Dev ({scarce_level} pts)', alpha=0.2, color='red')
                plt.title(f"t = {frac}")
                plt.xlabel("x")
                if i == 1:
                    plt.ylabel("u(x, t)")
                plt.xlim(-1, 1)
                plt.ylim(-1.1, 1.1)
                if i == 3:
                    plt.legend(loc='lower right')
            plt.tight_layout()
            save_path = os.path.join(finetuned_scarce_level_dir, f"{noise_percent}_u_vs_x_slices.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved 3-slice plot to {save_path}")
    print("-" * 80)

    s_1, s_2, n = 400, 1000, "noise_0.05"
    n_p = n.split("_")[1]
    n_p = f"{float(n_p)*100:.2f}%"
    plt.figure(figsize=(9, 3), dpi=300)
    for i, frac in enumerate(time_fractions, start=1):
        y_mean_1 = y_plot_dict[s_1][n].reshape(n_x, n_t)
        y_mean_2 = y_plot_dict[s_2][n].reshape(n_x, n_t)
        y_std_1 = y_std_dict[s_1][n].reshape(n_x, n_t)
        y_std_2 = y_std_dict[s_2][n].reshape(n_x, n_t)
        l2_error_1 = y_error_plot_dict[s_1][n]
        l2_error_2 = y_error_plot_dict[s_2][n]
        plt.subplot(1, 3, i)
        plt.plot(x.squeeze(), ground_truth.T[:, t_idx], label='Ground Truth', color='black')
        plt.plot(x.squeeze(), y_mean_1[:, t_idx], label=f'{s_1} pts (L2: {l2_error_1:.2e})', linestyle='--', color='tab:red')
        plt.plot(x.squeeze(), y_mean_2[:, t_idx], label=f'{s_2} pts (L2: {l2_error_2:.2e})', linestyle='--', color='tab:blue')
        plt.fill_between(x.squeeze(), y_mean_1[:, t_idx]-2*y_std_1[:, t_idx], y_mean_1[:, t_idx]+2*y_std_1[:, t_idx], label=f'±2 Std Dev ({s_1} pts)', alpha=0.2, color='red')
        plt.fill_between(x.squeeze(), y_mean_2[:, t_idx]-2*y_std_2[:, t_idx], y_mean_2[:, t_idx]+2*y_std_2[:, t_idx], label=f'±2 Std Dev ({s_2} pts)', alpha=0.2, color='red')
        plt.title(f"t = {frac}")
        plt.xlabel("x")
        if i == 1:
            plt.ylabel("u(x, t)")
        plt.xlim(-1, 1)
        plt.ylim(-1.1, 1.1)
        if i == 3:
            plt.legend(loc='lower right')
    plt.tight_layout()
    save_path = os.path.join(finetuned_dir, f"comparison_{s_1}_{s_2}_{n}_u_vs_x_slices.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved 3-slice plot to {save_path}")
    # # --------------------------------
    # Plot mean L2 error ± std vs. scarce data level (with equal spacing on s)   
    level_ints = sorted([int(lvl) for lvl in y_plot_dict.keys()])

    for scarce_level in level_ints:
        finetuned_scarce_level_dir = os.path.join(finetuned_dir, f"{scarce_level}")
        noise_level_mean_dict = y_error_plot_dict[scarce_level]
        noise_level_std_dict = y_error_std_dict[scarce_level]

        assert noise_level_mean_dict.keys() == noise_level_std_dict.keys()

        # Extract and format noise levels
        noise_levels = list(noise_level_mean_dict.keys())
        noise_percent_labels = [
            f"{float(n.split('_')[1]) * 100:.2f}%" for n in noise_levels
        ]
        x_pos = np.arange(len(noise_percent_labels))

        # Get error values and standard deviations
        mean_errors = np.array([noise_level_mean_dict[n] for n in noise_levels])
        std_errors = np.array([noise_level_std_dict[n] for n in noise_levels])

        # Plot mean error and confidence interval
        plt.figure(figsize=(8, 5))
        plt.plot(
            x_pos,
            mean_errors,
            marker='o',
            color='blue',
            label=f"{scarce_level} pts L2 Error (mean)",
            zorder=2
        )
        plt.fill_between(
            x_pos,
            mean_errors - 2 * std_errors,
            mean_errors + 2 * std_errors,
            color='blue',
            alpha=0.15,
            label='±2 Std Dev',
            zorder=0
        )

        # Axis and legend
        plt.xticks(ticks=x_pos, labels=noise_percent_labels)
        plt.xlabel("Noise Level")
        plt.ylabel("Relative L2 Error")
        plt.title(f"DAPINNs L2 Error vs. Noise Level ({case_title})")
        plt.legend()
        plt.tight_layout()

        # Save plot
        plot_filename = f"{scarce_level}pts-{case.lower()}-seeds-l2error-mean-std.png"
        error_plot_path = os.path.join(finetuned_scarce_level_dir, plot_filename)
        plt.savefig(error_plot_path, dpi=300)
        plt.close()

        print(f"L2 error plot with error bars saved to {error_plot_path}")

    s_1, s_2 = 400, 1000
    scarce_levels = [s_1, s_2]

    plt.figure(figsize=(8, 5))

    for scarce_level, color in zip(scarce_levels, ['blue', 'red']):
        finetuned_scarce_level_dir = os.path.join(finetuned_dir, f"{scarce_level}")
        noise_level_mean_dict = y_error_plot_dict[scarce_level]
        noise_level_std_dict = y_error_std_dict[scarce_level]

        assert noise_level_mean_dict.keys() == noise_level_std_dict.keys()

        noise_levels = list(noise_level_mean_dict.keys())
        noise_percent_labels = [
            f"{float(n.split('_')[1]) * 100:.0f}%" for n in noise_levels
        ]
        x_pos = np.arange(len(noise_percent_labels))

        mean_errors = np.array([noise_level_mean_dict[n] for n in noise_levels])
        std_errors = np.array([noise_level_std_dict[n] for n in noise_levels])

        plt.plot(
            x_pos,
            mean_errors,
            marker='o',
            color=color,
            label=f"{scarce_level} pts L2 Error (mean)",
            zorder=2
        )
        plt.fill_between(
            x_pos,
            mean_errors - 2 * std_errors,
            mean_errors + 2 * std_errors,
            color=color,
            alpha=0.15,
            label=f"{scarce_level} pts ±2 Std Dev",
            zorder=0
        )

    # Axis settings
    plt.xticks(ticks=x_pos, labels=noise_percent_labels)
    plt.xlabel("Noise Level")
    plt.ylabel("Relative L2 Error")
    plt.title(f"DAPINNs L2 Error across Noise Levels under Limited Data Regimes ({s_1} vs. {s_2} Pts)")
    plt.legend()
    plt.tight_layout()

    # Save combined plot
    plot_filename = f"{s_1}_{s_2}pts-{case.lower()}-seeds-l2error-comparison.png"
    error_plot_path = os.path.join(finetuned_dir, plot_filename)
    plt.savefig(error_plot_path, dpi=300)
    plt.close()

    print(f"Combined L2 error plot saved to {error_plot_path}")

def evaluate_corrector_by_seeds(config, workdir, dapinns, models, case):
    postprocess_dir = os.path.join(workdir, "noise_postprocess")
    os.makedirs(postprocess_dir, exist_ok=True)
    
    # System parameters
    params = config.system_viscous_burgers.system_params
    n_x = params['n_x']
    n_t = params['n_t']
    if case == "BURGERS":
        case_dir = os.path.join(postprocess_dir, "BURGERS")
        xyt_test, u_test, x, t, u, dudx_test, d2udx2_test = _get_viscous_burgers_test(config, workdir)
        ground_truth = u.cpu().detach().numpy() * dudx_test.cpu().detach().numpy()
        ground_truth = ground_truth.T

    os.makedirs(case_dir, exist_ok=True)
    y_plot_dict = dict()
    y_std_dict = dict()
    y_error_plot_dict = dict()
    y_error_std_dict = dict()
    checklist = []
    for scarce_level in models.keys():
        y_plot_dict[scarce_level] = dict()
        y_std_dict[scarce_level] = dict()
        y_error_plot_dict[scarce_level] = dict()
        y_error_std_dict[scarce_level] = dict()
        for noise_level in models[scarce_level].keys():
            seeds_models = models[scarce_level][noise_level]
            seeds_dapinns = dapinns[scarce_level][noise_level]
            y_list, y_error_list = [], [] # store the y prediction for each levels
            print(f"Evaluating model with scarce level: {scarce_level}, noise level: {noise_level}")
            for seed in seeds_models.keys():
                print(f"Evaluating model with seed: {seed}")
                xyt_test.requires_grad_(True)
                dapinn = seeds_dapinns[seed]
                dapinn.eval()
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
                if np.isnan(error):
                    checklist.append([scarce_level, noise_level, seed])
                y_error_list.append(error)
            
            # prediction y
            y_array = np.stack(y_list, axis=0)  # shape: (num_seeds, num_points)
            y_mean = np.mean(y_array, axis=0)
            y_std = np.std(y_array, axis=0)

            y_plot_dict[scarce_level][noise_level] = y_mean
            y_std_dict[scarce_level][noise_level] = y_std

            # error 
            y_error_array = np.stack(y_error_list, axis=0)
            y_error_mean = np.mean(y_error_array)
            y_error_std = np.std(y_error_array)

            y_error_plot_dict[scarce_level][noise_level] = y_error_mean
            y_error_std_dict[scarce_level][noise_level] = y_error_std

            print(f"{noise_level}, Relative L2 error (mean prediction): {y_error_mean:.4e}")
    print(f"nan: {checklist}")
    corrector_dir = os.path.join(case_dir, "corrector")
    os.makedirs(corrector_dir, exist_ok=True)

    sns.set_style("white")  # No grid

    title_map = {
        "QDHO": "Quadratic Damped Harmonic Oscillator",
        "DHO": "Damped Harmonic Oscillator",
        "BURGERS": "Burgers' Equation"
    }
    case_title = title_map.get(case, case)
    for scarce_level in y_plot_dict.keys():
        corrector_scarce_level_dir = os.path.join(corrector_dir, f"{scarce_level}")
        os.makedirs(corrector_scarce_level_dir, exist_ok=True)
    
        time_fractions = [0.25, 0.50, 0.75]
        for noise_level, y_mean in y_plot_dict[scarce_level].items():
            noise_str = noise_level.split("_")[1]
            noise_percent = f"{float(noise_str) * 100:.2f}%"
            l2_error = y_error_plot_dict[scarce_level][noise_level]
            y_mean = y_mean.reshape(n_x, n_t)
            y_std = y_std_dict[scarce_level][noise_level]
            y_std = y_std.reshape(n_x, n_t)
            plt.figure(figsize=(9, 3), dpi=300)
            for i, frac in enumerate(time_fractions, start=1):
                t_idx = int(frac * n_t)
                plt.subplot(1, 3, i)
                plt.plot(x.squeeze(), ground_truth.T[:, t_idx], label='Ground Truth', color='black')
                plt.plot(x.squeeze(), y_mean[:, t_idx], label=f'{scarce_level} pts (L2: {l2_error:.2e})', linestyle='--', color='tab:red')
                plt.fill_between(x.squeeze(), y_mean[:, t_idx] - 2*y_std[:, t_idx], y_mean[:, t_idx] + 2*y_std[:, t_idx],label=f'±2 Std Dev ({scarce_level} pts)', alpha=0.2, color='red')
                plt.title(f"t = {frac}")
                plt.xlabel("x")
                if i == 1:
                    plt.ylabel("u(x, t)")
                if i == 3:
                    plt.legend(loc='lower right')
            plt.tight_layout()
            save_path = os.path.join(corrector_scarce_level_dir, f"{noise_percent}_u_vs_x_slices.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved 3-slice plot to {save_path}")
    print("-" * 80)
   

    s_1, s_2, n = 400, 1000, "noise_0.05"
    n_p = n.split("_")[1]
    n_p = f"{float(n_p)*100:.2f}%"
    plt.figure(figsize=(9, 3), dpi=300)
    for i, frac in enumerate(time_fractions, start=1):
        y_mean_1 = y_plot_dict[s_1][n].reshape(n_x, n_t)
        y_mean_2 = y_plot_dict[s_2][n].reshape(n_x, n_t)
        y_std_1 = y_std_dict[s_1][n].reshape(n_x, n_t)
        y_std_2 = y_std_dict[s_2][n].reshape(n_x, n_t)
        l2_error_1 = y_error_plot_dict[s_1][n]
        l2_error_2 = y_error_plot_dict[s_2][n]
        plt.subplot(1, 3, i)
        plt.plot(x.squeeze(), ground_truth.T[:, t_idx], label='Ground Truth', color='black')
        plt.plot(x.squeeze(), y_mean_1[:, t_idx], label=f'{s_1} pts (L2: {l2_error_1:.2e})', linestyle='--', color='tab:red')
        plt.plot(x.squeeze(), y_mean_2[:, t_idx], label=f'{s_2} pts (L2: {l2_error_2:.2e})', linestyle='--', color='tab:blue')
        plt.fill_between(x.squeeze(), y_mean_1[:, t_idx]-2*y_std_1[:, t_idx], y_mean_1[:, t_idx]+2*y_std_1[:, t_idx], label=f'±2 Std Dev ({s_1} pts)', alpha=0.2, color='red')
        plt.fill_between(x.squeeze(), y_mean_2[:, t_idx]-2*y_std_2[:, t_idx], y_mean_2[:, t_idx]+2*y_std_2[:, t_idx], label=f'±2 Std Dev ({s_2} pts)', alpha=0.2, color='red')
        plt.title(f"t = {frac}")
        plt.xlabel("x")
        if i == 1:
            plt.ylabel("u(x, t)")

        if i == 3:
            plt.legend(loc='lower right')
    plt.tight_layout()
    save_path = os.path.join(corrector_dir, f"comparison_{s_1}_{s_2}_{n}_u_vs_x_slices.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved 3-slice plot to {save_path}")
    # # --------------------------------
    # Plot mean L2 error ± std vs. scarce data level (with equal spacing on x-axis)   
    # level_ints = sorted([int(lvl) for lvl in y_plot_dict.keys()])
    # for scarce_level in level_ints:
    #     corrector_scarce_level_dir = os.path.join(corrector_dir, f"{scarce_level}")
    #     noise_level_mean_dict = y_error_plot_dict[scarce_level]
    #     noise_level_std_dict = y_error_std_dict[scarce_level]
    #     assert noise_level_mean_dict.keys() == noise_level_std_dict.keys()
    #     noise_levels = list(noise_level_mean_dict.keys())
    #     noise_str_list = [n_level.split("_")[1] for n_level in noise_levels]
    #     noise_percent = [f"{float(n_s) * 100:.0f}%" for n_s in noise_str_list]
    #     noise_percent_array = np.array(noise_percent)
    #     x_pos = np.arange(len(noise_percent_array))

    #     error_array = np.array([noise_level_mean_dict[level] for level in noise_levels])
    #     std_array = np.array([noise_level_std_dict[level] for level in noise_levels])

    #     plt.figure(figsize=(8, 5))
    #     plt.plot(x_pos, error_array, marker='o', color='blue', label=f"{scarce_level} pts L2 Error (mean)", zorder=2)
    #     plt.fill_between(
    #         x_pos,
    #         error_array - 2 * std_array,
    #         error_array + 2 * std_array,
    #         color='blue',
    #         alpha=0.15,
    #         label='±2 Std Dev',
    #         zorder=0
    #     )
    #     plt.xticks(ticks=x_pos, labels=noise_percent_array)
    #     plt.xlabel("Noise Level")
    #     plt.ylabel("Relative L2 Error")
    #     plt.title(f"Corrector L2 Error vs. Noise Level ({case_title})")
    #     plt.legend()
    #     plt.tight_layout()

    #     error_plot_path = os.path.join(corrector_scarce_level_dir, f"{scarce_level}pts-{case.lower()}-seeds-l2error-mean-std.png")
    #     plt.savefig(error_plot_path, dpi=300)
    #     plt.close()
    #     print(f"L2 error plot with error bars saved to {error_plot_path}")

    s_1, s_2 = 400, 1000
    scarce_levels = [s_1, s_2]

    plt.figure(figsize=(8, 5))

    for scarce_level, color in zip(scarce_levels, ['blue', 'red']):
        noise_level_mean_dict = y_error_plot_dict[scarce_level]
        noise_level_std_dict = y_error_std_dict[scarce_level]

        assert noise_level_mean_dict.keys() == noise_level_std_dict.keys()

        noise_levels = list(noise_level_mean_dict.keys())
        noise_percent_labels = [
            f"{float(n.split('_')[1]) * 100:.2f}%" for n in noise_levels
        ]
        x_pos = np.arange(len(noise_percent_labels))

        mean_errors = np.array([noise_level_mean_dict[n] for n in noise_levels])
        std_errors = np.array([noise_level_std_dict[n] for n in noise_levels])

        plt.plot(
            x_pos,
            mean_errors,
            marker='o',
            color=color,
            label=f"{scarce_level} pts L2 Error (mean)",
            zorder=2
        )
        plt.fill_between(
            x_pos,
            mean_errors - 2 * std_errors,
            mean_errors + 2 * std_errors,
            color=color,
            alpha=0.15,
            label=f"{scarce_level} pts ±2 Std Dev",
            zorder=0
        )

    # Axis settings
    plt.xticks(ticks=x_pos, labels=noise_percent_labels)
    plt.xlabel("Noise Level")
    plt.ylabel("Relative L2 Error")
    plt.title(f"Corrector L2 Error across Noise Levels under Limited Data Regimes ({s_1} vs. {s_2} Pts)")
    plt.legend()
    plt.tight_layout()

    # Save combined plot
    plot_filename = f"{s_1}_{s_2}pts-{case.lower()}-seeds-l2error-comparison.png"
    error_plot_path = os.path.join(corrector_dir, plot_filename)
    plt.savefig(error_plot_path, dpi=300)
    plt.close()

    print(f"Combined L2 error plot saved to {error_plot_path}")

    print("-" * 80)

def main(argv):

    FLAGS = flags.FLAGS
    SUBFOLDERS = [400, 1000]
    flags.DEFINE_string("workdir", "."+os.sep+"results_sys", "Directory to store model data.")
    config_flags.DEFINE_config_file(
        "config",
        "examples/burgers/configs/fourier_emb_noise.py",
        "File path to the training hyperparameter configuration.",
        lock_config=True,
    )
    
    config = FLAGS.config

    noise_models, SUBFOLDERS = load_seeds_models(config, FLAGS.workdir, case="BURGERS", target="noise_seeds", SUBFOLDERS=SUBFOLDERS)
    evaluate_finetune_by_seeds(config, FLAGS.workdir, noise_models, case="BURGERS")
    noise_correctors, _, SUBFOLDERS = load_seeds_correctors(config, FLAGS.workdir, case="BURGERS", target="noise_seeds", SUBFOLDERS=SUBFOLDERS)
    evaluate_corrector_by_seeds(config, FLAGS.workdir, noise_models, noise_correctors, case="BURGERS")
    
if __name__ == "__main__":
    app.run(main)


