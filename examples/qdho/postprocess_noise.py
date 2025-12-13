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
                model = Qdho(config).to(config.device)
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
                y = model(x_test).cpu().detach().numpy().ravel()
                y_list.append(y)

                error = np.linalg.norm(y - ground_truth) / np.linalg.norm(ground_truth)
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
    finetuned_dir = os.path.join(case_dir, "finetuned")
    os.makedirs(finetuned_dir, exist_ok=True)

    sns.set_style("white")  # No grid

    title_map = {
        "QDHO": "Quadratic Damped Harmonic Oscillator",
    }
    case_title = title_map.get(case, case)

    for scarce_level in y_plot_dict.keys():
        finetuned_scarce_level_dir = os.path.join(finetuned_dir, f"{scarce_level}")
        os.makedirs(finetuned_scarce_level_dir, exist_ok=True)
        for noise_level, y_mean in y_plot_dict[scarce_level].items():
            noise_str = noise_level.split("_")[1]
            noise_percent = f"{float(noise_str) * 100:.0f}%"
            l2_error = y_error_plot_dict[scarce_level][noise_level]
            y_std = y_std_dict[scarce_level][noise_level]
            plt.figure(figsize=(8, 5))
            plt.plot(t, ground_truth, label="Reference", linewidth=2, color="dimgray")
            plt.plot(t, y_mean, label=f"{scarce_level} pts (L2: {l2_error:.2e})", linestyle="--", alpha=0.6, color="red")
            plt.fill_between(t, y_mean - 2*y_std, y_mean + 2*y_std,label=f'±2 Std Dev ({scarce_level} pts)', alpha=0.2, color='red')
            plt.xlabel("Time (t)")
            plt.ylabel("Displacement x(t)")
            plt.title(f"DAPINNs Prediction (mean) on {case_title} - Noise: {noise_percent}")
            plt.legend(loc="best", frameon=True)
            plt.savefig(os.path.join(finetuned_scarce_level_dir, f"{case.lower()}-{noise_str}-{scarce_level}pts-finetune-scarce-level.png"), dpi=300)
    plt.figure(figsize=(8, 5))
    s_1, s_2, n = 30, 100, "noise_0.02"
    n_p = n.split("_")[1]
    n_p = f"{float(n_p)*100:.0f}%"
    plt.plot(t, ground_truth, label="Reference", linewidth=2, color="dimgray")
    plt.plot(t, y_plot_dict[s_1][n], label=f"30 pts (L2: {y_error_plot_dict[s_1][n]:.2e})", linestyle="--", alpha=0.6, color='red')
    plt.plot(t, y_plot_dict[s_2][n], label=f"100 pts (L2: {y_error_plot_dict[s_2][n]:.2e})", linestyle="--", alpha=0.6, color='blue')
    plt.fill_between(t, y_plot_dict[s_1][n] - 2*y_std_dict[s_1][n], y_plot_dict[s_1][n] + 2*y_std_dict[s_1][n], label=f'±2 Std Dev ({s_1} pts)', alpha=0.2, color='red')
    plt.fill_between(t, y_plot_dict[s_2][n] - 2*y_std_dict[s_2][n], y_plot_dict[s_2][n] + 2*y_std_dict[s_2][n], label=f'±2 Std Dev ({s_2} pts)', alpha=0.2, color='blue')
    plt.xlabel("Time (t)")
    plt.ylabel("Displacement x(t)")
    plt.title(f"DAPINNs Prediction (mean) on {case_title} - Noise: {n_p}")
    plt.legend(loc="best", frameon=True)
    plt.savefig(os.path.join(finetuned_dir, f"comparison-{case.lower()}-finetune-scarce-noise-level.png"), dpi=300)

    # # --------------------------------
    # Plot mean L2 error ± std vs. scarce data level (with equal spacing on x-axis)   
    level_ints = sorted([int(lvl) for lvl in y_plot_dict.keys()])

    for scarce_level in level_ints:
        finetuned_scarce_level_dir = os.path.join(finetuned_dir, f"{scarce_level}")
        noise_level_mean_dict = y_error_plot_dict[scarce_level]
        noise_level_std_dict = y_error_std_dict[scarce_level]

        assert noise_level_mean_dict.keys() == noise_level_std_dict.keys()

        # Extract and format noise levels
        noise_levels = list(noise_level_mean_dict.keys())
        noise_percent_labels = [
            f"{float(n.split('_')[1]) * 100:.0f}%" for n in noise_levels
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

    s_1, s_2 = 30, 100
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

def evaluate_corrector_by_seeds(config, workdir, models, inputs, case):
    postprocess_dir = os.path.join(workdir, "noise_postprocess")
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
    checklist = []
    for scarce_level in models.keys():
        y_plot_dict[scarce_level] = dict()
        y_std_dict[scarce_level] = dict()
        y_error_plot_dict[scarce_level] = dict()
        y_error_std_dict[scarce_level] = dict()
        for noise_level in models[scarce_level].keys():
            seeds_models = models[scarce_level][noise_level]
            seeds_inputs_tensor = inputs[scarce_level][noise_level]
            y_list, y_error_list = [], [] # store the y prediction for each levels
            print(f"Evaluating model with scarce level: {scarce_level}, noise level: {noise_level}")
            for seed in seeds_models.keys():
                print(f"Evaluating model with seed: {seed}")
                model = seeds_models[seed]
                model.eval()
                y = model(seeds_inputs_tensor[seed])
                y = y.cpu().detach().numpy().ravel()[::10]  # align
                y_list.append(y)

                error = np.linalg.norm(y - ground_truth) / np.linalg.norm(ground_truth)
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
    }
    case_title = title_map.get(case, case)

    for scarce_level in y_plot_dict.keys():
        corrector_scarce_level_dir = os.path.join(corrector_dir, f"{scarce_level}")
        os.makedirs(corrector_scarce_level_dir, exist_ok=True)
        for noise_level, y_mean in y_plot_dict[scarce_level].items():
            noise_str = noise_level.split("_")[1]
            noise_percent = f"{float(noise_str) * 100:.0f}%"
            l2_error = y_error_plot_dict[scarce_level][noise_level]
            y_std = y_std_dict[scarce_level][noise_level]
            plt.figure(figsize=(8, 5))
            plt.plot(t, ground_truth, label="Reference", linewidth=2, color="dimgray")
            plt.plot(t, y_mean, label=f"{scarce_level} pts (L2: {l2_error:.2e})", linestyle="--", alpha=0.6, color="red")
            plt.fill_between(t, y_mean - 2*y_std, y_mean + 2*y_std,label=f'±2 Std Dev ({scarce_level} pts)', alpha=0.2, color='red')
            plt.xlabel("Time (t)")
            plt.ylabel("Displacement x(t)")
            plt.title(f"Corrector Prediction (mean) on {case_title} - Noise: {noise_percent}")
            plt.legend(loc="best", frameon=True)
            plt.savefig(os.path.join(corrector_scarce_level_dir, f"{case.lower()}-{noise_str}-{scarce_level}pts-finetune-scarce-level.png"), dpi=300)
    plt.figure(figsize=(8, 5))
    s_1, s_2, n = 30, 100, "noise_0.02"
    n_p = n.split("_")[1]
    n_p = f"{float(n_p)*100:.0f}%"
    plt.plot(t, ground_truth, label="Reference", linewidth=2, color="dimgray")
    plt.plot(t, y_plot_dict[s_1][n], label=f"30 pts (L2: {y_error_plot_dict[s_1][n]:.2e})", linestyle="--", alpha=0.6, color='red')
    plt.plot(t, y_plot_dict[s_2][n], label=f"100 pts (L2: {y_error_plot_dict[s_2][n]:.2e})", linestyle="--", alpha=0.6, color='blue')
    plt.fill_between(t, y_plot_dict[s_1][n] - 2*y_std_dict[s_1][n], y_plot_dict[s_1][n] + 2*y_std_dict[s_1][n], label=f'±2 Std Dev ({s_1} pts)', alpha=0.2, color='red')
    plt.fill_between(t, y_plot_dict[s_2][n] - 2*y_std_dict[s_2][n], y_plot_dict[s_2][n] + 2*y_std_dict[s_2][n], label=f'±2 Std Dev ({s_2} pts)', alpha=0.2, color='blue')
    plt.xlabel("Time (t)")
    plt.ylabel("Displacement x(t)")
    plt.title(f"Corrector Prediction (mean) on {case_title} - Noise: {n_p}")
    plt.legend(loc="best", frameon=True)
    plt.savefig(os.path.join(corrector_dir, f"comparison-{case.lower()}-corrector-scarce-noise-level.png"), dpi=300)
    
    # # --------------------------------
    # Plot mean L2 error ± std vs. scarce data level (with equal spacing on x-axis)   
    level_ints = sorted([int(lvl) for lvl in y_plot_dict.keys()])
    for scarce_level in level_ints:
        corrector_scarce_level_dir = os.path.join(corrector_dir, f"{scarce_level}")
        noise_level_mean_dict = y_error_plot_dict[scarce_level]
        noise_level_std_dict = y_error_std_dict[scarce_level]
        assert noise_level_mean_dict.keys() == noise_level_std_dict.keys()
        noise_levels = list(noise_level_mean_dict.keys())
        noise_str_list = [n_level.split("_")[1] for n_level in noise_levels]
        noise_percent = [f"{float(n_s) * 100:.0f}%" for n_s in noise_str_list]
        noise_percent_array = np.array(noise_percent)
        x_pos = np.arange(len(noise_percent_array))

        error_array = np.array([noise_level_mean_dict[level] for level in noise_levels])
        std_array = np.array([noise_level_std_dict[level] for level in noise_levels])

        plt.figure(figsize=(8, 5))
        plt.plot(x_pos, error_array, marker='o', color='blue', label=f"{scarce_level} pts L2 Error (mean)", zorder=2)
        plt.fill_between(
            x_pos,
            error_array - 2 * std_array,
            error_array + 2 * std_array,
            color='blue',
            alpha=0.15,
            label='±2 Std Dev',
            zorder=0
        )
        plt.xticks(ticks=x_pos, labels=noise_percent_array)
        plt.xlabel("Noise Level")
        plt.ylabel("Relative L2 Error")
        plt.title(f"Corrector L2 Error vs. Noise Level ({case_title})")
        plt.legend()
        plt.tight_layout()

        error_plot_path = os.path.join(corrector_scarce_level_dir, f"{scarce_level}pts-{case.lower()}-seeds-l2error-mean-std.png")
        plt.savefig(error_plot_path, dpi=300)
        plt.close()
        print(f"L2 error plot with error bars saved to {error_plot_path}")

    s_1, s_2 = 30, 100
    scarce_levels = [s_1, s_2]

    plt.figure(figsize=(8, 5))

    for scarce_level, color in zip(scarce_levels, ['blue', 'red']):
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
    SUBFOLDERS = [30, 100]
    flags.DEFINE_string("workdir", "."+os.sep+"results_sys", "Directory to store model data.")
    config_flags.DEFINE_config_file(
        "config",
        "examples/qdho/configs/default.py",
        "File path to the training hyperparameter configuration.",
        lock_config=True,
    )
    
    config = FLAGS.config

    noise_models, SUBFOLDERS = load_seeds_models(config, FLAGS.workdir, case="QDHO", target="noise_seeds", SUBFOLDERS=SUBFOLDERS)
    evaluate_finetune_by_seeds(config, FLAGS.workdir, noise_models, case="QDHO")
    noise_correctors, noise_correctors_inputs, SUBFOLDERS = load_seeds_correctors(config, FLAGS.workdir, case="QDHO", target="noise_seeds", SUBFOLDERS=SUBFOLDERS)
    evaluate_corrector_by_seeds(config, FLAGS.workdir, noise_correctors, noise_correctors_inputs, case="QDHO")
    
if __name__ == "__main__":
    app.run(main)


