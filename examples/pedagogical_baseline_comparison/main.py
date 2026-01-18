import ml_collections
import os
from absl import app
from absl import flags
from ml_collections import config_flags

from examples.pedagogical_baseline_comparison import trainner, eval, sr

FLAGS = flags.FLAGS

# ------------------------------------------------------------
# Command-line flags
# ------------------------------------------------------------
# Config file
config_flags.DEFINE_config_file(
    "config",
    "examples/pedagogical_baseline_comparison/configs/default.py",
    "Training configuration file.",
    lock_config=False,
)

# General
flags.DEFINE_string("workdir", ".", "Root directory for working.")
flags.DEFINE_string("save_subdir", "results_default", "Sub-directory for saving results (e.g. results_dapinn).")
flags.DEFINE_enum("mode", "train", ["train", "eval"], "Mode: train or eval")
flags.DEFINE_integer("seed", 42, "Random seed")

# System Params Override
flags.DEFINE_integer("sample_size", None, "Number of measurement samples")
flags.DEFINE_float("noise", None, "Noise level")

# Experiment Logic Control
flags.DEFINE_bool("use_corrector", False, "Use DA-PINN corrector (True) or vanilla PINN (False)")
flags.DEFINE_bool("run_pretrain", False, "Execute pre-training stage")
flags.DEFINE_bool("run_finetune", True, "Execute fine-tuning stage")
flags.DEFINE_bool("load_pretrained", False, "Load pre-trained weights before fine-tuning")


# ------------------------------------------------------------
# WandB Config Helper
# ------------------------------------------------------------
def setup_wandb_config(config):
    tag_list = []
    if config.run_pretrain: tag_list.append("pretrain")
    if config.run_finetune: tag_list.append("finetune")
    
    return ml_collections.ConfigDict({
        "project": "DAPINNs-Pedagogical-Baseline-Comparison",
        "name": f"{config.name}_{'DAPINN' if config.use_corrector else 'PINN'}",
        "mode": config.mode,
        "sample_size": config.sample_size,
        "tags": tag_list,
        "use_corrector": config.use_corrector
    })


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(argv):
    # Load config from file
    config = FLAGS.config

    # -------------------------------
    # Inject Flag Values into Config
    # -------------------------------
    # This maps the command line flags to the config object
    # so that trainer.py can access them via config.xxx
    
    config.mode = FLAGS.mode
    config.workdir = FLAGS.workdir
    config.seed = FLAGS.seed
    config.use_corrector = FLAGS.use_corrector
    
    # Flow control
    config.run_pretrain = FLAGS.run_pretrain
    config.run_finetune = FLAGS.run_finetune
    config.load_pretrained = FLAGS.load_pretrained

    # Optional overrides
    if FLAGS.sample_size is not None:
        config.sample_size = FLAGS.sample_size
    if FLAGS.noise is not None:
        config.noise = FLAGS.noise

    # -------------------------------
    # Handle Save Paths
    # -------------------------------
    # Construct the full save path: workdir/save_subdir
    # e.g., ./results_baseline
    full_save_path = os.path.join(FLAGS.workdir, FLAGS.save_subdir)
    config.saving.save_dir = full_save_path
    
    # Ensure root exists
    os.makedirs(full_save_path, exist_ok=True)

    # -------------------------------
    # WandB & Execution
    # -------------------------------
    config.wandb = setup_wandb_config(config)

    print(f"========================================")
    print(f" CONFIGURATION")
    print(f" Mode: {config.mode}")
    print(f" Device: {config.device}")
    print(f" Corrector: {config.use_corrector}")
    print(f" Pretrain: {config.run_pretrain}")
    print(f" Finetune: {config.run_finetune}")
    print(f" Save Dir: {config.saving.save_dir}")
    print(f"========================================")

    if config.mode == "train":
        trainner.train(config, FLAGS.workdir)

    elif config.mode == "eval":
        eval.evaluate(config, FLAGS.workdir)
        sr.execute_sr(config, FLAGS.workdir)
        # sr.generate_discrepancy_table(config, FLAGS.workdir, sample_sizes=[30, 50, 100, 200, 500, 1000], num_trials=10)


if __name__ == "__main__":
    app.run(main)