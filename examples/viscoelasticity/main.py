import ml_collections
from absl import app
from absl import flags
from ml_collections import config_flags
# from examples.viscoelasticity import sr
from examples.viscoelasticity import trainner, eval

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", ".", "Directory to store model data.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Mode: train or eval (Overrides config)")
flags.DEFINE_bool("is_pretrained", None, "Set to True for pretraining, False for finetuning (Overrides config)")
flags.DEFINE_integer("finetune_sample_size", None, "Sample size to use during finetuning (Overrides config)")
flags.DEFINE_float("noise", None, "Noise level in training data (Overrides config)")
flags.DEFINE_integer("seed", None, "seed to use generate data (Overrides config)")

def setup_wandb_config(config):
    return ml_collections.ConfigDict({
        "project": "DAPINNs-viscoelasticity",
        "name": config.name,
        "tag": "pretrain" if config.is_pretrained else "finetune",
        "sample_size": config.pretrained_sample_size if config.is_pretrained else config.finetune_sample_size,
        "lr": config.pretrain_optim.lr if config.is_pretrained else config.finetune_pinns_optim.lr,
        "u_w": config.pretraining.u_w if config.is_pretrained else config.finetuning.u_w,
        "f_w": config.pretraining.f_w if config.is_pretrained else config.finetuning.f_w,
        "scheduler": config.pretrain_optim.scheduler if config.is_pretrained else config.finetune_pinns_optim.scheduler,
        "alt_steps": None if config.is_pretrained else config.finetuning.alt_steps,
    })

def main(argv):
    if FLAGS.noise is None:
        config_name = "default.py"
    else:
        config_name = "default_noise.py"
    config_flags.DEFINE_config_file(
        "config",
        f"examples/viscoelasticity/configs/{config_name}", # default.py, default_noise.py, fourier_emb.py
        "File path to the training hyperparameter configuration.",
        lock_config=False,
    )
    config = FLAGS.config
    if FLAGS.mode is not None:
        config.mode = FLAGS.mode
    if FLAGS.is_pretrained is not None:
        config.is_pretrained = FLAGS.is_pretrained
    if FLAGS.finetune_sample_size is not None:
        config.finetune_sample_size = FLAGS.finetune_sample_size
    if FLAGS.noise is not None:
        config.system_dho.system_params["noise"] = flags.FLAGS.noise
    if FLAGS.seed is not None:
        config.seed = FLAGS.seed

    config.wandb = setup_wandb_config(config)
    
    if config.mode == "train":
        trainner.train(config, FLAGS.workdir)
    elif config.mode == "eval":
        eval.evaluate(config, FLAGS.workdir)
        # sr.execute_sr(config, FLAGS.workdir)
        
if __name__ == "__main__":
    app.run(main)