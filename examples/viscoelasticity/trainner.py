import os
import time
import wandb
import ml_collections
import torch

from .models import MemoryDiffusionPINN, Corrector
from .utils import generate_memory_diffusion_dataset, generate_pure_diffusion_dataset
from dapinns.samplers import RandomSampler
from dapinns.utils import save_checkpoint


# =====================================================
# Pretraining
# =====================================================
def pretrain(config: ml_collections.ConfigDict, workdir: str):
    wandb.init(
        project=config.wandb.project,
        name=config.wandb.name,
        config=dict(config.wandb)
    )

    params = config.system_memory.system_params
    X, y = generate_pure_diffusion_dataset(
        params,
        nx=params["nx"],
        nt=params["nt"]
    )

    sampler = RandomSampler(config, sample_size=config.pretrained_sample_size)
    X_train, y_train = sampler.generate_data(X, y)
    X_train, y_train = X_train.to(config.device), y_train.to(config.device)

    model = MemoryDiffusionPINN(config).to(config.device)
    model.train()

    max_epochs = config.pretraining.max_epochs
    u_w, f_w = config.pretraining.u_w, config.pretraining.f_w

    for epoch in range(max_epochs):
        model.optimizer.zero_grad()
        u_loss = model.u_loss(X_train, y_train)
        f_loss, _ = model.f_loss()
        loss = u_w * u_loss + f_w * f_loss
        loss.backward()
        model.optimizer.step()

        if (epoch + 1) % 100 == 0:
            wandb.log({"u_loss": u_loss.item(), "f_loss": f_loss.item(), "loss": loss.item()})

        if loss.item() < config.saving.epsilon:
            break

        if (epoch + 1) % config.saving.save_interval == 0:
            save_dir = os.path.join(workdir, config.saving.save_dir, config.saving.pretrain_path)
            os.makedirs(save_dir, exist_ok=True)
            save_checkpoint(
                {"epoch": epoch + 1, "model_state_dict": model.state_dict(), "loss": loss.item()},
                save_dir, epoch + 1, keep=config.saving.keep
            )


# =====================================================
# Finetuning (DAPINNs-style with LBFGS)
# =====================================================
def finetune(config: ml_collections.ConfigDict, workdir: str):
    wandb.init(
        project=config.wandb.project,
        name=config.wandb.name,
        config=dict(config.wandb)
    )

    params = config.system_memory.system_params
    X, y, _, _ = generate_memory_diffusion_dataset(
        params, nx=params["nx"], nt=params["nt"], noise=params.get("noise", 0.0)
    )

    sampler = RandomSampler(config, sample_size=config.finetune_sample_size)
    X_train, y_train = sampler.generate_data(X, y)
    X_train, y_train = X_train.to(config.device), y_train.to(config.device)

    model = MemoryDiffusionPINN(config).to(config.device)
    corrector = Corrector(config).to(config.device)

    pretrained_path = os.path.join(
        workdir, config.saving.save_dir, config.saving.pretrain_path, config.pretrained_model_name
    )
    model.load_pretrained_model(pretrained_path)

    # 設定儲存路徑
    save_dir = os.path.join(workdir, config.saving.save_dir)
    fine_tune_save_dir = os.path.join(save_dir, config.saving.finetune_path)
    corrector_save_dir = os.path.join(save_dir, config.saving.corrector_path)
    os.makedirs(fine_tune_save_dir, exist_ok=True)
    os.makedirs(corrector_save_dir, exist_ok=True)

    max_epochs = config.finetuning.max_epochs
    alt_steps = config.finetuning.alt_steps
    u_w, f_w = config.finetuning.u_w, config.finetuning.f_w
    
    best_total_loss = float("inf")
    best_epoch = None
    warmup_epochs = 2000 # 避免一開始就存 best

    print("Start Adam Alternating Phase...")
    for epoch in range(max_epochs):
        if ((epoch // alt_steps) % 2) == 0:
            model.optimizer.zero_grad()
            u_loss = model.u_loss(X_train, y_train)
            f_loss, correction_inputs = model.f_loss(corrector)
            loss = u_w * u_loss + f_w * f_loss
            loss.backward()
            model.optimizer.step()
        else:
            corrector.optimizer.zero_grad()
            f_loss, correction_inputs = model.f_loss(corrector)
            with torch.no_grad():
                u_loss = model.u_loss(X_train, y_train)
            f_loss.backward()
            corrector.optimizer.step()
            loss = u_w * u_loss + f_w * f_loss

        if (epoch + 1) % 100 == 0:
            wandb.log({"u_loss": u_loss.item(), "f_loss": f_loss.item(), "loss": loss.item()})

        # 追蹤最佳模型
        if (epoch + 1) > warmup_epochs and loss.item() < best_total_loss:
            best_total_loss = loss.item()
            best_epoch = epoch + 1
            save_checkpoint({"model_state_dict": model.state_dict()}, fine_tune_save_dir, epoch + 1, name="best_total_loss_model.pt", keep=1, verbose=False)
            save_checkpoint({"model_state_dict": corrector.state_dict(), "corrector_inputs": correction_inputs}, corrector_save_dir, epoch + 1, name="best_total_loss_corrector.pt", keep=1, verbose=False)

        if (epoch + 1) % config.saving.save_interval == 0:
            save_checkpoint({"epoch": epoch + 1, "model_state_dict": model.state_dict()}, fine_tune_save_dir, epoch + 1, keep=config.saving.keep)
            save_checkpoint({"epoch": epoch + 1, "model_state_dict": corrector.state_dict(), "corrector_inputs": correction_inputs}, corrector_save_dir, epoch + 1, keep=config.saving.keep)

    # -------------------------------------------------
    # LBFGS 最終優化階段
    # -------------------------------------------------
    print("\nStarting LBFGS final optimization...")
    best_model_path = os.path.join(fine_tune_save_dir, "best_total_loss_model.pt")
    best_corrector_path = os.path.join(corrector_save_dir, "best_total_loss_corrector.pt")
    
    model.load_state_dict(torch.load(best_model_path, map_location=config.device)["model_state_dict"])
    corrector.load_state_dict(torch.load(best_corrector_path, map_location=config.device)["model_state_dict"])

    lbfgs_optimizer = torch.optim.LBFGS(
        list(model.parameters()) + list(corrector.parameters()),
        max_iter=500,
        tolerance_grad=1e-5,
        tolerance_change=1e-7
    )

    def closure():
        lbfgs_optimizer.zero_grad()
        u_l = model.u_loss(X_train, y_train)
        f_l, _ = model.f_loss(corrector)
        total_l = u_w * u_l + f_w * f_l
        total_l.backward()
        return total_l

    lbfgs_optimizer.step(closure)

    # 存檔供 eval.py 使用
    print("Saving final LBFGS results...")
    save_checkpoint({"model_state_dict": model.state_dict()},
                    fine_tune_save_dir,
                    epoch + 1,
                    name=config.finetuned_model_name,
                    keep=config.saving.keep)
    save_checkpoint({"model_state_dict": corrector.state_dict(), 
                     "corrector_inputs": correction_inputs},
                     corrector_save_dir, epoch + 1, 
                     name=config.corrector_model_name,
                     keep=config.saving.keep)


def train(config: ml_collections.ConfigDict, workdir: str):
    if config.is_pretrained:
        pretrain(config, workdir)
    else:
        finetune(config, workdir)