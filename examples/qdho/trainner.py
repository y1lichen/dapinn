import os
import time
import wandb
import ml_collections
import torch

from .models import Qdho, Corrector
from .utils import generate_uho_dataset, generate_qdho_dataset
from .eval import evaluate_pretrained_pinns

from torch.utils.data import DataLoader, TensorDataset
from dapinns.samplers import UniformSampler, RandomSampler
from dapinns.utils import save_checkpoint

def pretrain(config: ml_collections.ConfigDict, workdir:str):
    
    # Initialize wandb
    wandb_config = config.wandb
    wandb.init(
        config=dict(wandb_config),
        project=wandb_config.project,
        name=wandb_config.name,
        tags=[wandb_config.tag]  # optional but useful
    )

    # System parameters setting
    params = config.system_uho.system_params
    T = params['T']
    x0 = params['x0']
    v0 = params['v0']
    n_t = params['n_t']

    x, y, _ = generate_uho_dataset(params, T=T, x0=x0, v0=v0, n_t=n_t)

    # sample
    sampler = UniformSampler(sample_size=config.pretrained_sample_size)
    x_train, y_train = sampler.generate_data(x, y)
    x_train = x_train.to(config.device)
    y_train = y_train.to(config.device)

    # model
    model = Qdho(config).to(config.device)
    model.train()

    # load arguments from config
    max_epochs = config.pretraining.max_epochs
    u_w = config.pretraining.u_w
    f_w = config.pretraining.f_w

    print("Start pretraining...")
    for epoch in range(max_epochs):

        model.optimizer.zero_grad()

        # training
        u_loss = model.u_loss(x_train, y_train)
        f_loss = model.f_loss()
        total_loss = u_w * u_loss + f_w * f_loss
        total_loss.backward()
        model.optimizer.step()

        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'loss': total_loss.item(),
        }

        # wandb log
        wandb.log({
            "u_loss": u_loss.item(),
            "f_loss": f_loss.item(),
            "total_loss": total_loss.item()
        })

        if (epoch + 1) % 6000 == 0:
            model.scheduler.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch: {epoch + 1}, u_loss: {u_loss.item()}, f_loss: {f_loss.item()}, Loss: {total_loss.item()}")

        # Save checkpoint
        if (epoch + 1) % config.saving.save_interval == 0:
            
            save_dir = os.path.join(workdir, config.saving.save_dir)  # Use save_dir from config
            save_dir = os.path.join(save_dir, config.saving.pretrain_path)
            os.makedirs(save_dir, exist_ok=True)

            save_checkpoint(state, save_dir, epoch + 1, keep=config.saving.keep)
        
        # stopping criteria
        if total_loss.item() < config.saving.epsilon:
            print(f"Stopping criteria met at epoch {epoch + 1}.")
            break

    save_checkpoint(state, save_dir, epoch + 1, keep=config.saving.keep)
    evaluate_pretrained_pinns(config, workdir)

def finetune(config: ml_collections.ConfigDict, workdir: str):
    
    # Initialize wandb
    wandb_config = config.wandb
    wandb.init(
        config=dict(wandb_config),
        project=wandb_config.project,
        name=wandb_config.name,
        tags=[wandb_config.tag]
    )

    # System parameters
    params = config.system_qdho.system_params
    T, x0, v0, n_t = params['T'], params['x0'], params['v0'], params['n_t']

    if "noise" in config.system_qdho.system_params:
        noise = config.system_qdho.system_params['noise']
        params['noise'] = noise
        print(f"Noise level: {noise}")
    else:
        print("No noise level specified.")

    # Dataset
    x, y, _ = generate_qdho_dataset(params, T=T, x0=x0, v0=v0, n_t=n_t)
    sampler = RandomSampler(config, sample_size=config.finetune_sample_size)
    x_train, y_train = sampler.generate_data(x, y)
    x_train, y_train = x_train.to(config.device), y_train.to(config.device)

    # Model
    model = Qdho(config).to(config.device)
    # pretrained_dir = os.path.join(workdir, config.saving.save_dir, config.saving.pretrain_path, config.pretrained_model_name)
    # model.load_pretrained_model(pretrained_dir)

    corrector = Corrector(config).to(config.device)
    model.train()
    corrector.train()

    print("Start finetuning...\n" + "-" * 80)

    # Config
    max_epochs = config.finetuning.max_epochs
    alt_steps = config.finetuning.alt_steps
    u_w, f_w = config.finetuning.u_w, config.finetuning.f_w
    save_interval = config.saving.save_interval
    epsilon = config.saving.epsilon
    keep = config.saving.keep

    # Save dirs
    save_dir = os.path.join(workdir, config.saving.save_dir)
    fine_tune_save_dir = os.path.join(save_dir, config.saving.finetune_path)
    corrector_save_dir = os.path.join(save_dir, config.saving.corrector_path)
    os.makedirs(fine_tune_save_dir, exist_ok=True)
    os.makedirs(corrector_save_dir, exist_ok=True)

    # Track global best
    best_val_f_loss = float("inf")
    best_total_loss = float("inf")
    best_epoch = None
    best_model_state = None
    best_corrector_state = None
    warmup_epochs = 1000

    for epoch in range(max_epochs):
        if alt_steps != 0:
            if ((epoch + 1) // alt_steps) % 2 == 0:
                model.optimizer.zero_grad()
                u_loss = model.u_loss(x_train, y_train)
                f_loss, correction_inputs = model.f_loss(corrector)
                total_loss = u_w * u_loss + f_w * f_loss
                total_loss.backward()
                model.optimizer.step()
            else:
                corrector.optimizer.zero_grad()
                f_loss, correction_inputs = model.f_loss(corrector)
                f_loss.backward()
                corrector.optimizer.step()
        else: # no alternating updates
            model.optimizer.zero_grad()
            corrector.optimizer.zero_grad()
            u_loss = model.u_loss(x_train, y_train)
            f_loss, correction_inputs = model.f_loss(corrector)
            total_loss = u_w * u_loss + f_w * f_loss
            total_loss.backward()
            model.optimizer.step()
            corrector.optimizer.step() # using total loss

        val_f_loss = model.val_f_loss(corrector)
        # wandb log
        wandb.log({
            "u_loss": u_loss.item(),
            "f_loss (train)": f_loss.item(),
            "f_loss (val)": val_f_loss.item(),
            "total_loss": total_loss.item()
        })

        if (epoch + 1) % 1000 == 0:
            print(f"Learning rate: {model.scheduler.get_last_lr()[0]:.4}, Epoch: {epoch + 1}, u_loss: {u_w*u_loss.item():.4e}, f_loss: {f_w*f_loss.item():.4e}, Loss: {total_loss.item():.4e}, Val_loss: {f_w*val_f_loss.item():.4e}, Best epoch: {best_epoch}")

        model_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'loss': total_loss.item(),
        }
        corrector_state = {
            'epoch': epoch + 1,
            'corrector_inputs': correction_inputs,
            'model_state_dict': corrector.state_dict(),
            'optimizer_state_dict': corrector.optimizer.state_dict(),
            'loss': total_loss.item(),
        }

        # monitor val f loss
        model.scheduler.step(val_f_loss.item())
        corrector.scheduler.step(val_f_loss.item())

        is_best = (epoch + 1) > warmup_epochs and total_loss.item() < best_total_loss
        if is_best:
            best_total_loss = total_loss.item()
            best_epoch = epoch + 1
            save_checkpoint(model_state, fine_tune_save_dir, epoch + 1, keep=keep, verbose=False, name="best_total_loss_model.pt")
            save_checkpoint(corrector_state, corrector_save_dir, epoch + 1, keep=keep, verbose=False, name="best_total_loss_corrector.pt")

        # Periodic save
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(model_state, fine_tune_save_dir, epoch + 1, keep=keep)
            save_checkpoint(corrector_state, corrector_save_dir, epoch + 1, keep=keep)
        # Early stopping and LR decay
        if best_epoch is not None and config.saving.early_stopping_patience:
 
            if (epoch + 1 - best_epoch) >= config.saving.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break


    # Final LBFGS optimization step
    best_f_model_dir = os.path.join(fine_tune_save_dir, "best_total_loss_model.pt")
    best_f_corrector_dir = os.path.join(corrector_save_dir)
    model.load_finetuned_model(best_f_model_dir)
    corrector.load_corrector_model(best_f_corrector_dir, "best_total_loss_corrector.pt")
    print("\nStarting LBFGS final optimization...")
    lbfgs_params = list(model.parameters()) + list(corrector.parameters())
    lbfgs_optimizer = torch.optim.LBFGS(lbfgs_params, max_iter=50000, tolerance_grad=1e-7, tolerance_change=1e-9)
    # lbfgs_optimizer = torch.optim.LBFGS(lbfgs_params, max_iter=10, tolerance_grad=1e-5, tolerance_change=1e-7)

    def closure():
        lbfgs_optimizer.zero_grad()
        u_loss = model.u_loss(x_train, y_train)
        f_loss, _ = model.f_loss(corrector)
        loss = u_w * u_loss + f_w * f_loss
        loss.backward()
        return loss

    lbfgs_optimizer.step(closure)

    # Save after LBFGS
    print("Saving after LBFGS...")
    final_model_state = {
        'epoch': epoch + 1,  # Use the latest epoch value
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'loss': u_w * u_loss + f_w * f_loss,  # Update the loss to the final value
    }
    final_corrector_state = {
        'epoch': epoch + 1,  # Use the latest epoch value
        'corrector_inputs': correction_inputs,
        'model_state_dict': corrector.state_dict(),
        'optimizer_state_dict': corrector.optimizer.state_dict(),
        'loss': u_w * u_loss + f_w * f_loss,  # Update the loss to the final value
    }

    # Save the final states after LBFGS optimization
    save_checkpoint(final_model_state, fine_tune_save_dir, epoch + 1, keep=keep, name="lbfgs_finetuned_model.pt")
    save_checkpoint(final_corrector_state, corrector_save_dir, epoch + 1, keep=keep, name="lbfgs_finetuned_corrector.pt")

    # # Track global best
    # best_f_loss = float("inf")
    # best_total_loss = float("inf")
    # best_model_state = None
    # best_corrector_state = None

    # for epoch in range(max_epochs):

    #     if (epoch // alt_steps) % 2 == 0:
    #         model.optimizer.zero_grad()
    #         u_loss = model.u_loss(x_train, y_train)
    #         f_loss, correction_inputs = model.f_loss(corrector)
    #         total_loss = u_w * u_loss + f_w * f_loss
    #         total_loss.backward()
    #         model.optimizer.step()
    #     else:
    #         corrector.optimizer.zero_grad()
    #         f_loss, correction_inputs = model.f_loss(corrector)
    #         u_loss = model.u_loss(x_train, y_train)  # for wandb/logging
    #         f_loss.backward()
    #         corrector.optimizer.step()
    #         total_loss = u_w * u_loss + f_w * f_loss # for recording

    #     if (epoch + 1) % 16000 == 0:
    #         model.scheduler.step()
    #         corrector.scheduler.step()

    #     # Log to wandb
    #     wandb.log({
    #         "u_loss": u_loss.item(),
    #         "f_loss": f_loss.item(),
    #         "total_loss": total_loss.item()
    #     })

    #     if (epoch + 1) % 1000 == 0:
    #         print(f"Epoch: {epoch + 1}, u_loss: {u_w*u_loss.item():.4e}, f_loss: {f_w*f_loss.item():.4e}, Loss: {total_loss.item():.4e}")

    #     # Track best f_loss
    #     if total_loss.item() < best_total_loss:
    #         best_total_loss = total_loss.item()
    #         best_model_state = {
    #             'epoch': epoch + 1,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': model.optimizer.state_dict(),
    #             'loss': total_loss.item(),
    #         }
    #         best_corrector_state = {
    #             'epoch': epoch + 1,
    #             'corrector_inputs': correction_inputs,
    #             'model_state_dict': corrector.state_dict(),
    #             'optimizer_state_dict': corrector.optimizer.state_dict(),
    #             'loss': total_loss.item(),
    #         }

    #     # Periodic save
    #     if (epoch + 1) % save_interval == 0:
    #         model_state = {
    #             'epoch': epoch + 1,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': model.optimizer.state_dict(),
    #             'loss': total_loss.item(),
    #         }
    #         corrector_state = {
    #             'epoch': epoch + 1,
    #             'corrector_inputs': correction_inputs,
    #             'model_state_dict': corrector.state_dict(),
    #             'optimizer_state_dict': corrector.optimizer.state_dict(),
    #             'loss': total_loss.item(),
    #         }
    #         save_checkpoint(model_state, fine_tune_save_dir, epoch + 1, keep=keep)
    #         save_checkpoint(corrector_state, corrector_save_dir, epoch + 1, keep=keep)

    #     # Early stopping
    #     if total_loss.item() < epsilon:
    #         print(f"Stopping criteria met at epoch {epoch + 1}.")
    #         break
        
    #     early_stop_patience = getattr(config.saving, "early_stop_patience", None)
    #     if early_stop_patience and  (epoch + 1 - best_model_state['epoch']) > early_stop_patience:
    #         print(f"Early stopping at epoch {epoch + 1}.")
    #         break        

    # # Save best f_loss model at the end
    # save_checkpoint(best_model_state, fine_tune_save_dir, epoch + 1, keep=keep, name="best_f_loss_model.pt")
    # save_checkpoint(best_corrector_state, corrector_save_dir, epoch + 1, keep=keep, name="best_f_loss_corrector.pt")

def train(config: ml_collections.ConfigDict, workdir:str):
    if config.is_pretrained:
        pretrain(config, workdir)
    else:
        finetune(config, workdir)
