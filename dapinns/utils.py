import os
import torch

def save_checkpoint(state, save_dir, epoch, keep, verbose=True, name=None):
    """
    Save the model checkpoint and manage old checkpoints.

    Args:
        state (dict): The state to save (e.g., model and optimizer states).
        save_dir (str): Directory to save the checkpoint.
        epoch (int): Current epoch number.
        keep (int): Number of checkpoints to keep.
        name (str): Optional custom checkpoint name.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Define the checkpoint file name
    if name is None:
        name = f"checkpoint_epoch_{epoch}.pt"
    checkpoint_path = os.path.join(save_dir, name)

    # Save the checkpoint
    torch.save(state, checkpoint_path)
    if verbose:
        print(f"Checkpoint saved to {checkpoint_path}")

    # Manage old checkpoints
    checkpoint_files = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    checkpoint_files.sort(key=lambda f: os.path.getmtime(os.path.join(save_dir, f)), reverse=True)

    # Remove old checkpoints if the number exceeds 'keep'
    for old_checkpoint in checkpoint_files[keep:]:
        old_checkpoint_path = os.path.join(save_dir, old_checkpoint)
        os.remove(old_checkpoint_path)
        print(f"Deleted old checkpoint {old_checkpoint_path}")