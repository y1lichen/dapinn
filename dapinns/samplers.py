import numpy as np
import torch

# For ODEs (e.g., simple, damped harmonic oscillator)
class UniformSampler:
    def __init__(self, sample_size):
        self.sample_size = sample_size

    def generate_data(self, time, values):
        indices = torch.linspace(0, len(values)-1, self.sample_size).int()
        return time[indices], values[indices]

class RandomSampler:
    def __init__(self, config, sample_size):
        self.sample_size = sample_size
        self.seed = config.seed

    def generate_data(self, time, values, return_indices=False):
        torch.manual_seed(self.seed)
        indices = torch.randperm(len(values))[:self.sample_size]
        if return_indices:
            return time[indices], values[indices], indices
        else:
            # Return sampled time and values
            return time[indices], values[indices]
    
# For PDEs
class TimeSpaceUniformSampler:
    def __init__(self, sample_size):
        self.sample_size = sample_size
    
    def generate_data(self, x, t, u):
        """
        Inputs:
            x: [n_x, 1] spatial points
            t: [n_t, 1] temporal points
            u: [n_x, n_t] solution values
        Returns:
            xyt: [sample_size, 2] tensor of sampled (x, t) pairs
            u_sample: [sample_size, 1] corresponding solution values
        """
        n_x, n_t = x.shape[0], t.shape[0]

        X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
        X_flat = X.reshape(-1, 1)
        T_flat = T.reshape(-1, 1)
        U_flat = u.reshape(-1, 1)

        # Uniformly sample indices
        total_points = n_x * n_t
        indices = torch.linspace(0, total_points - 1, steps=self.sample_size).long()

        xyt_sample = torch.cat([X_flat[indices], T_flat[indices]], dim=1)
        u_sample = U_flat[indices]

        return xyt_sample, u_sample
    

class TimeSpaceRandomSampler:
    def __init__(self, config, sample_size: int):
        """
        Randomly samples (x, t) pairs from the spatiotemporal grid.
        """
        self.sample_size = sample_size
        self.seed = config.seed

    def generate_data(self, x: torch.Tensor, t: torch.Tensor, u: torch.Tensor):
        """
        Inputs:
            x: [n_x, 1] spatial points
            t: [n_t, 1] temporal points
            u: [n_x, n_t] solution values
        Returns:
            xyt: [sample_size, 2] tensor of sampled (x, t) pairs
            u_sample: [sample_size, 1] corresponding solution values
        """
        assert x.dim() == 2 and t.dim() == 2, "Expected x and t to be of shape [N, 1]"

        n_x, n_t = x.shape[0], t.shape[0]

        # Create full grid and flatten
        X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing="ij")
        X_flat = X.reshape(-1, 1)
        T_flat = T.reshape(-1, 1)
        U_flat = u.reshape(-1, 1)

        # Randomly sample indices
        total_points = n_x * n_t
        torch.manual_seed(self.seed)
        indices = torch.randperm(total_points)[:self.sample_size]

        xyt_sample = torch.cat([X_flat[indices], T_flat[indices]], dim=1)
        u_sample = U_flat[indices]

        return xyt_sample, u_sample