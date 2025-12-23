import numpy as np
import torch
from scipy.stats import qmc


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
    
class LHSSampler:
    def __init__(self, config=None, sample_size=1000):
        self.sample_size = sample_size
        self.seed = getattr(config, "seed", 42) if config else 42

    def generate_data_1d(self, t_range, device="cpu"):
        """
        針對 ODE (1D) 的 LHS 抽樣
        t_range: (t_min, t_max) 例如 (0, 1)
        """
        sampler = qmc.LatinHypercube(d=1, seed=self.seed)
        sample = sampler.random(n=self.sample_size) # 產生 [0, 1) 之間的樣本
        
        t_min, t_max = t_range
        t_lhs = qmc.scale(sample, t_min, t_max)
        
        return torch.tensor(t_lhs, dtype=torch.float32).to(device)

    def generate_data_2d(self, x_range, t_range, device="cpu"):
        """
        針對 PDE (2D: Space-Time) 的 LHS 抽樣
        x_range: (x_min, x_max)
        t_range: (t_min, t_max)
        """
        sampler = qmc.LatinHypercube(d=2, seed=self.seed)
        sample = sampler.random(n=self.sample_size)
        
        # 定義邊界並縮放
        l_bounds = [x_range[0], t_range[0]]
        u_bounds = [x_range[1], t_range[1]]
        xt_lhs = qmc.scale(sample, l_bounds, u_bounds)
        
        # 返回 [N, 2] 的 tensor, 第一欄是 x, 第二欄是 t
        return torch.tensor(xt_lhs, dtype=torch.float32).to(device)