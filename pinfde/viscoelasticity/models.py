import torch
import torch.nn as nn
from math import gamma

class PiNFDE_Model(nn.Module):
    def __init__(self, config, input_dim=2, hidden_dim=64, output_dim=1):
        super().__init__()
        self.config = config
        self.device = config.device
        
        # 可學習的分數階階數 alpha (Inverse Problem)
        self.alpha = nn.Parameter(torch.tensor([0.5], dtype=torch.float32)) 
        
        # NN 部分：學習未知的非線性動力學
        self.nn_part = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def f_phy(self, x):
        """已知物理項 (通常是系統的線性部分)"""
        D = 0.1 # 假設瞬時擴散係數已知
        # 這裡需要根據具體系統定義，例如 D * u_xx
        return -D * x

    def adams_solver(self, x0, t_steps):
        """Adams 預測-校正法數值求解器"""
        h = t_steps[1] - t_steps[0]
        alpha = torch.clamp(self.alpha, 0.01, 0.99) # 限制在 (0,1)
        n_steps = len(t_steps)
        x = torch.zeros((n_steps, x0.shape[0])).to(self.device)
        x[0] = x0
        
        # 存儲 NN + f_phy 的歷史值以供積分
        history_f = []

        for l in range(n_steps - 1):
            curr_t = t_steps[l].unsqueeze(0)
            # 組合導函數項: d^alpha/dt^alpha x = NN + f_phy
            f_val = self.nn_part(torch.cat([curr_t.expand(x[l].shape[0], 1), x[l].unsqueeze(1)], dim=1))
            f_val = f_val.squeeze() + self.f_phy(x[l])
            history_f.append(f_val)
            
            # 預測步 (Predictor)
            b = torch.tensor([( (l+1-j)**alpha - (l-j)**alpha ) for j in range(l+1)]).to(self.device)
            x_p = x0 + (h**alpha / gamma(alpha + 1)) * torch.sum(b.view(-1, 1) * torch.stack(history_f), dim=0)
            
            # 校正步 (Corrector)
            # 計算 a_j,l+1 係數 (此處簡化實作，建議依論文 3.4 公式細節補足)
            # x[l+1] = x0 + ... (預測校正邏輯)
            x[l+1] = x_p # 簡化演示，實作時應加入 Corrector 項
            
        return x

    def forward(self, x0, t_steps):
        return self.adams_solver(x0, t_steps)