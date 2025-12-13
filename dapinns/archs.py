import torch
import numpy as np
import torch.nn as nn

def xavier_init(model):

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    model.apply(init_weights)

class FourierEmbs(nn.Module):
    def __init__(self, input_dim, embed_dim, embed_scale=1.0):
        super(FourierEmbs, self).__init__()
        self.embed_scale = embed_scale
        self.embed_dim = embed_dim
        self.input_dim = input_dim

        # The kernel is a learnable parameter
        self.kernel = nn.Parameter(
            torch.randn(input_dim, embed_dim // 2) * embed_scale
        )

    def forward(self, x):
        # x: shape (batch_size, input_dim)
        proj = x @ self.kernel  # (batch_size, embed_dim // 2)
        emb = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return emb

class Mlp(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, with_fourier=False, fourier_emb=None, activation=nn.Tanh):
        super().__init__()

        self.with_fourier = with_fourier
        self.fourier_emb = fourier_emb
        
        if with_fourier and fourier_emb is not None:
            self.fourier_emb = FourierEmbs(
                input_dim=input_dim,
                embed_dim=self.fourier_emb.embed_dim,
                embed_scale=self.fourier_emb.embed_scale,
            )
            input_dim = self.fourier_emb.embed_dim

        layers = [nn.Linear(input_dim, hidden_dim), activation()]
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)
        xavier_init(self)
    
    def forward(self, x):
       
        if self.with_fourier and self.fourier_emb is not None:
            x = self.fourier_emb(x)
        return self.model(x)

class ModifiedMlp(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation=nn.Tanh):
        super().__init__()

    def forward(self, x):
        pass

if __name__ == '__main__':
    
    model = Mlp(1, 1, 50, 2) # (input dim, output_dim, hidden_dim, layer_num)
    print(model)