import ml_collections
from examples.viscoelasticity.utils import generate_viscoelastic_dataset
from .models import  PiNFDE_Model
from .trainner import train
from .eval import evaluate
import torch

def main():
    config = ml_collections.ConfigDict({
        "lr": 0.01,
        "epochs": 200, # 論文顯示 PiNFDE 通常在 100-300 次迭代內收斂
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })

    # 1. 產生具備「長程記憶」的真實數據 (Power-law kernel)
    params = {"D": 0.1, "beta": 0.5, "gamma": 0.5, "T": 1.0}
    X, U, W, _ = generate_viscoelastic_dataset(params, nx=50, nt=100)
    
    # 2. 初始化 PiNFDE 模型
    model = PiNFDE_Model(config).to(config.device)
    
    # 3. 執行訓練
    # 這裡假設簡化為時間序列訓練
    t_steps = torch.linspace(0, 1.0, 100).to(config.device)
    train(config, model, (t_steps, U_reshaped), ".")
    
    # 4. 評估
    evaluate(model, (t_steps, U_reshaped))

if __name__ == "__main__":
    main()