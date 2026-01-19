import torch
import torch.optim as optim

def train(config, model, train_data, workdir):
    # train_data 包含 (t_obs, u_obs)
    t_obs, u_obs = train_data
    x0 = u_obs[0]
    
    # 使用 Adam 優化器
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    mse_loss = torch.nn.MSELoss() #

    for epoch in range(config.epochs):
        optimizer.zero_grad()
        
        # Forward pass: 求解分數階系統
        u_pred = model(x0, t_obs)
        
        # 計算 Loss: 直接比較預測路徑與實際觀測值
        loss = mse_loss(u_pred, u_obs)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6e} | Alpha: {model.alpha.item():.4f}")