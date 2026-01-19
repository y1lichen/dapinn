import numpy as np
from sklearn.metrics import r2_score

def evaluate(model, test_data):
    t_test, u_true = test_data
    x0 = u_true[0]
    
    model.eval()
    with torch.no_grad():
        u_pred = model(x0, t_test).cpu().numpy()
        u_true_np = u_true.cpu().numpy()

    # 計算論文說明的 error metrics
    mae = np.mean(np.abs(u_true_np - u_pred))
    mse = np.mean((u_true_np - u_pred)**2)
    r2 = r2_score(u_true_np.flatten(), u_pred.flatten())
    
    print(f"Evaluation Metrics:")
    print(f"MAE: {mae:.4e} | MSE: {mse:.4e} | R2: {r2:.4f}")
    
    # 此處可調用之前的 3D 繪圖函數進行視覺化對照