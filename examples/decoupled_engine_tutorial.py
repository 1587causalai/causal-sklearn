import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1. 导入所有新架构的模块
from causal_sklearn.core.engine import CausalEngine
from causal_sklearn.defaults.mlp import MLPPerception, MLPAbduction, LinearAction
from causal_sklearn.tasks.regression import RegressionTask

# --- 设置模型参数 ---
INPUT_SIZE = 10
REPRE_SIZE = 20
CAUSAL_SIZE = 5
OUTPUT_SIZE = 1
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10

def main():
    print("--- Decoupled Causal Engine Tutorial ---")

    # 2. 初始化所有模块
    print("Step 2: Initializing all modules...")
    perception_net = MLPPerception(
        input_size=INPUT_SIZE, repre_size=REPRE_SIZE, hidden_layers=(32, 32)
    )
    abduction_net = MLPAbduction(
        repre_size=REPRE_SIZE, causal_size=CAUSAL_SIZE, hidden_layers=(32,)
    )
    action_net = LinearAction(causal_size=CAUSAL_SIZE, output_size=OUTPUT_SIZE)
    
    # 关键：通过配置创建任务
    regression_task = RegressionTask(distribution="cauchy")

    # 3. 组装引擎
    print("Step 3: Assembling the CausalEngine...")
    engine = CausalEngine(
        perception=perception_net,
        abduction=abduction_net,
        action=action_net,
        task=regression_task
    )
    print("Engine assembled successfully.")
    print(engine)

    # 4. 创建虚拟数据
    print("\nStep 4: Creating dummy data...")
    X_train = torch.randn(BATCH_SIZE * 10, INPUT_SIZE)
    # y = 2*x_1 - 3*x_2 + noise
    y_train = 2 * X_train[:, 0] - 3 * X_train[:, 1] + 0.5 * torch.randn(BATCH_SIZE*10)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    # 5. 手动编写训练循环
    print("\nStep 5: Starting training loop...")
    optimizer = torch.optim.Adam(engine.parameters(), lr=LEARNING_RATE)
    loss_fn = engine.task.loss  # 从task中获取智能损失函数

    engine.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # 引擎输出S分布 (使用默认 'standard' 模式)
            mu_S, gamma_S = engine(x_batch)
            
            # loss_fn 内部会自动处理模式
            loss = loss_fn(y_batch, (mu_S, gamma_S))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {total_loss / len(train_loader):.4f}")
    
    print("Training finished.")

    # 6. 使用 predict 方法进行推理
    print("\nStep 6: Performing inference...")
    engine.eval()
    X_test = torch.randn(5, INPUT_SIZE)
    with torch.no_grad():
        # 获取点估计预测 (使用'deterministic'模式)
        y_pred = engine.predict(X_test, mode='deterministic')
        print(f"Test input:\n{X_test}")
        print(f"Deterministic predictions:\n{y_pred.squeeze()}")

        # 获取完整分布用于分析
        mu_S_dist, gamma_S_dist = engine(X_test, mode='standard')
        print(f"\nDistribution parameters (mu_S):\n{mu_S_dist.squeeze()}")
        print(f"Distribution parameters (gamma_S):\n{gamma_S_dist.squeeze()}")

if __name__ == "__main__":
    main()
