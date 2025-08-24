import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification

# 1. 导入所有新架构的模块
from causal_engine.core.engine import CausalEngine
from causal_engine.defaults.mlp import MLPPerception, MLPAbduction, LinearAction
from causal_engine.tasks.classification import ClassificationTask

# --- 设置模型参数 ---
INPUT_SIZE = 20
REPRE_SIZE = 32
CAUSAL_SIZE = 10
N_CLASSES = 3
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 10
DISTRIBUTION = "cauchy"

def main():
    print(f"--- Decoupled Causal Engine Classification Tutorial (Distribution: {DISTRIBUTION}) ---")

    # 2. 初始化所有模块
    print("Step 2: Initializing all modules...")
    perception_net = MLPPerception(
        input_size=INPUT_SIZE, repre_size=REPRE_SIZE, hidden_layers=(64, 64)
    )
    abduction_net = MLPAbduction(
        repre_size=REPRE_SIZE, causal_size=CAUSAL_SIZE, hidden_layers=(64,)
    )
    action_net = LinearAction(
        causal_size=CAUSAL_SIZE, output_size=N_CLASSES, distribution=DISTRIBUTION
    )
    classification_task = ClassificationTask(n_classes=N_CLASSES, distribution=DISTRIBUTION)

    # 3. 组装引擎
    print("Step 3: Assembling the CausalEngine...")
    engine = CausalEngine(
        perception=perception_net,
        abduction=abduction_net,
        action=action_net,
        task=classification_task
    )
    print("Engine assembled successfully.")
    print(engine)

    # 4. 创建虚拟数据
    print("\nStep 4: Creating dummy data...")
    X, y = make_classification(
        n_samples=BATCH_SIZE * 20, n_features=INPUT_SIZE, n_informative=10, 
        n_redundant=5, n_classes=N_CLASSES, n_clusters_per_class=2, random_state=42
    )
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.long)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    # 5. 手动编写训练循环
    print("\nStep 5: Starting training loop...")
    optimizer = torch.optim.Adam(engine.parameters(), lr=LEARNING_RATE)
    loss_fn = engine.task.loss

    engine.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # 使用 'standard' 模式进行训练
            mu_S, gamma_S = engine(x_batch, mode='standard')
            
            loss = loss_fn(y_batch, (mu_S, gamma_S))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {total_loss / len(train_loader):.4f}")
    
    print("Training finished.")

    # 6. 使用 predict 方法进行推理
    print("\nStep 6: Performing inference...")
    engine.eval()
    X_test = X_train[:5]
    y_test = y_train[:5]
    with torch.no_grad():
        # 获取点估计预测 (使用'deterministic'模式)
        y_pred_probs = engine.predict(X_test, mode='deterministic')
        y_pred_labels = torch.argmax(y_pred_probs, dim=-1)

        print(f"Test input shape: {X_test.shape}")
        print(f"True labels:      {y_test}")
        print(f"Predicted labels: {y_pred_labels}")
        print(f"Predicted probs (deterministic):\n{y_pred_probs}")

        # 获取 OvR 概率
        ovr_probs = engine.predict(X_test, mode='standard')
        print(f"\nPredicted OvR probs (standard):\n{ovr_probs}")

if __name__ == "__main__":
    main()
