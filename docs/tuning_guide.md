# CausalEngine™ 调参指南

本指南提供CausalEngine™的调参流程，确保从baseline到最终因果模型的性能优化。

调参流程：

• Step 1: Baseline建立
  - 使用Logistic Regression、XGBoost/LightGBM等传统算法
  - 目标：获得数据集上的基础性能指标

• Step 2: PyTorch MLP调试
  - 确保神经网络基础设置正确（数据处理、初始化、学习率、优化器）
  - 目标：确保神经网络基础配置合理

• Step 3: CausalEngine™优化
  - 加入因果组件，调优因果参数（分类阈值、噪声等）
  - 调整学习率（可能与传统MLP不同）
  - 选择因果推理模式
  - 目标：获得优于PyTorch MLP的性能

注意事项：
- CausalEngine™的学习率可能需要特殊调整
- 因果参数初始化对最终性能影响显著
- 建议逐步调参，确保每个步骤的性能提升 

