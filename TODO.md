# TODO

构建一个调参指南文档，用于指导用户如何使用CausalEngine™进行调参。

step 1. 使用 logistic regression, xgboost/lightgbm 等算法跑通 baseline

step 2. 使用 pytorch MLP 算法调试出来一个基本的性能， 确保我们神经网络用到的数据处理， 初始化， 学习速率，优化器等基本上合适。 

step 3. 加入 causal-sklearn 算法，调试出来比 pytorch MLP 更好的性能，确保我们的相关因果参数，比如分类阈值和噪声等设置初始化基本正确，注意因果引擎适合的学习速率可能与 pytorch MLP 不同。 


