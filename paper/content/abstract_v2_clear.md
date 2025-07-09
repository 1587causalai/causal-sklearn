# Abstract (v2 - Clear and Accessible)

## English Version

Traditional regression analysis learns conditional expectations E[Y|X], aiming to predict average outcomes for given features. This approach fundamentally treats individual differences as statistical noise to be averaged out. However, this creates a critical blind spot: the inability to understand why specific individuals produce specific outcomes, leading to models that fail when individual-level understanding matters.

This paper introduces **Causal Regression**, a new framework that transforms regression from learning statistical associations to discovering causal mechanisms. We replace the traditional goal of learning E[Y|X] with learning individual causal mechanisms Y = f(U, ε), where U represents each individual's unique causal characteristics and ε represents true environmental randomness. The revolutionary insight is recognizing that individual differences are not noise but meaningful causal information waiting to be decoded.

We implement this framework through **CausalEngine**, an algorithm that performs causal reasoning in four interpretable stages: (1) Perception extracts relevant features, (2) Abduction infers each individual's latent characteristics U, (3) Action applies universal causal laws, and (4) Decision produces final predictions. By modeling U with Cauchy distributions, we achieve exact mathematical inference without expensive sampling procedures.

Our experiments reveal striking advantages. When 30% of training labels are corrupted with noise—a common real-world scenario—traditional neural networks' error increases by 500%, while Causal Regression maintains stable performance. More importantly, our approach provides unprecedented interpretability: for each prediction, the model can report "I am 70% uncertain because I have limited information about this specific instance, and 30% uncertain because of inherent randomness in the outcome."

This work demonstrates that the path to robust and interpretable machine learning lies not in more complex architectures or larger datasets, but in correctly modeling the causal structure of the problem. By transforming regression from pattern matching to causal understanding, we open new possibilities for AI systems that can reason about individuals while acknowledging the limits of their knowledge.

## 中文版本

传统回归分析学习条件期望E[Y|X]，旨在预测给定特征下的平均结果。这种方法从根本上将个体差异视为需要平均掉的统计噪声。然而，这造成了一个关键盲点：无法理解为什么特定个体产生特定结果，导致模型在需要个体层面理解时失败。

本文提出**因果回归（Causal Regression）**，一个将回归从学习统计关联转变为发现因果机制的新框架。我们将传统的学习E[Y|X]目标替换为学习个体因果机制Y = f(U, ε)，其中U代表每个个体独特的因果特征，ε代表真正的环境随机性。革命性的洞察是认识到：个体差异不是噪声，而是等待被解码的有意义的因果信息。

我们通过**CausalEngine**算法实现这个框架，它以四个可解释的阶段执行因果推理：（1）感知提取相关特征，（2）溯因推断每个个体的潜在特征U，（3）行动应用普适因果定律，（4）决策产生最终预测。通过用柯西分布建模U，我们无需昂贵的采样过程就能实现精确的数学推断。

我们的实验揭示了惊人的优势。当30%的训练标签被噪声污染时——这是常见的真实场景——传统神经网络的误差增加500%，而因果回归保持稳定性能。更重要的是，我们的方法提供了前所未有的可解释性：对于每个预测，模型可以报告"我有70%的不确定性是因为对这个特定实例的信息有限，30%的不确定性是因为结果的固有随机性。"

这项工作表明，通向鲁棒和可解释机器学习的道路不在于更复杂的架构或更大的数据集，而在于正确建模问题的因果结构。通过将回归从模式匹配转变为因果理解，我们为能够推理个体同时承认其知识局限的AI系统开辟了新的可能性。

---

## Key Improvements

1. **更友好的开篇**：先解释传统方法做什么，再指出其问题
2. **清晰的概念介绍**：解释U和ε是什么，为什么重要
3. **具体的算法描述**：四个阶段各做什么
4. **生动的实验结果**：500%的误差增加 vs 稳定性能
5. **更大的图景**：最后一段将工作放在AI发展的背景下

约250词，但每句话都能被没有背景的读者理解。