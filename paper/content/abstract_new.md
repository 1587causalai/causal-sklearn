# Abstract (New Version - Based on Core Story)

## English Version

Traditional regression learns conditional expectations E[Y|X], treating individual differences as statistical noise. We introduce **Causal Regression**, which learns individual causal mechanisms Y=f(U,ε), treating these differences as meaningful information. The key innovation is U—an individual selection variable with dual identity: it identifies who we're examining while encoding their causal attributes. This enables causal analysis without interventions: housing prices have no "treatment," yet every individual has unique causal characteristics.

Our framework transforms how we understand prediction uncertainty. Where traditional methods see a single noise term, we see two distinct sources: epistemic uncertainty (γ_U) about individual characteristics, and aleatoric randomness (b_noise) from the world itself. The **CausalEngine** algorithm implements this through four interpretable stages mirroring human causal reasoning: Perception→Abduction→Action→Decision. Using Cauchy distributions' analytical properties, we achieve exact inference without sampling.

Experiments demonstrate remarkable robustness—30% label noise reduces traditional methods' accuracy by 5x while Causal Regression remains stable. More fundamentally, our models explain their uncertainty: "70% uncertain due to limited knowledge about this individual, 30% due to inherent randomness." This marks machine learning's evolution from memorizing patterns to understanding mechanisms.

## 中文版本

传统回归学习条件期望E[Y|X]，将个体差异视为统计噪声。我们提出**因果回归**，学习个体因果机制Y=f(U,ε)，将这些差异视为有意义的信息。关键创新是U——具有双重身份的个体选择变量：既识别我们正在检查的个体，又编码其因果属性。这使得无需干预即可进行因果分析：房价没有"处理变量"，但每个个体都有独特的因果特征。

我们的框架转变了对预测不确定性的理解。传统方法看到单一的噪声项，我们看到两种不同的来源：关于个体特征的认知不确定性（γ_U），以及来自世界本身的偶然随机性（b_noise）。**CausalEngine**算法通过四个可解释阶段实现这一理念，镜像人类因果推理：感知→溯因→行动→决策。利用柯西分布的解析性质，我们无需采样即可实现精确推断。

实验展示了卓越的鲁棒性——30%标签噪声使传统方法准确率降低5倍，而因果回归保持稳定。更根本的是，我们的模型能解释其不确定性："70%的不确定性源于对该个体认知不足，30%源于固有随机性。"这标志着机器学习从记忆模式到理解机制的演进。

---

## Key Elements Coverage

✅ **开篇悬念**: 房价数据无treatment但可做因果分析
✅ **核心范式转变**: 从E[Y|X]到Y=f(U,ε)的根本转换
✅ **U的革命性**: 双重身份（选择+表征）清晰阐述
✅ **双源分解**: 内生vs外生不确定性
✅ **技术优雅**: 柯西分布和解析计算
✅ **实际影响**: 30%噪声下的鲁棒性
✅ **认知重塑**: 个体差异从"噪声"到"信息"

字数：英文~200词，中文~250字