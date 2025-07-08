# Introduction (v5 - Enhanced Integration)

When machine learning practitioners mention regression, the California housing dataset immediately comes to mind—a collection of features predicting prices, with no "treatment" in sight. When they mention causality, the conversation invariably turns to randomized trials or A/B tests, where explicit interventions enable causal conclusions. This stark contrast reveals a fundamental tension: Can we extract causal insights from the vast ocean of observational data, or are we forever condemned to correlational analysis? This paper introduces **Causal Regression**, a framework that dissolves this false dichotomy by reconceptualizing what regression truly means.

The need for such reconceptualization becomes painfully clear when we examine the brittleness of traditional regression under real-world conditions. Consider a sobering fact: with just 30% label noise, a standard neural network's mean absolute error on housing price prediction explodes from 8.5 to 47.6—a five-fold degradation. This catastrophic failure is not a bug but a feature of the conventional regression paradigm. By optimizing to minimize squared error, `min E[(Y - f(X))²]`, these models treat every deviation as a mistake to be eliminated. They attempt to memorize noise as signal, leading to inevitable collapse.

This brittleness stems from a deeper philosophical error: confusing statistical association with causal mechanism. Traditional regression asks "What is E[Y|X]?"—a question about conditional averages in a particular dataset. But averages are fragile; they shift with population, time, and measurement noise. The question we should ask is fundamentally different: "What mechanism generates Y?" This is not about finding patterns but discovering laws.

Our framework answers this question through a radical reformulation. Instead of learning conditional expectations, Causal Regression learns structural equations of the form:

**Y = f(U, ε)**

This seemingly simple change conceals profound implications. The function f represents a universal causal law—the same mechanism that operates across all individuals. The variable U serves a revolutionary dual purpose: as an **individual selection variable** that identifies which specific entity we're examining (this particular house, this specific patient), and as a **causal representation** encoding all intrinsic properties that drive outcomes. Observable features X become mere evidence for inferring U, not direct causes of Y.

Most critically, this reformulation transforms the optimization objective itself. Where traditional regression minimizes prediction error—treating uncertainty as an enemy—Causal Regression maximizes likelihood:

**Traditional**: `min E[(Y - f(X))²]`  
**Causal**: `max P(Y|X) = ∫ P(Y|U,ε)P(U|X)dU`

This is not merely a technical adjustment but a philosophical revolution. By maximizing likelihood, we acknowledge that variability in outcomes is not noise to be suppressed but information to be understood. This naturally leads to a principled decomposition of uncertainty into two sources:

- **Endogenous uncertainty (γ_U)**: Our epistemic limitations in knowing an individual's true causal factors
- **Exogenous randomness (b_noise)**: The irreducible stochasticity of the world itself

A model can now explain its uncertainty: "I am 70% uncertain because I don't fully understand this district's unique characteristics, and 30% uncertain because housing markets are inherently volatile."

The mathematical elegance of this framework yields surprising computational benefits. By modeling U with Cauchy distributions—heavy-tailed distributions that assign non-negligible probability to extreme events—we achieve exact analytical propagation of uncertainty throughout the model. The Cauchy's stability under linear transformations (`if U ~ Cauchy(μ,γ), then aU+b ~ Cauchy(aμ+b,|a|γ)`) eliminates the need for Monte Carlo sampling, making inference both efficient and exact.

Furthermore, our four-stage architecture is not an arbitrary design but a computational manifestation of causal reasoning itself:
- **Perception** extracts evidence from raw inputs
- **Abduction** infers latent causal factors from evidence  
- **Action** applies universal causal laws
- **Decision** maps causal outcomes to predictions

Each stage produces interpretable intermediate representations, achieving transparency not through post-hoc explanations but through **interpretability-by-design**.

The empirical consequences are striking. Where traditional methods collapse under noise, Causal Regression maintains remarkable stability—reducing error by over 75% in high-noise regimes. It provides calibrated uncertainty estimates that correctly distinguish epistemic from aleatoric sources. Most surprisingly, these benefits come not at the cost of accuracy but often with improvements in clean-data performance as well.

This paper makes the following contributions:

1. We introduce Causal Regression, a new paradigm that learns causal mechanisms `Y=f(U,ε)` rather than conditional expectations `E[Y|X]`, enabling causal analysis on purely observational data.

2. We present a likelihood-based learning framework that naturally decomposes uncertainty into endogenous and exogenous sources, providing principled uncertainty quantification.

3. We develop an efficient implementation using Cauchy distributions for exact analytical inference, eliminating sampling-based computational bottlenecks.

4. We demonstrate empirically that this causal perspective yields models with exceptional robustness to noise while maintaining interpretability through design.

The implications extend beyond regression. By showing that causal understanding emerges not from interventions but from proper modeling of individual heterogeneity, we open the door to causal analysis across the vast landscape of observational data. The question is no longer whether we can do causal inference without experiments, but why we ever thought we couldn't.

---

# Introduction (v5 - 增强融合版 中文)

当机器学习从业者提到回归时，加州房价数据集会立即浮现在脑海中——一组预测价格的特征，却没有任何"处理变量"的踪影。当他们提到因果时，话题必然转向随机试验或A/B测试，在那里明确的干预使因果结论成为可能。这种鲜明的对比揭示了一个根本性的张力：我们能否从浩瀚的观测数据中提取因果洞察，还是永远被困在相关性分析的囚笼里？本文介绍**因果回归（Causal Regression）**，一个通过重新定义回归的本质来消解这种虚假二分法的框架。

当我们审视传统回归在真实世界条件下的脆弱性时，这种重新定义的必要性变得痛苦地清晰。考虑一个发人深省的事实：仅仅30%的标签噪声，就能让标准神经网络在房价预测上的平均绝对误差从8.5暴增到47.6——五倍的性能退化。这种灾难性的失败不是bug而是feature，是传统回归范式的必然结果。通过优化平方误差最小化，`min E[(Y - f(X))²]`，这些模型将每一个偏差都视为需要消除的错误。它们试图记忆噪声如同信号，导致不可避免的崩溃。

这种脆弱性源于一个更深层的哲学错误：混淆了统计关联与因果机制。传统回归问"E[Y|X]是什么？"——这是关于特定数据集中条件平均值的问题。但平均值是脆弱的；它们随着人群、时间和测量噪声而变化。我们应该问的是一个根本不同的问题："什么机制产生了Y？"这不是关于发现模式，而是关于发现定律。

我们的框架通过一个激进的重构来回答这个问题。因果回归不学习条件期望，而是学习如下形式的结构方程：

**Y = f(U, ε)**

这个看似简单的改变隐藏着深刻的含义。函数f代表一个普适的因果定律——在所有个体上运作的同一机制。变量U扮演着革命性的双重角色：作为**个体选择变量**，它识别我们正在检查的特定实体（这栋特定的房子，这个特定的病人）；作为**因果表征**，它编码了驱动结果的所有内在属性。可观测特征X成为推断U的证据，而非Y的直接原因。

最关键的是，这种重构改变了优化目标本身。传统回归最小化预测误差——将不确定性视为敌人，而因果回归最大化似然：

**传统**：`min E[(Y - f(X))²]`  
**因果**：`max P(Y|X) = ∫ P(Y|U,ε)P(U|X)dU`

这不仅仅是技术调整，而是哲学革命。通过最大化似然，我们承认结果的变异性不是需要压制的噪声，而是需要理解的信息。这自然导致了不确定性的原则性分解：

- **内生不确定性（γ_U）**：我们在认知个体真实因果因子上的认识论局限
- **外生随机性（b_noise）**：世界本身不可约的随机性

模型现在可以解释它的不确定性："我有70%的不确定性是因为我不完全理解这个街区的独特特征，30%的不确定性是因为房地产市场本身就具有内在的波动性。"

这个框架的数学优雅性带来了令人惊讶的计算优势。通过用柯西分布——对极端事件赋予不可忽略概率的重尾分布——建模U，我们在整个模型中实现了精确的解析不确定性传播。柯西分布在线性变换下的稳定性（`若U ~ Cauchy(μ,γ)，则aU+b ~ Cauchy(aμ+b,|a|γ)`）消除了蒙特卡洛采样的需求，使推理既高效又精确。

此外，我们的四阶段架构不是任意设计，而是因果推理本身的计算体现：
- **感知**从原始输入中提取证据
- **溯因**从证据推断潜在因果因子
- **行动**应用普适因果定律
- **决策**将因果结果映射到预测

每个阶段产生可解释的中间表征，不是通过事后解释而是通过**设计即可解释**来实现透明性。

实证结果是惊人的。在传统方法崩溃的噪声环境下，因果回归保持了显著的稳定性——在高噪声条件下将误差降低超过75%。它提供了正确区分认知不确定性和偶然不确定性的校准不确定性估计。最令人惊讶的是，这些优势不是以准确性为代价，反而常常伴随着在干净数据上的性能提升。

本文做出以下贡献：

1. 我们介绍因果回归，一个学习因果机制`Y=f(U,ε)`而非条件期望`E[Y|X]`的新范式，在纯观测数据上实现因果分析。

2. 我们提出基于似然的学习框架，自然地将不确定性分解为内生和外生来源，提供原则性的不确定性量化。

3. 我们开发了使用柯西分布进行精确解析推理的高效实现，消除了基于采样的计算瓶颈。

4. 我们通过实验证明，这种因果视角产生的模型在保持设计可解释性的同时，对噪声具有卓越的鲁棒性。

其含义超越了回归本身。通过展示因果理解不是来自干预而是来自对个体异质性的正确建模，我们为在广阔的观测数据领域进行因果分析打开了大门。问题不再是我们是否能在没有实验的情况下进行因果推断，而是为什么我们曾经认为不能。