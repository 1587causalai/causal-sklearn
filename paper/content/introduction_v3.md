# Introduction (v3 - 学术规范版)

Regression analysis and causal inference have traditionally occupied distinct territories in machine learning. The former focuses on predicting outcomes from features, exemplified by tasks like estimating housing prices from district characteristics. The latter emphasizes understanding interventional effects, typically requiring explicit treatment variables as in clinical trials or A/B tests. This separation has created a methodological gap: datasets with clear interventions enable causal analysis, while observational datasets remain confined to associational studies.

This paper introduces **Causal Regression**, a framework that bridges this divide by enabling causal analysis on standard observational data. Consider the California housing dataset—a canonical example in regression analysis that contains features like median income and house age, but no intervention or treatment variable. Under traditional causal inference frameworks, such data would not support causal conclusions. Yet every observable feature is the effect of some cause, whether or not we explicitly intervened. The challenge lies in discovering and modeling these latent causal factors.

## The Core Innovation: Individual Selection Variable U

Causal Regression fundamentally reframes the regression objective through the introduction of an **individual selection variable U**. This is not merely a technical choice but a theoretical necessity for modeling counterfactuals, as rigorously established in recent work on distribution-consistent structural causal models.

Traditional regression methods learn conditional expectations E[Y|X], treating all individuals as draws from a homogeneous population. In contrast, our framework learns structural equations of the form:

**Y = f(U, ε)**

where U serves a dual role:
- **As a selection variable**: Each realization U=u selects a unique individual from the population
- **As a causal representation**: The vector u encodes all intrinsic properties that drive this individual's behavior

This formulation embodies a profound insight: **the causal law f is universal across all individuals, while individual differences in outcomes arise solely from differences in their intrinsic representations u**. The observable features X serve merely as evidence for inferring which individual (or more precisely, which sub-population of individuals) we are dealing with.

For the housing example, this means:
- The mechanism f that translates district qualities into prices is the same universal law for all districts
- Each district has its unique U encoding intrinsic qualities—development potential, community cohesion, educational infrastructure
- Observable features like median income provide evidence to infer U, but do not directly cause prices

This reconceptualization liberates causal analysis from its traditional dependence on explicit interventions. We do not need randomized experiments or natural experiments to discover causal relationships. Instead, we recognize that **every observable feature is already the effect of latent causal factors**, and our task is to infer these factors from the evidence available.

## Technical Contributions

Our framework introduces several key innovations:

**1. Dual-Source Uncertainty Decomposition**

A fundamental insight underlying our approach is that prediction uncertainty arises from two distinct sources:

- **Endogenous uncertainty (γ_U)**: Reflecting incomplete knowledge about individual causal factors
- **Exogenous randomness (b_noise)**: Representing irreducible stochasticity in the environment

Traditional regression conflates these sources, treating all uncertainty as noise to minimize. By explicitly decomposing them, our framework can distinguish "I am uncertain because I don't fully understand this instance" from "I am uncertain because the world is inherently stochastic." This decomposition enables more nuanced uncertainty quantification and improved robustness.

**2. Abductive Inference for Causal Discovery**

We introduce a novel abduction mechanism that infers latent causal representations U from observable features X. This is achieved through dual neural networks that simultaneously learn:

- The location parameters μ_U representing the most likely causal factors
- The scale parameters γ_U quantifying epistemic uncertainty about these factors

The resulting distribution U ~ Cauchy(μ_U, γ_U) captures both the inferred causal factors and our confidence in that inference.

**3. Analytical Uncertainty Propagation**

By choosing the Cauchy distribution for representing causal factors, we enable exact analytical computation throughout the inference pipeline. The Cauchy distribution's stability under linear transformations allows us to propagate uncertainty without Monte Carlo sampling:

- If U ~ Cauchy(μ, γ), then aU + b ~ Cauchy(aμ + b, |a|γ)

This mathematical elegance translates to computational efficiency and theoretical clarity.

**4. Interpretable-by-Design Architecture**

Our four-stage architecture (Perception → Abduction → Action → Decision) is not merely a computational pipeline but a manifestation of causal reasoning:

- **Perception**: Extracts task-relevant features from raw inputs
- **Abduction**: Infers latent causal factors from observed evidence
- **Action**: Applies causal mechanisms to generate outcomes
- **Decision**: Maps causal outcomes to task-specific predictions

Each stage produces semantically meaningful representations, providing interpretability without sacrificing performance.

## Empirical Validation

We validate Causal Regression across diverse regression benchmarks, with particular focus on robustness to label noise. Key findings include:

- **Superior noise resilience**: While traditional methods degrade rapidly under label corruption, Causal Regression maintains stable performance by correctly attributing noise to exogenous sources rather than attempting to fit it.

- **Meaningful uncertainty quantification**: The dual-source decomposition provides calibrated uncertainty estimates that distinguish epistemic from aleatoric uncertainty.

- **Computational efficiency**: Analytical inference eliminates the need for sampling-based uncertainty quantification.

## Contributions Summary

This paper makes the following contributions:

1. We introduce Causal Regression, a new framework that enables causal analysis on observational data by learning structural equations Y = f(U, ε) rather than conditional expectations E[Y|X].

2. We develop a principled approach to uncertainty decomposition, distinguishing endogenous uncertainty about latent factors from exogenous environmental randomness.

3. We present an efficient implementation leveraging Cauchy distributions for exact analytical inference, avoiding the computational costs of sampling-based methods.

4. We demonstrate empirically that this causal perspective yields models with superior robustness and interpretability compared to traditional regression approaches.

The remainder of this paper is organized as follows. Section 2 reviews related work in robust regression, causal inference, and uncertainty quantification. Section 3 presents the theoretical framework and derives our dual-source uncertainty decomposition. Section 4 details the CausalEngine algorithm and its implementation. Section 5 presents comprehensive experiments validating our approach. Section 6 discusses implications and future directions.

---

# Introduction (v3 - 学术规范版 中文)

回归分析和因果推断在机器学习中传统上占据着不同的领域。前者专注于从特征预测结果，如从街区特征估计房价；后者强调理解干预效应，通常需要明确的处理变量，如临床试验或A/B测试。这种分离造成了方法论上的鸿沟：具有明确干预的数据集能够进行因果分析，而观测数据集仍局限于关联性研究。

本文提出**因果回归（Causal Regression）**，一个通过在标准观测数据上实现因果分析来弥合这一鸿沟的框架。考虑加州房价数据集——回归分析中的典型案例，包含中位收入和房龄等特征，但没有干预或处理变量。在传统因果推断框架下，此类数据无法支持因果结论。然而，每个可观测特征都是某个原因的结果，无论是否明确干预。挑战在于发现和建模这些潜在的因果因子。

## 核心创新：个体选择变量U

因果回归通过引入**个体选择变量U**从根本上重新构建了回归目标。这不仅是技术选择，更是建模反事实的理论必然性，正如最近关于分布一致性结构因果模型的工作所严格证明的。

传统回归方法学习条件期望E[Y|X]，将所有个体视为来自同质总体的抽样。相比之下，我们的框架学习如下形式的结构方程：

**Y = f(U, ε)**

其中U扮演双重角色：
- **作为选择变量**：每个实现U=u从总体中选择一个独特的个体
- **作为因果表征**：向量u编码了驱动该个体行为的所有内在属性

这一表述体现了深刻的洞察：**因果律f对所有个体都是普适的，而结果的个体差异完全源于其内在表征u的差异**。可观测特征X仅作为推断我们正在处理哪个个体（或更准确地说，哪个个体子群）的证据。

对于房价示例，这意味着：
- 将街区品质转化为价格的机制f是适用于所有街区的同一普适定律
- 每个街区都有其独特的U，编码了内在品质——发展潜力、社区凝聚力、教育基础设施
- 中位收入等可观测特征提供了推断U的证据，但并不直接导致价格

这种重新概念化将因果分析从对明确干预的传统依赖中解放出来。我们不需要随机实验或自然实验来发现因果关系。相反，我们认识到**每个可观测特征都已经是潜在因果因子的结果**，我们的任务是从可用证据中推断这些因子。

## 技术贡献

我们的框架引入了几项关键创新：

**1. 双源不确定性分解**

支撑我们方法的一个基本洞察是，预测不确定性源于两个不同的来源：

- **内生不确定性（γ_U）**：反映对个体因果因子的不完全认知
- **外生随机性（b_noise）**：代表环境中不可约的随机性

传统回归混淆了这些来源，将所有不确定性都视为需要最小化的噪声。通过明确分解它们，我们的框架能够区分"由于对该实例理解不充分而产生的不确定性"和"由于世界本质随机而产生的不确定性"。这种分解实现了更细致的不确定性量化和改进的鲁棒性。

**2. 因果发现的溯因推理**

我们引入了一种新颖的溯因机制，从可观测特征X推断潜在因果表征U。这通过双神经网络实现，它们同时学习：

- 位置参数μ_U，代表最可能的因果因子
- 尺度参数γ_U，量化关于这些因子的认知不确定性

结果分布U ~ Cauchy(μ_U, γ_U)既捕获了推断的因果因子，也捕获了我们对该推断的置信度。

**3. 解析不确定性传播**

通过选择柯西分布表示因果因子，我们在整个推理管道中实现了精确的解析计算。柯西分布在线性变换下的稳定性使我们能够传播不确定性而无需蒙特卡洛采样：

- 若U ~ Cauchy(μ, γ)，则aU + b ~ Cauchy(aμ + b, |a|γ)

这种数学优雅性转化为计算效率和理论清晰度。

**4. 设计即可解释架构**

我们的四阶段架构（感知→溯因→行动→决策）不仅是计算管道，更是因果推理的体现：

- **感知**：从原始输入中提取任务相关特征
- **溯因**：从观察证据推断潜在因果因子
- **行动**：应用因果机制生成结果
- **决策**：将因果结果映射到特定任务预测

每个阶段都产生语义有意义的表征，在不牺牲性能的情况下提供可解释性。

## 实证验证

我们在多样化的回归基准上验证了因果回归，特别关注对标签噪声的鲁棒性。主要发现包括：

- **卓越的噪声韧性**：当传统方法在标签损坏下迅速退化时，因果回归通过正确地将噪声归因于外生来源而非试图拟合它，保持了稳定的性能。

- **有意义的不确定性量化**：双源分解提供了校准的不确定性估计，区分了认知不确定性和偶然不确定性。

- **计算效率**：解析推理消除了基于采样的不确定性量化的需求。

## 贡献总结

本文做出以下贡献：

1. 我们提出因果回归，一个通过学习结构方程Y = f(U, ε)而非条件期望E[Y|X]，在观测数据上实现因果分析的新框架。

2. 我们开发了一种原则性的不确定性分解方法，区分关于潜在因子的内生不确定性和外生环境随机性。

3. 我们提出了一个利用柯西分布进行精确解析推理的高效实现，避免了基于采样方法的计算成本。

4. 我们通过实验证明，这种因果视角产生的模型相比传统回归方法具有卓越的鲁棒性和可解释性。

本文其余部分组织如下。第2节回顾鲁棒回归、因果推断和不确定性量化的相关工作。第3节介绍理论框架并推导双源不确定性分解。第4节详述CausalEngine算法及其实现。第5节展示验证我们方法的综合实验。第6节讨论意义和未来方向。