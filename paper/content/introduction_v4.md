# Introduction (v4 - Integrated Narrative)

The fields of regression and causal inference have long operated in parallel universes. Regression analysis, typified by predicting California housing prices from district features, excels at learning statistical patterns `P(Y|X)`. Causal inference, conversely, seeks to understand the effects of interventions, `P(Y|do(T=t))`, and is traditionally confined to datasets with explicit "treatments". This division raises a fundamental question for the vast landscape of observational data where no treatments exist: are we forever limited to mere correlation, or can we uncover deeper, causal truths?

This paper argues that the fragility of traditional regression models, especially their failure in noisy environments, is a direct symptom of their superficial learning objective. By focusing on minimizing prediction error for `E[Y|X]`, these models learn a brittle statistical shadow of reality, not the robust, data-generating mechanism itself. The core problem is not about finding a better optimizer or a more complex architecture; it is about asking a fundamentally different question. Instead of "what is the pattern?", we ask, "what is the law?".

Our answer is **Causal Regression**, a new paradigm that redefines the learning objective as discovering the underlying structural causal model, **Y = f(U, ε)**. Here, `f` represents a universal causal law, while the unobservable variable `U` plays a revolutionary dual role: it is both a **selection variable** that uniquely identifies an individual (e.g., a specific housing district) and a **causal representation** that encodes all of that individual's intrinsic properties driving the outcome. The observable features `X` are not direct causes but are instead leveraged as evidence to infer the latent `U`. This reconceptualization, grounded in the theoretical necessity of `U` for modeling counterfactuals, liberates causal analysis from its dependence on explicit interventions.

This shift from error minimization to causal mechanism inference has profound consequences. First, it transforms regression from an optimization problem into a Bayesian inference problem, allowing us to decompose prediction uncertainty into two distinct sources: **endogenous uncertainty (γ_U)**, stemming from our incomplete knowledge of an individual's `U`, and **exogenous randomness (b_noise)**, representing the irreducible stochasticity `ε` of the world. A model can now tell us *why* it is uncertain. Second, this framework allows for true causal discovery from observational data by inferring `U` through a novel abductive reasoning process.

The theoretical elegance of our approach yields significant practical benefits. By modeling uncertainty with the Cauchy distribution, we achieve exact, analytical propagation of uncertainty throughout the model, eliminating the need for computationally expensive sampling. Furthermore, our four-stage architecture (Perception → Abduction → Action → Decision) is not an arbitrary pipeline but a direct implementation of the causal reasoning process, making it **interpretable-by-design**. The model's internal states are not opaque vectors but semantically meaningful representations of its reasoning.

Empirically, Causal Regression demonstrates remarkable robustness, maintaining stable performance in high-noise regimes where traditional methods collapse. It provides meaningful, decomposed uncertainty estimates and achieves this without sacrificing, and often improving upon, predictive accuracy. This paper makes the following contributions:

1.  We introduce Causal Regression, a new framework that enables causal analysis on observational data by learning structural equations `Y = f(U, ε)` rather than conditional expectations `E[Y|X]`.

2.  We develop a principled approach to uncertainty decomposition, distinguishing endogenous uncertainty about latent factors from exogenous environmental randomness.

3.  We present an efficient implementation leveraging Cauchy distributions for exact analytical inference, avoiding the computational costs of sampling-based methods.

4.  We demonstrate empirically that this causal perspective yields models with superior robustness and interpretability compared to traditional regression approaches.

---

# Introduction (v4 - 融合叙事版 中文)

长期以来，回归分析与因果推断仿佛存在于两个平行的宇宙。回归分析的典型任务，如根据加州不同区域的特征预测房价，其核心是学习统计规律 `P(Y|X)`。而因果推断则致力于理解“干预”所产生的效应，即 `P(Y|do(T=t))`，因而其应用场景通常被局限于那些拥有明确“处理”变量的数据集。这种学科上的分野，为海量的、不存在明确干预的观测数据带来了一个根本性的问题：我们是永远只能停留在发现相关性的层面，还是能够揭示其背后更深层次的因果真相？

本文主张，传统回归模型之所以脆弱——尤其是在噪声环境下性能急剧下降——正是其学习目标过于肤浅所导致的直接后果。由于其致力于最小化对 `E[Y|X]` 的预测误差，这些模型学习到的只是现实世界一个脆弱的、随样本变化的统计学投影，而非那个稳健的、真正驱动数据生成的过程本身。核心问题不在于寻找一个更好的优化器或更复杂的模型架构，而在于提出一个完全不同的问题。我们不再问“数据有什么规律？”，我们问：“世界有什么法则？”。

我们的答案是**因果回归（Causal Regression）**。这是一个全新的范式，它将学习的目标重新定义为发现底层的结构因果模型：**Y = f(U, ε)**。在这里，`f` 代表了一条普适的因果定律，而那个不可观测的变量 `U` 则扮演了一个革命性的双重角色：它既是一个用以唯一识别某个体（例如，一个特定的房产区域）的**选择变量**，又是一个编码了该个体所有内在、并驱动最终结果的**因果表征**。我们观测到的特征 `X` 并非结果的直接原因，而是被用作推断潜在变量 `U` 的证据。这种对因果关系的新认知，植根于 `U` 对于反事实建模的理论必然性，从而将因果分析从对“显式干预”的依赖中解放出来。

这一从“最小化误差”到“推断因果机制”的范式转变，带来了极其深远的影响。首先，它将回归从一个优化问题转化为了一个贝叶斯推断问题，这使我们得以将预测的不确定性分解为两个来源：其一是**内生不确定性 (γ_U)**，源于我们对个体 `U` 认知上的局限；其二是**外生随机性 (b_noise)**，代表了世界本身固有的、不可简化的随机扰动 `ε`。模型从此可以告诉我们它为何不确定。其次，该框架通过一种新颖的溯因推理过程来推断 `U`，从而实现了真正意义上从观测数据中发现因果。

我们方法的理论优雅性带来了显著的实践优势。通过选择柯西分布对不确定性进行建模，我们在整个模型中实现了精确的、解析式的不确定性传播，彻底告别了计算昂贵的蒙特卡洛采样。此外，我们设计的四阶段架构（感知→溯因→行动→决策）并非一个随意的计算流程，而是对因果推理过程的直接实现，这使其具备了**“因设计而可解释”**的特质。模型的内部状态不再是晦涩的向量，而是对其推理过程的、拥有明确语义的表征。

在实证方面，因果回归展现了惊人的鲁棒性，在传统方法因数据污染而失效的高噪声环境下，依旧保持了稳定的性能。它能提供有意义的、可分解的不确定性量化，并且这一切优势的获得，非但没有牺牲，反而常常提升了预测的准确性。本文的主要贡献如下：

1.  我们提出因果回归，一个通过学习结构方程 `Y = f(U, ε)` 而非条件期望 `E[Y|X]`，在观测数据上实现因果分析的新框架。

2.  我们开发了一种原则性的不确定性分解方法，能够明确区分关于潜在因子的内生不确定性与来自环境的外生随机性。

3.  我们提出了一个利用柯西分布进行精确解析推理的高效实现，避免了基于采样方法的计算成本。

4.  我们通过实验证明，相比于传统回归方法，这种因果视角所构建的模型具有卓越的鲁棒性和可解释性。 