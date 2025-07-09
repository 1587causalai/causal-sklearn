# 关键信息



  1. 开篇悬念（必须保留）

  - 房价数据集没有treatment的悖论
  - 回归分析 vs 因果推断的术语冲突
  - "因果回归"这个名字本身就是矛盾

  2. 四重认知重塑（隐性主线）

  - 学习"物理法则"而非"统计现象"
  - 解放"因果"——从"干预"到"观测"
  - 重构"回归"——从"最小化误差"到"理解随机性来源"
  - 定义"可解释性"——从"外部观察"到"内在同构"

  3. 最大的范式转变

  - 从误差最小化到最大似然
  - 从优化问题到推断问题
  - 从点估计到分布估计
  - 从"拟合"到"理解"

  4. U的革命性意义

  - 理论必然性：建模反事实的数学要求
  - 双重身份：选择变量 + 因果表征
  - 解放因果：不需要explicit treatment

  5. 双源不确定性分解

  - Y的随机性来自两个源头
  - 内生不确定性γ_U（认知局限）
  - 外生随机性b_noise（世界本质）

  6. 技术优雅性

  - 柯西分布的线性稳定性
  - 解析计算，无需采样
  - 四阶段架构的内在可解释性

  7. 实际影响

  - 卓越的噪声鲁棒性
  - 诚实的不确定性量化
  - 真正的可解释性


# Introduction

When one mentions regression analysis, the California housing dataset often comes to mind: a collection of features about districts, and a house price to predict. When one mentions causality, it almost invariably involves a "treatment" from a clinical trial or a marketing campaign. Its core lies in an intervention, a concept seemingly absent in the housing dataset. This paper introduces **Causal Regression**, a novel regression algorithm rooted in causality. An immediate and unavoidable question arises: for a dataset like California housing that lacks any explicit "treatment", what does our "Causal Regression" actually mean?

To answer this, we must view "features" from an entirely new perspective. While traditional regression treats all features as a flat list of predictors, Causal Regression probes deeper, asking: what if these observable features are merely surface-level manifestations of a deeper, unobservable set of **individual causal attributes (`U`)**? For the housing dataset, this latent `U` might represent a district's intrinsic "community quality", "development potential", or "school district prestige"—the true drivers of housing prices. The features we observe, such as median income or house age, are just projections of these deeper truths. Therefore, our Causal Regression on this dataset is not about finding a treatment; it is about **inferring and modeling the unobservable causal variable `U`**. This is the secret to how we infuse causality into traditional regression.

By committing to this deeper, causal pursuit, our algorithm gains two decisive advantages. It achieves **exceptional robustness**, making stable and accurate predictions in noisy environments because it is anchored to causal mechanisms, not brittle correlations. It also offers **unprecedented interpretability**. For instance, it can decompose the uncertainty of a prediction, quantifying how much arises from our limited knowledge of a district's intrinsic quality (`U`), and how much stems from irreducible, external randomness.

This powerful causal framework is not a one-trick pony. We demonstrate its versatility by successfully extending it to classification tasks, delivering similar gains in both performance and clarity. This paper makes the following principal contributions:

1.  **A New Paradigm for Regression.** We redefine the objective of regression analysis. Instead of learning conditional expectations (`E[Y|X]`), we introduce a framework to learn the underlying individual causal mechanisms (`Y = f(U, ε)`), where individual differences are treated as meaningful causal information rather than statistical noise.

2.  **Causal Discovery from Observational Data.** We introduce a novel methodology to discover and model latent causal factors (`U`) from standard, non-interventional datasets where no explicit "treatment" variable exists. This dramatically expands the scope and applicability of causal thinking to a vast range of conventional machine learning problems.

3.  **A Sampling-Free Framework for Uncertainty.** We propose a new, analytical approach to uncertainty quantification. By leveraging the unique properties of the Cauchy distribution, our framework reasons about uncertainty without relying on computationally expensive sampling, while offering a principled decomposition of uncertainty into its endogenous and exogenous sources.

4.  **A New Class of Interpretable-by-Design Models.** We contribute a four-stage architecture that is interpretable by design. Its structure mirrors a transparent causal reasoning process, moving beyond post-hoc explanations to a model whose internal workings are inherently intelligible.

The remainder of this paper details the theory, implementation, and empirical validation of Causal Regression.

---

## 中文草稿

谈及回归分析，很多人脑海中会浮现出那个经典的加州房价数据集：一堆关于街区的特征，一个需要预测的房价。而谈及因果，我们想到的几乎总是临床试验中的某个"处理"（treatment），或是市场营销中的一次干预。它的核心是干预，一个在房价数据集中似乎完全不存在的概念。本文提出了**因果回归（Causal Regression）**，一种根植于因果理论的新型回归算法。一个直接且无法回避的疑问立刻涌现：对于像加州房价这样缺乏明确"处理"变量的数据集，我们的"因果回归"究竟意味着什么？

要回答这个问题，我们必须从一个全新的视角来看待"特征"。传统回归将所有特征视为一张扁平的预测变量列表，而因果回归则探究得更深，它追问：这些可观测的特征，是否仅仅是更深层次的、不可观测的一组**个体因果属性（`U`）**的表层体现？对于房价数据集，这个潜在的`U`可能代表了一个街区内在的"社区品质"、"发展潜力"或"学区声望"——这些才是驱动房价的真正原因。我们观测到的中位数收入或房屋年龄等特征，只是这些深层属性在数据上的投影。因此，我们对该数据集进行的"因果回归"，其目的不是寻找一个处理变量，而是在于**推断和建模那个看不见的因果变量`U`**。这，就是我们将因果思想注入传统回归的秘密。

正因为我们致力于这一更深层次的因果追求，我们的算法获得了两大决定性的优势。它实现了**卓越的鲁棒性**，能在噪声环境中做出稳定而准确的预测，因为它锚定的是因果机制，而非脆弱的相关性。它也提供了**前所未有的可解释性**。例如，它能分解一次预测的不确定性，量化其中有多少源于我们对一个街区内在品质（`U`）的认知局限，又有多少来自不可约的、外部的随机性。

这个强大的因果框架并非只能用于一隅。我们通过将其成功地扩展到分类任务，证明了它的通用性，在性能和清晰度上都带来了相似的增益。本文的主要贡献如下：

1.  **一个回归分析的新范式。** 我们重新定义了回归分析的目标。我们的框架不再是学习条件期望（`E[Y|X]`），而是学习底层的个体因果机制（`Y = f(U, ε)`），其中个体差异被视为有意义的因果信息而非统计噪声。

2.  **从观测数据中发现因果。** 我们引入了一种全新的方法论，用于从没有明确"处理"变量的标准、非干预性数据集中，发现并建模潜在的因果因素（`U`）。这极大地扩展了因果思维在广大传统机器学习问题上的应用范围。

3.  **一个优雅的、非采样的不确定性框架。** 我们提出了一个全新的、解析化的不确定性量化方法。通过利用柯西分布的独特数学特性，我们的框架在推理不确定性时，无需依赖计算昂贵的采样过程，同时还能对不确定性进行原则性的、溯源至内生和外生的分解。

4.  **一类全新的"为解释而生"的模型。** 我们贡献了一个在设计上即具备可解释性的四阶段架构。其结构本身就与一个透明的因果推理过程同构，超越了"事后解释"，实现了模型内在运作机理的真正可理解性。

本文的其余部分将详细介绍因果回归的理论、实现和实证验证。