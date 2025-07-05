# Methodology: CausalEngine Algorithm

## 中文版本

### 1. 因果回归理论框架

#### 1.1 核心数学表述

因果回归建立在结构因果模型的基础上，将传统的条件期望学习E[Y|X]重新表述为结构方程学习：

$$Y = f(U, \varepsilon)$$

其中：
- **个体因果表征** $U$：从观察证据$X$推断得到的个体特性表征
- **外生噪声** $\varepsilon$：环境随机性，与$U$独立
- **普适因果律** $f$：对所有个体适用的确定性函数

#### 1.2 个体选择变量的双重身份

个体选择变量$U$具有双重数学含义：

1. **选择变量身份**：$U=u$代表从所有可能个体中选择特定个体$u$
2. **表征载体身份**：向量$u$包含该个体所有内在的、驱动其行为的潜在属性

这种双重身份使得我们能够：
- 从有限观察证据$X$推断个体所属的子群体
- 在该子群体内进行因果推理和预测

#### 1.3 学习目标分解

因果回归将学习问题分解为两个相互关联的子问题：

**个体推断问题**：
$$g^*: X \rightarrow P(U)$$
学习从观察证据到个体表征分布的映射

**因果机制学习**：
$$f^*: U \times \varepsilon \rightarrow Y$$
学习从个体表征和环境噪声到结果的普适映射

总体学习目标：
$$\{f^*, g^*\} = \arg\min_{f,g} \mathbb{E}_{(X,Y) \sim \mathcal{D}}[-\log p(Y|U,\varepsilon)] \text{ where } U \sim g(X)$$

### 2. CausalEngine四阶段架构

#### 2.1 总体设计理念

CausalEngine通过四个透明的推理阶段实现因果回归：

```
感知(Perception) → 归因(Abduction) → 行动(Action) → 决断(Decision)
     ↓                ↓               ↓            ↓
  特征提取Z         推断个体U        计算决策S      任务输出Y
```

每个阶段都有明确的数学定义和因果解释，确保整个推理过程的透明性。

#### 2.2 第一阶段：感知(Perception)

**目标**：从原始输入中提取有意义的特征表示

**数学表述**：
$$Z = \text{Perception}(X) \in \mathbb{R}^{B \times S \times H}$$

**实现方式**：
- 可以使用任何特征提取方法（传统特征工程、深度网络等）
- 关键要求：$Z$应包含识别个体因果表征所需的信息
- 输出：高层特征表示$Z$

#### 2.3 第二阶段：归因(Abduction)

**目标**：从特征证据推断个体因果表征的分布

**核心创新**：基于柯西分布的个体推断
$$P(U|Z) = \text{Cauchy}(\mu_U(Z), \gamma_U(Z))$$

**数学实现**：
```
μ_U = W_loc · Z + b_loc     # 个体群体中心
γ_U = softplus(W_scale · Z + b_scale)  # 群体内部多样性
```

**柯西分布选择的三重考量**：
1. **重尾特性**：为"黑天鹅"个体保留不可忽略概率
2. **未定义矩**：数学上承认个体不可完全刻画的事实
3. **线性稳定性**：实现解析计算而无需采样

#### 2.4 第三阶段：行动(Action)

**目标**：应用普适因果律，从个体表征计算决策分布

**外生噪声注入**：
$$U' = U + \mathbf{b}_{\text{noise}} \cdot \varepsilon$$
其中$\varepsilon \sim \text{Cauchy}(0, 1)$，$\mathbf{b}_{\text{noise}}$是可学习参数

**线性因果律应用**：
$$S = W_{\text{action}} \cdot U' + b_{\text{action}}$$

**分布传播**：由于柯西分布的线性稳定性：
$$S \sim \text{Cauchy}(\mu_S, \gamma_S)$$
其中：
- $\mu_S = W_{\text{action}} \cdot \mu_U + b_{\text{action}}$
- $\gamma_S = |W_{\text{action}}| \cdot (\gamma_U + |\mathbf{b}_{\text{noise}}|)$

#### 2.5 第四阶段：决断(Decision)

**目标**：将抽象决策分布转化为任务特定输出

**结构方程**：
$$Y = \tau(S)$$

**不同任务的实现**：

**回归任务**（路径A：可逆变换）：
- $\tau(s) = s$（恒等映射）
- 损失：柯西负对数似然
$$\mathcal{L} = -\log p_{\text{Cauchy}}(y_{\text{true}}|\mu_S, \gamma_S)$$

**分类任务**（路径B：不可逆变换）：
- $\tau_k(s_k) = \mathbb{I}(s_k > C_k)$（阈值函数）
- One-vs-Rest概率：
$$P_k = \frac{1}{2} + \frac{1}{\pi}\arctan\left(\frac{\mu_{S_k} - C_k}{\gamma_{S_k}}\right)$$
- 损失：二元交叉熵损失

### 3. 关键技术创新

#### 3.1 柯西分布的线性稳定性

**核心性质**：
$$X_1 \sim \text{Cauchy}(\mu_1, \gamma_1), X_2 \sim \text{Cauchy}(\mu_2, \gamma_2)$$
$$\Rightarrow aX_1 + bX_2 \sim \text{Cauchy}(a\mu_1 + b\mu_2, |a|\gamma_1 + |b|\gamma_2)$$

**算法优势**：
- 全流程解析计算，无需采样
- 高效的前向传播和梯度计算
- 数值稳定的不确定性传播

#### 3.2 不确定性的显式分解

**认知不确定性**：
- 来源：归因阶段对个体表征的推断不确定性
- 表示：$\gamma_U$（个体分布的尺度参数）
- 含义：我们对个体的了解程度

**外生不确定性**：
- 来源：行动阶段注入的环境噪声
- 表示：$|\mathbf{b}_{\text{noise}}|$（噪声强度参数）
- 含义：环境的内在随机性

**总不确定性**：
$$\gamma_S = |W_{\text{action}}| \cdot (\gamma_U + |\mathbf{b}_{\text{noise}}|)$$

#### 3.3 推理模式的统一控制

**温度参数**控制噪声强度：
- $T = 0$：纯因果模式，$U' = U$
- $T > 0$：带噪声推理，增强鲁棒性

**采样模式**控制噪声作用方式：
- `do_sample=False`：噪声影响尺度参数
- `do_sample=True`：噪声影响位置参数

### 4. 训练算法

#### 4.1 端到端优化

**目标函数**：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{reg}}$$

其中$\mathcal{L}_{\text{task}}$根据任务类型选择（柯西NLL或BCE），$\mathcal{L}_{\text{reg}}$为正则化项。

**优化策略**：
1. **联合训练**：所有阶段参数同时优化
2. **梯度传播**：通过柯西分布参数进行反向传播
3. **参数初始化**：保持与传统方法的初始等价性

#### 4.2 初始化策略

**知识继承原则**：
- 感知阶段：使用预训练特征提取器
- 归因阶段：初始化为恒等映射
- 行动阶段：初始化为小幅度线性变换
- 决断阶段：根据任务类型初始化

### 5. 算法复杂度分析

**时间复杂度**：
- 前向传播：$O(B \times S \times H)$（与标准神经网络相同）
- 后向传播：$O(B \times S \times H)$（解析梯度计算）

**空间复杂度**：
- 参数存储：$O(H^2 + HV)$（H为隐藏维度，V为词汇表大小）
- 中间结果：$O(B \times S \times H)$

**相比传统方法的优势**：
- 无采样开销
- 解析不确定性计算
- 参数量增加minimal

---

## English Version

### 1. Theoretical Framework of Causal Regression

#### 1.1 Core Mathematical Formulation

Causal Regression is built upon structural causal models, reformulating traditional conditional expectation learning E[Y|X] as structural equation learning:

$$Y = f(U, \varepsilon)$$

where:
- **Individual Causal Representation** $U$: Individual characteristic representation inferred from observed evidence $X$
- **Exogenous Noise** $\varepsilon$: Environmental randomness, independent of $U$
- **Universal Causal Law** $f$: Deterministic function applicable to all individuals

#### 1.2 Dual Identity of Individual Selection Variables

Individual selection variable $U$ has dual mathematical meanings:

1. **Selection Variable Identity**: $U=u$ represents selecting a specific individual $u$ from all possible individuals
2. **Representation Carrier Identity**: Vector $u$ contains all intrinsic properties that drive this individual's behavior

This dual identity enables us to:
- Infer individual subpopulations from limited observed evidence $X$
- Perform causal reasoning and prediction within these subpopulations

#### 1.3 Learning Objective Decomposition

Causal Regression decomposes the learning problem into two interconnected sub-problems:

**Individual Inference Problem**:
$$g^*: X \rightarrow P(U)$$
Learning mapping from observed evidence to individual representation distribution

**Causal Mechanism Learning**:
$$f^*: U \times \varepsilon \rightarrow Y$$
Learning universal mapping from individual representation and environmental noise to outcomes

Overall Learning Objective:
$$\{f^*, g^*\} = \arg\min_{f,g} \mathbb{E}_{(X,Y) \sim \mathcal{D}}[-\log p(Y|U,\varepsilon)] \text{ where } U \sim g(X)$$

### 2. CausalEngine Four-Stage Architecture

#### 2.1 Overall Design Philosophy

CausalEngine implements Causal Regression through four transparent reasoning stages:

```
Perception → Abduction → Action → Decision
     ↓           ↓          ↓        ↓
Feature Z   Infer U    Compute S   Output Y
```

Each stage has clear mathematical definitions and causal interpretations, ensuring transparency throughout the reasoning process.

#### 2.2 Stage 1: Perception

**Objective**: Extract meaningful feature representations from raw inputs

**Mathematical Formulation**:
$$Z = \text{Perception}(X) \in \mathbb{R}^{B \times S \times H}$$

**Implementation**:
- Can use any feature extraction method (traditional feature engineering, deep networks, etc.)
- Key requirement: $Z$ should contain information necessary for identifying individual causal representations
- Output: High-level feature representation $Z$

#### 2.3 Stage 2: Abduction

**Objective**: Infer distribution of individual causal representations from feature evidence

**Core Innovation**: Cauchy distribution-based individual inference
$$P(U|Z) = \text{Cauchy}(\mu_U(Z), \gamma_U(Z))$$

**Mathematical Implementation**:
```
μ_U = W_loc · Z + b_loc     # Individual population center
γ_U = softplus(W_scale · Z + b_scale)  # Within-population diversity
```

**Triple Rationale for Cauchy Distribution**:
1. **Heavy-tail Property**: Preserves non-negligible probability for "black swan" individuals
2. **Undefined Moments**: Mathematically acknowledges the fact that individuals cannot be completely characterized
3. **Linear Stability**: Enables analytical computation without sampling

#### 2.4 Stage 3: Action

**Objective**: Apply universal causal laws, computing decision distributions from individual representations

**Exogenous Noise Injection**:
$$U' = U + \mathbf{b}_{\text{noise}} \cdot \varepsilon$$
where $\varepsilon \sim \text{Cauchy}(0, 1)$, $\mathbf{b}_{\text{noise}}$ is learnable parameter

**Linear Causal Law Application**:
$$S = W_{\text{action}} \cdot U' + b_{\text{action}}$$

**Distribution Propagation**: Due to Cauchy distribution's linear stability:
$$S \sim \text{Cauchy}(\mu_S, \gamma_S)$$
where:
- $\mu_S = W_{\text{action}} \cdot \mu_U + b_{\text{action}}$
- $\gamma_S = |W_{\text{action}}| \cdot (\gamma_U + |\mathbf{b}_{\text{noise}}|)$

#### 2.5 Stage 4: Decision

**Objective**: Transform abstract decision distributions into task-specific outputs

**Structural Equation**:
$$Y = \tau(S)$$

**Implementation for Different Tasks**:

**Regression Tasks** (Path A: Invertible Transform):
- $\tau(s) = s$ (Identity mapping)
- Loss: Cauchy negative log-likelihood
$$\mathcal{L} = -\log p_{\text{Cauchy}}(y_{\text{true}}|\mu_S, \gamma_S)$$

**Classification Tasks** (Path B: Non-invertible Transform):
- $\tau_k(s_k) = \mathbb{I}(s_k > C_k)$ (Threshold function)
- One-vs-Rest probability:
$$P_k = \frac{1}{2} + \frac{1}{\pi}\arctan\left(\frac{\mu_{S_k} - C_k}{\gamma_{S_k}}\right)$$
- Loss: Binary cross-entropy loss

### 3. Key Technical Innovations

#### 3.1 Linear Stability of Cauchy Distributions

**Core Property**:
$$X_1 \sim \text{Cauchy}(\mu_1, \gamma_1), X_2 \sim \text{Cauchy}(\mu_2, \gamma_2)$$
$$\Rightarrow aX_1 + bX_2 \sim \text{Cauchy}(a\mu_1 + b\mu_2, |a|\gamma_1 + |b|\gamma_2)$$

**Algorithmic Advantages**:
- Full-pipeline analytical computation without sampling
- Efficient forward propagation and gradient computation
- Numerically stable uncertainty propagation

#### 3.2 Explicit Uncertainty Decomposition

**Epistemic Uncertainty**:
- Source: Inference uncertainty about individual representations in Abduction stage
- Representation: $\gamma_U$ (scale parameter of individual distribution)
- Meaning: Degree of our knowledge about individuals

**Aleatoric Uncertainty**:
- Source: Environmental noise injected in Action stage
- Representation: $|\mathbf{b}_{\text{noise}}|$ (noise intensity parameter)
- Meaning: Intrinsic randomness of the environment

**Total Uncertainty**:
$$\gamma_S = |W_{\text{action}}| \cdot (\gamma_U + |\mathbf{b}_{\text{noise}}|)$$

#### 3.3 Unified Control of Inference Modes

**Temperature Parameter** controls noise intensity:
- $T = 0$: Pure causal mode, $U' = U$
- $T > 0$: Noisy inference, enhanced robustness

**Sampling Mode** controls how noise acts:
- `do_sample=False`: Noise affects scale parameters
- `do_sample=True`: Noise affects location parameters

### 4. Training Algorithm

#### 4.1 End-to-End Optimization

**Objective Function**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{reg}}$$

where $\mathcal{L}_{\text{task}}$ is selected based on task type (Cauchy NLL or BCE), $\mathcal{L}_{\text{reg}}$ is regularization term.

**Optimization Strategy**:
1. **Joint Training**: All stage parameters optimized simultaneously
2. **Gradient Propagation**: Backpropagation through Cauchy distribution parameters
3. **Parameter Initialization**: Maintaining initial equivalence with traditional methods

#### 4.2 Initialization Strategy

**Knowledge Inheritance Principle**:
- Perception Stage: Use pre-trained feature extractors
- Abduction Stage: Initialize as identity mapping
- Action Stage: Initialize as small-magnitude linear transform
- Decision Stage: Initialize based on task type

### 5. Algorithm Complexity Analysis

**Time Complexity**:
- Forward Propagation: $O(B \times S \times H)$ (same as standard neural networks)
- Backward Propagation: $O(B \times S \times H)$ (analytical gradient computation)

**Space Complexity**:
- Parameter Storage: $O(H^2 + HV)$ (H: hidden dimension, V: vocabulary size)
- Intermediate Results: $O(B \times S \times H)$

**Advantages over Traditional Methods**:
- No sampling overhead
- Analytical uncertainty computation
- Minimal parameter increase