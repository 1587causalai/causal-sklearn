# Causal-Sklearn 数学文档

本目录包含 causal-sklearn 中 CausalEngine 算法的完整数学基础和理论框架。

## 核心数学文档

为了明确当前开发阶段的重点，我们将文档分为两类：

### 🎯 CausalML for Sklearn (当前分支核心)

这些文档构成了 `causal-sklearn` 实现的直接理论和数学基础，专注于解决常规的分类与回归任务。

- **[ONE_PAGER.md](ONE_PAGER.md)** - 算法概览与高管摘要，面向一般受众的快速入门
- **[mathematical_foundation.md](mathematical_foundation.md)** - 🌟 **最核心** CausalEngine 数学基础 (中文完整版)，包含完整的理论框架
- **[U_deep_dive.md](U_deep_dive.md)** - 深入解析个体选择变量 U 的哲学与数学意义，CausalEngine 的核心创新
- **[decision_framework.md](decision_framework.md)** - 决策阶段的深度理论分析，连接决策分数与任务特定输出

### 🚀 CausalLLM (未来探索方向)

这些文档为项目最终目标——将 CausalEngine 与大语言模型（LLM）结合——提供前瞻性的理论探索。
- **[core_mathematical_framework.md](core_mathematical_framework.md)** - CausalLLM 核心数学框架实现细节，基于 Qwen2.5 的技术参考
- **[core_mathematical_framework_num_extended.md](core_mathematical_framework_num_extended.md)** - 扩展数值理论，专门处理文本-数值混合数据的统一方法

## 关键数学概念

### CausalEngine 四大公理

1. **智能 = 归因 + 行动**：从观测到自我理解再到决策
2. **柯西数学**：唯一支持因果推理解析计算的分布
3. **结构方程决策**：每个选择都由确定性函数计算

### 核心数学框架

CausalEngine 算法基于结构因果方程：

$$
Y = f(U, E)
$$

其中：
- $U$：个体因果表征（从上下文 $X$ 学习得到）
- $E$：外生噪声（独立随机扰动）
- $f$：普适因果机制（确定性函数）

### 三阶段架构

1. **归因推断阶段**：$X → U$（证据到个体表征）
   - AbductionNetwork：将观测映射到因果表征
   - 使用柯西分布实现解析不确定性传播

2. **行动决策阶段**：$U → S$（个体表征到决策得分）
   - ActionNetwork：将表征映射到决策潜能
   - 利用柯西分布线性稳定性实现解析计算

3. **任务激活阶段**：$S → Y$（决策得分到任务输出）
   - ActivationHead：将潜能转换为具体输出（分类/回归）
   - 支持多种推理模式和任务类型

### 数学特性

- **解析计算**：利用柯西分布特性无需采样
- **重尾鲁棒性**：自然处理异常值和极端事件
- **未定义矩**：与真实不确定性哲学对齐
- **尺度不变性**：跨不同尺度的一致行为

## 实现中的使用

这些数学文档作为权威参考用于：

1. **正确性验证**：确保实现与理论框架匹配
2. **参数理解**：所有超参数的数学含义
3. **调试指导**：排查实现问题的理论基础
4. **功能扩展**：添加新特性的数学基础

## 阅读顺序

### 对于实现者和开发者：
1. 从 `ONE_PAGER.md` 开始了解高层概览
2. 阅读 `mathematical_foundation.md` 获得完整理论（**最重要**）
3. 深入理解 `U_deep_dive.md` 中的核心概念
4. 参考 `decision_framework.md` 了解决策阶段的详细方程

### 对于研究者和理论家：
1. 从 `mathematical_foundation.md` 开始（**核心文档**）
2. 深入研究 `U_deep_dive.md` 理解个体选择变量的哲学基础
3. 学习 `decision_framework.md` 的决策理论
4. 探索 `core_mathematical_framework.md` 和 `core_mathematical_framework_num_extended.md` 的高级理论

### 对于 CausalLLM 开发者：
1. 先完成上述基础阅读
2. 重点学习 `core_mathematical_framework.md` 的技术实现
3. 深入研究 `core_mathematical_framework_num_extended.md` 的混合数据处理

## 文档详细说明

### 📋 各文档功能详述

- **ONE_PAGER.md**: 执行摘要和营销推介，解释 CausalEngine 的革命性质和核心优势
- **mathematical_foundation.md**: 完整的理论基础，包含 CausalEngine 的数学原理和架构演进
- **U_deep_dive.md**: 个体选择变量 U 的深度解析，解释因果推理的核心创新
- **decision_framework.md**: 决策阶段的数学基础，涵盖结构方程和似然计算
- **core_mathematical_framework.md**: CausalLLM 的技术实现指南，基于 Qwen2.5 的完整框架
- **core_mathematical_framework_num_extended.md**: 混合数据处理的扩展理论，支持文本-数值统一建模

## 重要说明

> **🎯 当前分支目标**：值得注意的是，本项目的最终目标是将因果推理引擎与大语言模型（LLM）结合。然而，当前 `causal-sklearn` 分支的焦点是**将因果引擎应用于常规的分类和回归任务**，为 `sklearn` 生态提供一个功能强大、理论完备的因果模型。

> **📋 权威规范**：这些文档是 CausalEngine 的权威数学规范。causal-sklearn 中的任何实现都必须严格遵循这些数学定义。
> 
> **🌟 核心文档**：`mathematical_foundation.md` 是最核心、最完整、最准确的数学基础文档，包含最新的理论更新和图解说明。
> 
> **🔍 验证标准**：所有代码实现的正确性都应以这些数学文档为标准进行验证。

## 文档完整性

当前文档集合覆盖了从概念介绍到技术实现的完整光谱：
- 高层理念 → 数学基础 → 核心概念 → 决策理论 → 技术实现 → 扩展应用