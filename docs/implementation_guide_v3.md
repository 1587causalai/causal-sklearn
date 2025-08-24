# 解耦因果引擎：新分支完整实施指导 (v3.0 - Final)

> **项目背景**: 当前 `causal_sklearn` 的核心引擎存在严重耦合，关键组件被"写死"，无法灵活扩展，限制了算法迭代效率。
>
> **核心目标**: 从第一性原理出发，构建一个完全解耦、高度模块化的新引擎架构。
>
> **开发时长**: 约6小时 (MVP循环式开发)
>
> **最终架构**: 基于**依赖注入**、**“任务即模块”**和**策略模式**的设计，实现数学纯粹性与工程稳健性的统一。

本文档是新分支 `feature/decoupled-engine` 的完整实施指导，涵盖从分支创建到最终验证的每一个步骤。

---

## 阶段 0: 准备工作 (约 0.5 小时)

### 1. 创建分支
```bash
git checkout -b feature/decoupled-engine
```

### 2. 定义新的代码结构
在 `causal_sklearn/` 目录下创建以下结构：
```
causal_sklearn/
├── core/                  # 引擎核心抽象与编排器
│   ├── __init__.py
│   ├── interfaces.py      # 所有模块的ABC接口
│   └── engine.py          # CausalEngine 实现
├── tasks/                 # 预置的TaskModule实现
│   ├── __init__.py
│   ├── base.py            # Head/Loss 模块的ABC接口
│   ├── regression.py      # RegressionTask 及相关组件
│   └── classification.py  # ClassificationTask 及相关组件
└── defaults/              # 默认的 P/A/A 模块实现
    ├── __init__.py
    └── mlp.py             # 基于MLP的默认实现
```
**Why**: 这个结构清晰地分离了核心抽象 (`core`)、任务封装 (`tasks`) 和具体实现 (`defaults`)，是“关注点分离”原则的最佳实践。

---

## 阶段 1: 定义核心抽象 (约 1 小时)

### 1. 创建 `core/interfaces.py`
定义四个核心模块的抽象基类 (ABC)，这是我们架构的契约。
```python
# causal_sklearn/core/interfaces.py
import abc
from typing import Tuple
import torch
import torch.nn as nn

class PerceptionModule(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class AbductionModule(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class ActionModule(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, mu_U: torch.Tensor, gamma_U: torch.Tensor, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class TaskModule(abc.ABC):
    @property
    @abc.abstractmethod
    def head(self) -> nn.Module:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def loss(self) -> nn.Module:
        raise NotImplementedError
```

### 2. 创建 `tasks/base.py`
定义 `Head` 和 `Loss` 模块的接口。
```python
# causal_sklearn/tasks/base.py
import abc
from typing import Tuple
import torch
import torch.nn as nn

class Head(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, decision_scores: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

class Loss(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, y_true: torch.Tensor, decision_scores: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
```

### 3. 提交
```bash
git add causal_sklearn/core/interfaces.py causal_sklearn/tasks/base.py
git commit -m "feat(core): Define abstract interfaces for all engine components"
```
**暂停点**: 确认所有接口定义清晰、完整，这是整个架构的基石。

---

## 阶段 2: 实现核心引擎 (约 1 小时)

### 1. 创建 `core/engine.py`
实现 `CausalEngine` 编排器。
```python
# causal_sklearn/core/engine.py
from typing import Tuple
import torch
import torch.nn as nn
from .interfaces import PerceptionModule, AbductionModule, ActionModule, TaskModule

class CausalEngine(nn.Module):
    def __init__(self, perception: PerceptionModule, abduction: AbductionModule, action: ActionModule, task: TaskModule):
        super().__init__()
        self.perception = perception
        self.abduction = abduction
        self.action = action
        self.task = task

    def forward(self, x: torch.Tensor, mode: str = 'standard') -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.perception(x)
        mu_U, gamma_U = self.abduction(z)
        mu_S, gamma_S = self.action(mu_U, gamma_U, mode=mode)
        return mu_S, gamma_S

    def predict(self, x: torch.Tensor, mode: str = 'standard') -> torch.Tensor:
        decision_scores = self.forward(x, mode=mode)
        return self.task.head(decision_scores)
```

### 2. 提交
```bash
git add causal_sklearn/core/engine.py
git commit -m "feat(core): Implement CausalEngine orchestrator"
```
**暂停点**: 确认引擎的编排逻辑和接口是否与设计完全一致。

---

## 阶段 3: 实现默认组件与任务 (约 2.5 小时)

### 1. 创建 `defaults/mlp.py`
提供开箱即用的 MLP 实现。
```python
# causal_sklearn/defaults/mlp.py
# 实现:
# - MLPPerception(PerceptionModule)
# - MLPAbduction(AbductionModule)
# - LinearAction(ActionModule) -> 关键：实现五种模式逻辑
#   - 在 deterministic 模式下返回 (mu_S, torch.zeros_like(mu_S))
```

### 2. 创建 `tasks/regression.py`
实现 `RegressionTask` 和其依赖的 `Head` 与 `Loss`。
```python
# causal_sklearn/tasks/regression.py
from .base import Head, Loss
from ..core.interfaces import TaskModule
import torch.nn.functional as F

class RegressionHead(Head):
    def forward(self, decision_scores):
        mu_S, _ = decision_scores
        return mu_S

class NLLLoss(Loss):
    def __init__(self, distribution: str = 'cauchy'):
        # ...
    def forward(self, y_true, mu, gamma):
        # ... 实现 cauchy_nll 或 gaussian_nll ...

class SmartRegressionLoss(Loss):
    def __init__(self, distribution: str = 'cauchy'):
        super().__init__()
        self.probabilistic_loss = NLLLoss(distribution)
        self.deterministic_loss = F.mse_loss
    
    def forward(self, y_true, decision_scores):
        mu_S, gamma_S = decision_scores
        if torch.all(gamma_S == 0):
            return self.deterministic_loss(mu_S, y_true)
        else:
            return self.probabilistic_loss(y_true, mu_S, gamma_S)

class RegressionTask(TaskModule):
    def __init__(self, distribution: str = 'cauchy'):
        self._head = RegressionHead()
        self._loss = SmartRegressionLoss(distribution)
    # ... 实现 head 和 loss 的 property ...
```

### 3. 提交
```bash
git add causal_sklearn/defaults/ causal_sklearn/tasks/
git commit -m "feat(defaults): Implement default components and regression task"
```
**暂停点**: 这是工作量最大的一步。确认所有默认组件和任务逻辑正确，特别是 `LinearAction` 的五种模式和 `SmartRegressionLoss` 的策略分发。

---

## 阶段 4: 集成验证与示例 (约 1 小时)

### 1. 创建 `examples/decoupled_engine_tutorial.py`
编写一个端到端的训练示例来验证整个架构。
```python
# examples/decoupled_engine_tutorial.py
# 1. 导入所有模块:
from causal_sklearn.core.engine import CausalEngine
from causal_sklearn.defaults.mlp import MLPPerception, MLPAbduction, LinearAction
from causal_sklearn.tasks.regression import RegressionTask
# 2. 初始化所有模块 (MLPPerception, MLPAbduction, LinearAction, RegressionTask)
# 3. 组装引擎
# 4. 创建虚拟数据
# 5. 手动编写训练循环，验证损失下降
# 6. 使用 predict 方法进行推理
```

### 2. 提交
```bash
git add examples/decoupled_engine_tutorial.py
git commit -m "docs(examples): Add end-to-end tutorial for decoupled engine"
```

---

## 阶段 5: 收尾与文档 (约 0.5 小时)

1.  **代码审查**: 回顾所有代码，检查命名、注释和类型提示。
2.  **更新`README.md`**: 简要介绍新引擎的架构和使用方法，链接到新的教程。
3.  **最终提交**: `git commit -m "refactor(engine): Complete initial implementation of decoupled engine"`
4.  **发起PR (Pull Request)**: 准备合并到主分支。

**任务完成**: 我们将在约6小时内，拥有一个全新的、解耦的、强大的因果推理引擎。
