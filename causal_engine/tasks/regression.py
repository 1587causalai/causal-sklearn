from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Head, Loss
from ..core.interfaces import TaskModule
from ..core.math import CausalMath

# --- Head Implementations ---

class RegressionHead(Head):
    def forward(self, decision_scores: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mu_S, _ = decision_scores
        return mu_S

# --- Loss Implementations ---

class NLLLoss(Loss):
    def __init__(self, distribution: str = 'cauchy', reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        if distribution == 'cauchy':
            self.nll_fn = CausalMath.cauchy_nll
        elif distribution == 'normal':
            self.nll_fn = CausalMath.gaussian_nll
        else:
            raise ValueError(f"Unsupported distribution for NLLLoss: {distribution}")

    def forward(self, y_true: torch.Tensor, decision_scores: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mu_S, gamma_S = decision_scores
        # Reshape y_true if necessary to match broadcasting rules
        if y_true.ndim == 1:
            y_true = y_true.unsqueeze(-1)
        return self.nll_fn(y_true, mu_S, gamma_S, reduction=self.reduction)

class SmartRegressionLoss(Loss):
    def __init__(self, distribution: str = 'cauchy', reduction: str = 'mean'):
        super().__init__()
        self.probabilistic_loss = NLLLoss(distribution, reduction=reduction)
        self.deterministic_loss = nn.MSELoss(reduction=reduction)
    
    def forward(self, y_true: torch.Tensor, decision_scores: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mu_S, gamma_S = decision_scores
        if y_true.ndim == 1:
            y_true = y_true.unsqueeze(-1)
            
        if torch.all(gamma_S == 0):
            return self.deterministic_loss(mu_S, y_true)
        else:
            return self.probabilistic_loss(y_true, decision_scores)

# --- TaskModule Implementation ---

class RegressionTask(TaskModule):
    def __init__(self, distribution: str = 'cauchy', reduction: str = 'mean'):
        self._head = RegressionHead()
        self._loss = SmartRegressionLoss(distribution, reduction=reduction)

    @property
    def head(self) -> nn.Module:
        return self._head

    @property
    def loss(self) -> nn.Module:
        return self._loss
