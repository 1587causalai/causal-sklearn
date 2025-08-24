from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Head, Loss
from ..core.interfaces import TaskModule
from ..utils.math import CauchyMath

# --- Head Implementations ---

class RegressionHead(Head):
    def forward(self, decision_scores: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mu_S, _ = decision_scores
        return mu_S

# --- Loss Implementations ---

def _gaussian_nll(y_true, mu, gamma):
    # Here, gamma is consistently treated as the variance (sigma^2)
    sigma_sq = gamma.clamp(min=1e-8) # for stability
    return 0.5 * torch.log(sigma_sq) + 0.5 * ((y_true - mu)**2 / sigma_sq)

class NLLLoss(Loss):
    def __init__(self, distribution: str = 'cauchy'):
        super().__init__()
        if distribution == 'cauchy':
            self.nll_fn = lambda y, mu, gamma: CauchyMath.nll_loss(y, mu, gamma, reduction='none')
        elif distribution == 'normal':
            self.nll_fn = _gaussian_nll
        else:
            raise ValueError(f"Unsupported distribution for NLLLoss: {distribution}")

    def forward(self, y_true: torch.Tensor, decision_scores: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mu_S, gamma_S = decision_scores
        # Reshape y_true if necessary to match broadcasting rules
        if y_true.ndim == 1:
            y_true = y_true.unsqueeze(-1)
        return self.nll_fn(y_true, mu_S, gamma_S).mean()

class SmartRegressionLoss(Loss):
    def __init__(self, distribution: str = 'cauchy'):
        super().__init__()
        self.probabilistic_loss = NLLLoss(distribution)
        self.deterministic_loss = nn.MSELoss()
    
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
    def __init__(self, distribution: str = 'cauchy'):
        self._head = RegressionHead()
        self._loss = SmartRegressionLoss(distribution)

    @property
    def head(self) -> nn.Module:
        return self._head

    @property
    def loss(self) -> nn.Module:
        return self._loss
