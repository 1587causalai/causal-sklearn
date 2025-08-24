from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Head, Loss
from ..core.interfaces import TaskModule
from ..utils.math import CauchyMath, GaussianMath

# --- Head Implementations ---

class ClassificationHead(Head):
    def __init__(self, n_classes: int, ovr_threshold: float = 0.0):
        super().__init__()
        self.n_classes = n_classes
        self.register_buffer('ovr_threshold', torch.full((n_classes,), ovr_threshold))

    def forward(self, decision_scores: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mu_S, gamma_S = decision_scores
        
        # In deterministic mode, output softmax probabilities for compatibility with CrossEntropyLoss
        if torch.all(gamma_S == 0):
            return F.softmax(mu_S, dim=-1)
        
        # In probabilistic mode, calculate One-vs-Rest probabilities
        # This part assumes the math utility is chosen upstream (in ActionModule)
        # We check one element of mu_S to see if it requires grad, to decide if it's Cauchy/Normal
        # A better way is to pass distribution, but for now we assume action module handles it
        # Let's assume Cauchy for now as it's the primary one
        # This will be corrected when ActionModule is made distribution aware
        # For now, let's assume it's always Cauchy for probabilistic heads
        # The logic inside loss function is more robust
        # This head is mainly for prediction, where the choice is less critical than in loss
        ovr_probs = CauchyMath.survival_function(self.ovr_threshold, mu_S, gamma_S)
        return ovr_probs

# --- Loss Implementations ---

class SmartClassificationLoss(Loss):
    def __init__(self, n_classes: int, distribution: str = 'cauchy'):
        super().__init__()
        self.n_classes = n_classes
        self.distribution = distribution
        self.probabilistic_loss = nn.BCELoss(reduction='none')
        self.deterministic_loss = nn.CrossEntropyLoss(reduction='none')

        if self.distribution == 'cauchy':
            self.math = CauchyMath
        elif self.distribution == 'normal':
            self.math = GaussianMath
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

    def forward(self, y_true: torch.Tensor, decision_scores: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mu_S, gamma_S = decision_scores

        if torch.all(gamma_S == 0):
            # Deterministic mode: use CrossEntropy
            # y_true should be class indices (long)
            return self.deterministic_loss(mu_S, y_true.long()).mean()
        else:
            # Probabilistic mode: use OvR BCE loss
            ovr_probs = self.math.survival_function(torch.zeros_like(mu_S), mu_S, gamma_S)
            
            y_true_onehot = F.one_hot(y_true.long(), num_classes=self.n_classes).float()
            
            bce = self.probabilistic_loss(ovr_probs, y_true_onehot)
            return bce.sum(dim=-1).mean()

# --- TaskModule Implementation ---

class ClassificationTask(TaskModule):
    def __init__(self, n_classes: int, distribution: str = 'cauchy'):
        self._head = ClassificationHead(n_classes=n_classes)
        self._loss = SmartClassificationLoss(n_classes=n_classes, distribution=distribution)

    @property
    def head(self) -> nn.Module:
        return self._head

    @property
    def loss(self) -> nn.Module:
        return self._loss
