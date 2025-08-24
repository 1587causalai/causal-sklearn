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
