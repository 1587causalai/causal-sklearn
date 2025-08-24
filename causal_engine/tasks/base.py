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
