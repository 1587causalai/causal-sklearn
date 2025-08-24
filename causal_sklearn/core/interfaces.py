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
