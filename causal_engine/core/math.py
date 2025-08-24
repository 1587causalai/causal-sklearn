"""
Core mathematical functions for the Causal Engine.
This module provides stable, reusable implementations of mathematical formulas,
primarily related to the probability distributions used in the engine.
"""
import torch

class CausalMath:
    @staticmethod
    def cauchy_nll(y_true: torch.Tensor, mu: torch.Tensor, gamma: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """Computes the Negative Log-Likelihood for a Cauchy distribution with flexible reduction."""
        gamma = gamma.clamp(min=1e-8)  # for stability
        elementwise_loss = torch.log(torch.pi * gamma) + torch.log(1 + ((y_true - mu) / gamma)**2)
        
        if reduction == 'mean':
            return elementwise_loss.mean()
        elif reduction == 'sum':
            return elementwise_loss.sum()
        elif reduction == 'none':
            return elementwise_loss
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    @staticmethod
    def gaussian_nll(y_true: torch.Tensor, mu: torch.Tensor, gamma: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """
        Computes the Negative Log-Likelihood for a Gaussian distribution with flexible reduction.
        gamma is treated as the variance (sigma^2).
        """
        sigma_sq = gamma.clamp(min=1e-8)  # for stability
        elementwise_loss = 0.5 * torch.log(sigma_sq) + 0.5 * ((y_true - mu)**2 / sigma_sq)

        if reduction == 'mean':
            return elementwise_loss.mean()
        elif reduction == 'sum':
            return elementwise_loss.sum()
        elif reduction == 'none':
            return elementwise_loss
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    @staticmethod
    def cauchy_survival_function(x: torch.Tensor, mu: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        Computes the survival function (1 - CDF) for a Cauchy distribution.
        P(X > x) = 0.5 - (1/pi) * arctan((x - mu) / gamma)
        """
        gamma = gamma.clamp(min=1e-8)
        return 0.5 - (1.0 / torch.pi) * torch.atan((x - mu) / gamma)
