"""
Core mathematical functions for the Causal Engine.
This module provides stable, reusable implementations of mathematical formulas,
primarily related to the probability distributions used in the engine.
"""
import torch

def cauchy_nll(y_true: torch.Tensor, mu: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Computes the Negative Log-Likelihood for a Cauchy distribution."""
    gamma = gamma.clamp(min=1e-8)  # for stability
    return torch.log(torch.pi * gamma) + torch.log(1 + ((y_true - mu) / gamma)**2)

def gaussian_nll(y_true: torch.Tensor, mu: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """
    Computes the Negative Log-Likelihood for a Gaussian distribution.
    gamma is treated as the variance (sigma^2).
    """
    sigma_sq = gamma.clamp(min=1e-8)  # for stability
    return 0.5 * torch.log(sigma_sq) + 0.5 * ((y_true - mu)**2 / sigma_sq)

def cauchy_survival_function(x: torch.Tensor, mu: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """
    Computes the survival function (1 - CDF) for a Cauchy distribution.
    P(X > x) = 0.5 - (1/pi) * arctan((x - mu) / gamma)
    """
    gamma = gamma.clamp(min=1e-8)
    return 0.5 - (1.0 / torch.pi) * torch.atan((x - mu) / gamma)
