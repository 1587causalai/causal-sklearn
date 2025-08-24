from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.interfaces import PerceptionModule, AbductionModule, ActionModule
from ..utils.math import CauchyMath, GaussianMath

# --- Helper Function from original implementation ---

def build_mlp(
    input_size: int,
    output_size: Optional[int] = None,
    hidden_layers: Optional[Tuple[int, ...]] = None,
    activation: str = "relu",
    dropout: float = 0.0,
) -> nn.Module:
    if output_size is None:
        output_size = input_size
    if hidden_layers is None:
        hidden_layers = ()

    activation_fn = {
        "relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid, "tanh": nn.Tanh,
    }.get(activation.lower(), nn.ReLU)
    
    if not hidden_layers:
        return nn.Linear(input_size, output_size)
    
    layers = []
    current_size = input_size
    for layer_size in hidden_layers:
        layers.extend([nn.Linear(current_size, layer_size), activation_fn()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_size = layer_size
    layers.append(nn.Linear(current_size, output_size))
    return nn.Sequential(*layers)

# --- Default Module Implementations ---

class MLPPerception(PerceptionModule):
    def __init__(
        self,
        input_size: int,
        repre_size: int,
        hidden_layers: Optional[Tuple[int, ...]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.network = build_mlp(
            input_size=input_size, output_size=repre_size, hidden_layers=hidden_layers,
            activation=activation, dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class MLPAbduction(AbductionModule):
    def __init__(
        self,
        repre_size: int,
        causal_size: int,
        hidden_layers: Optional[Tuple[int, ...]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        gamma_init: float = 10.0,
    ):
        super().__init__()
        self.loc_net = build_mlp(
            repre_size, causal_size, hidden_layers, activation, dropout
        )
        self.scale_net = build_mlp(
            repre_size, causal_size, hidden_layers, activation, dropout
        )
        self._init_weights(gamma_init)

    def _init_weights(self, gamma_init):
        # A simplified initialization for default component
        final_layer = self.scale_net[-1] if isinstance(self.scale_net, nn.Sequential) else self.scale_net
        if isinstance(final_layer, nn.Linear):
            init_bias = torch.log(torch.exp(torch.tensor(gamma_init)) - 1)
            nn.init.constant_(final_layer.bias, init_bias.item())
            nn.init.zeros_(final_layer.weight)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_U = self.loc_net(z)
        gamma_U = F.softplus(self.scale_net(z))
        return mu_U, gamma_U

class LinearAction(ActionModule):
    def __init__(
        self,
        causal_size: int,
        output_size: int,
        b_noise_init: float = 0.1,
        b_noise_trainable: bool = True,
        distribution: str = 'cauchy'
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, causal_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        
        if b_noise_trainable:
            self.b_noise = nn.Parameter(torch.full((causal_size,), b_noise_init))
        else:
            self.register_buffer('b_noise', torch.full((causal_size,), b_noise_init))
        
        if distribution == 'cauchy':
            self.math = CauchyMath
        elif distribution == 'normal':
            self.math = GaussianMath
        else:
            raise ValueError(f"Unsupported distribution for LinearAction: {distribution}")

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(
        self, mu_U: torch.Tensor, gamma_U: torch.Tensor, mode: str = 'standard'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if mode == 'deterministic':
            mu_U_final = mu_U
            gamma_U_final = torch.zeros_like(gamma_U)
        elif mode == 'exogenous':
            mu_U_final = mu_U
            gamma_U_final = torch.abs(self.b_noise).unsqueeze(0).expand_as(gamma_U)
        elif mode == 'endogenous':
            mu_U_final = mu_U
            gamma_U_final = gamma_U
        elif mode == 'standard':
            # Use the selected math utility for adding distributions
            noise_loc = torch.zeros_like(mu_U)
            noise_gamma = torch.abs(self.b_noise).unsqueeze(0).expand_as(gamma_U)
            mu_U_final, gamma_U_final = self.math.add_distributions(
                mu_U, gamma_U, noise_loc, noise_gamma
            )
        elif mode == 'sampling':
            # Use the selected math utility for sampling
            noise_samples = self.math.sample(
                torch.zeros_like(mu_U), torch.ones_like(mu_U)
            )
            mu_U_final = mu_U + self.b_noise.unsqueeze(0) * noise_samples
            gamma_U_final = gamma_U
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Use the selected math utility for the linear transformation
        mu_S, gamma_S = self.math.linear_transform(
            mu_U_final, gamma_U_final, self.weight, self.bias
        )
        
        return mu_S, gamma_S
