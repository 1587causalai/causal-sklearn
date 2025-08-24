"""
Causal Engine: A Decoupled, Modular, and Extensible Causal Inference Engine

This package provides the core components for building causal machine learning models
based on the CausalEngine™ algorithm.

Core Innovations of CausalEngine:
- 🧠 **Causal Reasoning**: Models the causal mechanism Y = f(U, ε) instead of correlating P(Y|X).
- 🏗️ **Modular Architecture**: Composable modules (Perception, Abduction, Action, Task) via dependency injection.
- 📐 **Cauchy Mathematics**: Utilizes the linear stability of the Cauchy distribution for analytical uncertainty propagation.
- 🔧 **Five Inference Modes**: Supports deterministic, exogenous, endogenous, standard, and sampling modes.

This library is the core PyTorch implementation. For a scikit-learn compatible wrapper, please see the `causal-sklearn` package (TBD).
"""

from ._version import __version__
from .core.engine import CausalEngine
from . import defaults
from . import tasks

__all__ = [
    "__version__",
    "CausalEngine",
    "defaults",
    "tasks",
]

# Package metadata
__author__ = "CausalEngine Team"
__email__ = ""
__license__ = "Apache-2.0"
__description__ = "A decoupled, modular, and extensible causal inference engine in PyTorch"
__theoretical_foundation__ = "Distribution-consistency Structural Causal Models (arXiv:2401.15911)"
__core_innovation__ = "Four-stage causal reasoning: Perception → Abduction → Action → Decision"