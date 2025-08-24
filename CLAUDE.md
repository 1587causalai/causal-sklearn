# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**causal-engine** (formerly causal-sklearn) is a PyTorch-based library implementing causal machine learning algorithms based on the CausalEngine™ framework. It models underlying causal mechanisms **Y = f(U, ε)** instead of traditional correlation-based approaches **E[Y|X]**.

The library is currently undergoing a major refactoring on the `decoupled-engine` branch to provide a modular, extensible, and mathematically pure architecture.

## Development Commands

### Quick Testing
```bash
# Run regression example
python examples/tutorial_regression.py

# Run classification example  
python examples/tutorial_classification.py
```

### Package Development
```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with examples dependencies for plotting
pip install -e ".[examples]"

# Build package
python -m build --no-isolation

# Check package integrity
twine check dist/*

# Publishing
python publish.py               # Interactive mode
python publish.py --build-only  # Build only
python publish.py --test        # Upload to TestPyPI
python publish.py --release     # Upload to PyPI
```

### Code Quality
```bash
# Format code (88 char line length)
black . --line-length=88 --target-version=py38

# Linting
flake8 .

# Type checking
mypy . --python-version=3.8 --warn-return-any --warn-unused-configs

# Testing
pytest tests/
pytest --cov=causal_engine tests/
```

## New Decoupled Architecture

### Core Philosophy
- **Composition over Inheritance**: CausalEngine is an orchestrator composing independent modules
- **Task as a Module**: Task-specific logic encapsulated in swappable TaskModule
- **Contract-Driven Design**: All components adhere to strict ABC interfaces
- **Distribution-Aware**: Consistent handling of probability distributions (Cauchy, Gaussian)

### Four Injectable Modules

```
1. PerceptionModule: X → Z (Feature Extraction)
2. AbductionModule: Z → U (Causal Representation Distribution)  
3. ActionModule: U → S (Decision Score Distribution)
4. TaskModule: Encapsulates Head (prediction) and Loss (training)
```

### Five Inference Modes

- **`deterministic`**: U' = μ_U (traditional ML baseline)
- **`exogenous`**: U' ~ Cauchy(μ_U, |b_noise|) (environmental randomness)
- **`endogenous`**: U' ~ Cauchy(μ_U, γ_U) (cognitive uncertainty)
- **`standard`**: U' ~ Cauchy(μ_U, γ_U + |b_noise|) (combined, typically best)
- **`sampling`**: U' ~ Cauchy(μ_U + b_noise*ε, γ_U) (location perturbation)

## Core Implementation Structure

### New Engine (`causal_engine/`)
- **`core/engine.py`**: Main CausalEngine orchestrator
- **`core/interfaces.py`**: Abstract base classes for all modules
- **`defaults/mlp.py`**: Default MLP implementations (MLPPerception, MLPAbduction, LinearAction)
- **`tasks/regression.py`**: RegressionTask module
- **`tasks/classification.py`**: ClassificationTask module
- **`utils/math.py`**: Mathematical utilities for distributions

### Module Interfaces

```python
# PerceptionModule
forward(x) -> z

# AbductionModule  
forward(z) -> (μ_U, γ_U)

# ActionModule
forward(mu_U, gamma_U, mode='standard') -> (μ_S, γ_S)

# TaskModule
@property head -> nn.Module  # Head.forward(decision_scores) -> y_pred
@property loss -> nn.Module  # Loss.forward(y_true, decision_scores) -> loss
```

## Usage Pattern

```python
# 1. Initialize modules independently
perception = MLPPerception(input_size=10, repre_size=20)
abduction = MLPAbduction(repre_size=20, causal_size=5)
action = LinearAction(causal_size=5, output_size=1, distribution="cauchy")
task = RegressionTask(distribution="cauchy")

# 2. Assemble engine via dependency injection
engine = CausalEngine(
    perception=perception,
    abduction=abduction,
    action=action,
    task=task
)

# 3. Standard PyTorch training
optimizer = torch.optim.Adam(engine.parameters(), lr=1e-3)
loss_fn = engine.task.loss

# Forward pass
mu_S, gamma_S = engine(X_batch)
loss = loss_fn(y_batch, (mu_S, gamma_S))
```

## Key Development Notes

### Mathematical Foundation
- **Cauchy Distribution**: Central due to linear stability property
- **Linear Stability**: aX + b ~ Cauchy(aμ + b, |a|γ) if X ~ Cauchy(μ, γ)
- **Individual Selection Variable U**: Dual identity as selector and causal representation

### Distribution Handling
- ActionModule manages distribution type (cauchy/gaussian)
- TaskModule's loss function auto-selects strategy based on γ_S
- Deterministic mode returns (μ_S, zeros) for interface consistency

### Three-Step Tuning Approach
1. **Baseline**: Traditional algorithms (logistic regression, XGBoost)
2. **PyTorch MLP**: Validate neural network basics
3. **CausalEngine**: Apply causal framework (may need different learning rates)

## Important Documentation

- **`docs/decoupled_causal_engine_spec.md`**: Architecture specification
- **`docs/mathematical_foundation.md`**: Theoretical framework
- **`docs/U_deep_dive.md`**: Individual selection variable theory
- **`docs/tuning_guide.md`**: Performance tuning guidance

## Project Metadata

- **Python Version**: 3.8-3.11
- **Main Dependencies**: PyTorch, NumPy, SciPy, scikit-learn
- **License**: Apache-2.0
- **Current Branch**: decoupled-engine
- **Status**: Alpha (major refactoring in progress)