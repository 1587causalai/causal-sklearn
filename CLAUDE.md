# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **causal-sklearn**, a Python library that provides scikit-learn compatible implementations of causal machine learning algorithms based on the CausalEngine™ framework. The library implements causal regression and classification that goes beyond traditional correlation-based machine learning to understand true causal relationships.

## Development Commands

### Testing
- Run quick validation tests: `python scripts/quick_test_causal_engine.py`
- Run extended tests: `python scripts/quick_test_causal_engine_extended.py`
- Run single test script: `python scripts/test_causal_split.py`

### Examples and Tutorials
- Quick start with comprehensive tutorial: `python examples/comprehensive_causal_modes_tutorial_sklearn_style.py`
- Real-world regression examples: `python examples/real_world_regression_tutorial_sklearn_style.py`
- Extended real-world tutorial: `python examples/real_world_regression_tutorial_extended_sklearn_style.py`

### Robustness Testing
- Regression robustness testing: `python scripts/regression_robustness_real_datasets.py`
- Classification robustness testing: `python scripts/classification_robustness_real_datasets.py`

### Code Quality
- Format code: `black .` (requires `pip install black`)
- Run linter: `flake8 .` (requires `pip install flake8`)
- Type checking: `mypy .` (requires `pip install mypy`)
- Run tests: `pytest tests/` (requires `pip install pytest`)

### Package Management
- Install in development mode: `pip install -e .`
- Install with dev dependencies: `pip install -e ".[dev]"`
- Install with examples dependencies: `pip install -e ".[examples]"`

## Architecture Overview

### Core Components

1. **CausalEngine Core** (`causal_sklearn/_causal_engine/`):
   - Four-stage architecture: Perception → Abduction → Action → Decision
   - Networks: Perception, Abduction, Action modules
   - Decision heads: Regression and Classification heads
   - Math utilities: Cauchy distribution mathematics

2. **Main API** (`causal_sklearn/`):
   - `MLPCausalRegressor`: Causal regression with 5 inference modes
   - `MLPCausalClassifier`: Causal classification with OvR approach
   - Robust regressors: `MLPHuberRegressor`, `MLPPinballRegressor`, `MLPCauchyRegressor`
   - PyTorch baselines: `MLPPytorchRegressor`, `MLPPytorchClassifier`

### Four-Stage CausalEngine Architecture

```
Input (X) → Perception → High-level Representation (Z) → Abduction → 
Causal Representation (U) → Action → Decision Scores (S) → Decision Head → Output (Y)
```

1. **Perception**: Feature extraction using MLP layers
2. **Abduction**: Infer individual causal representations with Cauchy distributions
3. **Action**: Transform causal representations to decision scores
4. **Decision**: Task-specific heads for regression/classification output

### Five Inference Modes

- `deterministic`: U' = μ_U (no randomness)
- `exogenous`: U' ~ Cauchy(μ_U, |b_noise|) (exogenous noise dominates)
- `endogenous`: U' ~ Cauchy(μ_U, γ_U) (endogenous uncertainty dominates)
- `standard`: U' ~ Cauchy(μ_U, γ_U + |b_noise|) (both sources combined)
- `sampling`: U' ~ Cauchy(μ_U + b_noise*ε, γ_U) (location parameter perturbation)

## Key Features

- **Sklearn compatibility**: Drop-in replacement for `MLPRegressor`/`MLPClassifier`
- **Causal inference**: Beyond correlation to understand causal relationships
- **Robustness**: Superior performance with noisy labels and outliers
- **Distribution prediction**: Full distributional outputs, not just point estimates
- **Mathematical foundation**: Based on Cauchy distribution's linear stability

## Important Files

- `causal_sklearn/_causal_engine/engine.py`: Core CausalEngine implementation
- `causal_sklearn/regressor.py`: Regression models including causal and robust variants
- `causal_sklearn/classifier.py`: Classification models
- `docs/mathematical_foundation.md`: Detailed mathematical theory
- `examples/`: Comprehensive tutorials and real-world examples
- `scripts/`: Testing, benchmarking, and robustness evaluation scripts

## Development Notes

- The library uses PyTorch for neural network implementation
- All models follow sklearn API conventions (fit/predict/score)
- Supports sample weights and validation-based early stopping
- Automatic data standardization and label encoding
- Comprehensive testing includes synthetic and real-world datasets with various noise levels
- Focus on causal understanding rather than just pattern matching