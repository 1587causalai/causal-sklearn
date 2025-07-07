# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**causal-sklearn** is a scikit-learn compatible Python library implementing causal machine learning algorithms based on the CausalEngine™ framework. The core innovation is learning structural equations `Y = f(U, ε)` instead of conditional expectations `E[Y|X]`, where `U` represents unobservable individual causal representations.

## Development Commands

### Quick Testing and Validation
```bash
# Primary validation test - runs both regression and classification
python scripts/quick_test_causal_engine.py

# Extended testing with more methods
python scripts/quick_test_causal_engine_extended.py

# Specific functionality tests
python scripts/test_causal_split.py
python scripts/benchmark_example.py
```

### Robustness Testing (Key Feature)
```bash
# Test regression robustness across noise levels (0%-100%)
python scripts/regression_robustness_real_datasets.py

# Test classification robustness across noise levels
python scripts/classification_robustness_real_datasets.py

# Compare PyTorch vs sklearn MLP baselines
python scripts/pytorch_vs_sklearn_mlp_comparison.py
```

### Comprehensive Examples
```bash
# Main tutorial showcasing all CausalEngine modes
python examples/comprehensive_causal_modes_tutorial_sklearn_style.py

# Real-world data examples (California housing, etc.)
python examples/real_world_regression_tutorial_sklearn_style.py
python examples/real_world_regression_tutorial_extended_sklearn_style.py
```

### Package Development
```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with examples dependencies for plotting
pip install -e ".[examples]"

# Build and publish
python setup.py sdist bdist_wheel
python publish.py  # Automated publishing script
```

### Code Quality (from pyproject.toml)
```bash
# Code formatting
black .

# Linting
flake8 .

# Type checking
mypy .

# Testing
pytest tests/

# Run specific test
pytest tests/test_specific_functionality.py
```

## Architecture Overview

### Core CausalEngine™ Four-Stage Architecture

```
Input (X) → Perception → Representation (Z) → Abduction → 
Causal Representation (U) → Action → Decision Scores (S) → Decision Head → Output (Y)
```

**Stage Breakdown:**
1. **Perception**: Feature extraction using configurable MLP layers (`X → Z`)
2. **Abduction**: Infer individual causal representations as Cauchy distributions (`Z → U ~ Cauchy(μ_U, γ_U)`)
3. **Action**: Transform causal representations to decision scores with learnable noise (`U → S`)
4. **Decision**: Task-specific heads for regression/classification output (`S → Y`)

### Five Inference Modes (Critical for Performance)

- **`deterministic`**: `U' = μ_U` (equivalent to traditional ML, baseline)
- **`exogenous`**: `U' ~ Cauchy(μ_U, |b_noise|)` (environmental randomness dominates)
- **`endogenous`**: `U' ~ Cauchy(μ_U, γ_U)` (cognitive uncertainty dominates)  
- **`standard`**: `U' ~ Cauchy(μ_U, γ_U + |b_noise|)` (both sources combined, **typically best performance**)
- **`sampling`**: `U' ~ Cauchy(μ_U + b_noise*ε, γ_U)` (location parameter perturbation)

### Key Mathematical Foundation

- **Cauchy Distribution**: Central to the framework due to linear stability property enabling analytical computation
- **Individual Selection Variable U**: Dual identity as both individual selector and causal representation
- **Uncertainty Decomposition**: Separates epistemic (cognitive) and aleatoric (environmental) uncertainty

## Main API Components

### Primary Models (`causal_sklearn/`)
- **`MLPCausalRegressor`**: Main causal regression model with 5 inference modes
- **`MLPCausalClassifier`**: Causal classification using One-vs-Rest approach
- **Robust baselines**: `MLPHuberRegressor`, `MLPPinballRegressor`, `MLPCauchyRegressor`
- **PyTorch baselines**: `MLPPytorchRegressor`, `MLPPytorchClassifier`

### Core Engine (`causal_sklearn/_causal_engine/`)
- **`engine.py`**: Main CausalEngine implementation with four-stage architecture
- **`networks.py`**: Perception, Abduction, Action network modules
- **`heads.py`**: Regression and Classification decision heads
- **`math_utils.py`**: Cauchy distribution mathematical operations

## Key Performance Characteristics

**Exceptional robustness in noisy environments** - This is the library's main strength:
- 30% label noise regression: sklearn MLP (MAE: 47.60) vs CausalEngine standard (MAE: 11.41)
- 30% label noise classification: sklearn MLP (Acc: 0.8850) vs CausalEngine standard (Acc: 0.9225)

## Critical Development Notes

### Testing Strategy
- **Always run** `python scripts/quick_test_causal_engine.py` after changes - this is the primary validation
- The `standard` mode typically shows the best performance in noisy conditions
- The `deterministic` mode serves as a causal baseline roughly equivalent to traditional ML

### Data Handling
- All models auto-standardize features and encode labels
- Support for sample weights and early stopping with validation
- Scikit-learn API compatibility (`fit`, `predict`, `score`, `predict_proba`)

### Mathematical Stability
- Cauchy distribution enables analytical computation without sampling
- Linear stability property: `aX + b ~ Cauchy(aμ + b, |a|γ)` if `X ~ Cauchy(μ, γ)`

## Important Documentation

- **`docs/mathematical_foundation.md`**: Complete theoretical framework
- **`docs/ONE_PAGER.md`**: Executive summary
- **`docs/U_deep_dive.md`**: Deep dive into individual selection variable U
- **`docs/blog_post_causal_sklearn.md`**: Comprehensive introduction blog post

## Results and Outputs

- Test results saved in `results/` directory with subdirectories for different test types
- Performance plots and numerical results automatically generated
- Robustness curves show performance across noise levels (0%-100%)

## Development Philosophy

The library focuses on **causal understanding rather than just pattern matching**. The core insight is that individual differences are not "statistical noise" to be suppressed, but "causal information" to be understood through the individual selection variable U and universal causal laws f.