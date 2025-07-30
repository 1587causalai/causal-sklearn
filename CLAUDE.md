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

# Binary classification robustness testing
python scripts/binary_classification_robustness_real_datasets.py

# Generate flowchart explaining the scripts
python scripts/scripts_comparison_flowchart_english.py
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

# Build using modern Python build tools
python -m build --no-isolation

# Check package integrity
twine check dist/*

# Publishing options
python publish.py               # Interactive mode (asks for target)
python publish.py --build-only  # Build only, no upload
python publish.py --test        # Upload to TestPyPI
python publish.py --release     # Upload to PyPI (requires double confirmation)

# Test local installation
python -m venv test_env
source test_env/bin/activate  # Linux/Mac
test_env\Scripts\activate     # Windows
pip install dist/*.whl
python -c 'import causal_sklearn; print(causal_sklearn.__version__)'
deactivate
```

### Code Quality
```bash
# Code formatting (88 char line length)
black . --line-length=88 --target-version=py38

# Linting
flake8 .

# Type checking
mypy . --python-version=3.8 --warn-return-any --warn-unused-configs

# Testing with pytest
pytest tests/
pytest --cov=causal_sklearn tests/  # with coverage

# Run specific test
pytest tests/test_specific_functionality.py::test_function_name
```

### Paper and Documentation
```bash
# Compile LaTeX paper
cd paper/AuthorKit26/AnonymousSubmission/LaTeX/
./compile.sh
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
- Real tests are in `scripts/` directory, not `tests/` (which is minimal)

### Data Handling
- All models auto-standardize features and encode labels
- Support for sample weights and early stopping with validation
- Scikit-learn API compatibility (`fit`, `predict`, `score`, `predict_proba`)
- Support for sklearn built-in datasets and OpenML dataset fetching

### Mathematical Stability
- Cauchy distribution enables analytical computation without sampling
- Linear stability property: `aX + b ~ Cauchy(aμ + b, |a|γ)` if `X ~ Cauchy(μ, γ)`

### Three-Step Tuning Approach (from TODO.md)
1. **Baseline**: Get traditional MLP working well
2. **PyTorch MLP**: Ensure neural network basics are correct
3. **CausalEngine**: Apply causal framework (may need different learning rates)

## Important Documentation

- **`docs/mathematical_foundation.md`**: Complete theoretical framework
- **`docs/ONE_PAGER.md`**: Executive summary
- **`docs/U_deep_dive.md`**: Deep dive into individual selection variable U
- **`docs/blog_post_causal_sklearn.md`**: Comprehensive introduction blog post
- **`docs/cognitive_reconstruction.md`**: Cognitive reconstruction theory
- **`docs/core_mathematical_framework.md`**: Extended mathematical frameworks
- **`docs/decision_framework.md`**: Decision making framework

## Results and Outputs

- Test results saved in `results/` directory with subdirectories for different test types:
  - `results/regression_real_datasets/`: Regression robustness results
  - `results/binary_classification_robustness/`: Classification robustness results
  - `results/quick_test_results/`: Quick validation results
- Performance plots (`.png`) and numerical results (`.npy`, `.txt`) automatically generated
- Robustness curves show performance across noise levels (0%-100%)

## Development Philosophy

The library focuses on **causal understanding rather than just pattern matching**. The core insight is that individual differences are not "statistical noise" to be suppressed, but "causal information" to be understood through the individual selection variable U and universal causal laws f.

## Project Language and Localization

- Primary documentation and comments are in **Chinese** (中文)
- Mathematical notation follows international standards
- API and code interfaces follow English conventions for scikit-learn compatibility
- README.md and key documentation use Chinese to serve the primary user base

## Key Files to Understand

### Core Implementation
- `causal_sklearn/_causal_engine/engine.py`: The heart of CausalEngine implementation
- `causal_sklearn/regressor.py`: User-facing regression models
- `causal_sklearn/classifier.py`: User-facing classification models

### Quick Understanding Scripts
- `scripts/quick_test_causal_engine.py`: Best starting point to understand performance
- `examples/comprehensive_causal_modes_tutorial_sklearn_style.py`: Demonstrates all modes

## Common Development Tasks

### Adding a New Model
1. Inherit from appropriate base class in `causal_sklearn/`
2. Follow existing model patterns (see `MLPCausalRegressor` for reference)
3. Ensure scikit-learn API compatibility
4. Add to `__init__.py` exports
5. Test with `quick_test_causal_engine.py`

### Modifying CausalEngine
1. Core logic is in `_causal_engine/engine.py`
2. Network components in `_causal_engine/networks.py`
3. Mathematical operations in `_causal_engine/math_utils.py`
4. Always test changes with noise robustness scripts

### Publishing Updates
```bash
# Clean old builds first
rm -rf build dist *.egg-info

# Use the automated publishing script
python publish.py

# Or manually:
python -m build --no-isolation
twine check dist/*
twine upload dist/*
```

## Installation Verification

After installation, verify with:
```python
import causal_sklearn
print(f"Causal-sklearn version: {causal_sklearn.__version__}")
print("安装成功！🎉")
```

## Project Metadata

- **Python Version**: 3.8-3.11 supported
- **Development Status**: Alpha
- **License**: Apache-2.0
- **Homepage**: https://github.com/1587causalai/causal-sklearn
- **Bug Reports**: https://github.com/1587causalai/causal-sklearn/issues