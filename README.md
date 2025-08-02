# Out-of-Bag Prediction Balls for Random Forests in Metric Spaces

This repository contains the implementation and reproducible code for the paper "Out-of-Bag Prediction Balls for Random Forests in Metric Spaces."

## Overview

The code implements out-of-bag (OOB) prediction balls for random forests on various metric spaces, providing uncertainty quantification without requiring data splitting.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements_diego.txt
   ```

2. **Run the main analysis:**
   ```bash
   jupyter notebook main_results.ipynb
   ```

3. **View results:**
   All plots and tables are generated in the `results_plots/` directory.

## Repository Structure

```
├── main_results.ipynb          # Main notebook reproducing all paper results
├── original_functions.py       # Core analysis and plotting functions
├── pyfrechet/                  # Metric spaces library
├── simulations_*/              # Raw simulation data for each metric space
├── results_plots/              # Generated plots and visualizations
└── requirements_diego.txt      # Python dependencies
```

## Supported Metric Spaces

- **Euclidean spaces** (ℝᵈ)
- **Unit sphere** (S²)  
- **Hyperboloid** (H²)
- **SPD manifolds** with three metrics:
  - Affine Invariant
  - Log-Euclidean  
  - Log-Cholesky

## Usage

The main interface is through `original_functions.py`:

```python
from original_functions import (
    load_coverage_results,
    create_type_ii_plots,
    create_radius_plots
)

# Load results for a metric space
coverage_df = load_coverage_results('sphere')

# Generate coverage plots
create_type_ii_plots(coverage_df, 'sphere')

# Generate radius analysis
create_radius_plots(coverage_df, 'sphere')
```

## Generated Results

The notebook reproduces all paper results:

- **Coverage analysis:** Type I-IV coverage comparisons between OOB and split-conformal methods
- **Radius plots:** Distribution of prediction ball radii across different parameters
- **MSE comparisons:** Prediction accuracy analysis
- **Geometric visualizations:** Manifold structure and regression curves

## Key Features

- **No data splitting required:** Uses out-of-bag samples for uncertainty quantification
- **Multiple metric spaces:** Unified framework across Euclidean and non-Euclidean spaces
- **Reproducible:** All results generated from included simulation data
- **Publication-ready:** High-quality plots with consistent formatting

## Dependencies

Main requirements:
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn` for random forests
- `scipy` for statistical computations

See `requirements_diego.txt` for complete dependency list.

## Citation

```bibtex
@article{out_of_bag_prediction_balls,
  title={Out-of-Bag Prediction Balls for Random Forests in Metric Spaces},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```
