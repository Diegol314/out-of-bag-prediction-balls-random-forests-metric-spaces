# Out-of-Bag Prediction Balls for Random Forests in Metric Spaces

This repository contains the code and data to reproduce all results from the paper "Out-of-Bag Prediction Balls for Random Forests in Metric Spaces".

## Repository Structure

```
├── main_results.ipynb          # Main notebook for generating all results
├── simulation_utils.py         # Utility functions for data loading and plotting
├── pyfrechet/                  # Core library for metric spaces and regression
├── simulations_euc/            # Euclidean space simulations
├── simulations_sphere/         # Sphere simulations  
├── simulations_H2/             # Hyperboloid (H²) simulations
├── simulations_SPD/            # SPD manifold simulations
├── paper_results/              # Generated results (tables, plots)
└── requirements_diego.txt      # Python dependencies
```

## Quick Start

To reproduce all results from the paper:

1. **Install dependencies:**
   ```bash
   pip install -r requirements_diego.txt
   ```

2. **Run the main notebook:**
   ```bash
   jupyter notebook main_results.ipynb
   ```

3. **View results:**
   All tables, plots, and analyses will be saved in the `paper_results/` directory.

## What the Main Notebook Generates

### Euclidean Space
- Type I coverage probability tables (OOB vs Split-Conformal)
- Type II coverage plots showing performance vs sample size
- Type III coverage probability tables  
- Type IV coverage plots
- MSE comparison between methods

### Sphere
- Type I & III coverage probability tables
- Type II & IV coverage plots  
- Radius plots showing OOB quantiles

### Hyperboloid (H²)
- Type I & III coverage probability tables
- Type II & IV coverage plots
- Radius plots showing OOB quantiles

### SPD Manifolds
For each metric (Affine Invariant, Log-Cholesky, Log-Euclidean):
- Type I & III coverage probability tables
- Type II & IV coverage plots
- Radius plots showing OOB quantiles

## Manual Simulation Running (Optional)

If you want to re-run the simulations (this may take considerable time):

### Euclidean Space
```bash
cd simulations_euc
python conformal_euc_main_parallel.py
```

### Sphere  
```bash
cd simulations_sphere
python sphere_parallel.py
```

### Hyperboloid
```bash
cd simulations_H2
python H2_parallel.py 1  # block number
```

### SPD Manifolds
```bash
cd simulations_SPD  
python main_parallel.py 1  # block number
```

## Key Features

- **Reproducible:** All paths are relative, no hard-coded absolute paths
- **Clean:** Single notebook generates all results with clear organization
- **Comprehensive:** Covers all four metric spaces and analysis types from the paper
- **Publication-ready:** High-quality plots and properly formatted tables

## Dependencies

The main dependencies are:
- numpy, pandas, matplotlib, seaborn
- scikit-learn, joblib
- scipy
- tqdm (for progress bars)
- Custom pyfrechet library (included)

See `requirements_diego.txt` for complete list with versions.

## Citation

If you use this code, please cite:

```
[Paper citation to be added]
```

## Contact

For questions or issues, please open an issue on this repository.
