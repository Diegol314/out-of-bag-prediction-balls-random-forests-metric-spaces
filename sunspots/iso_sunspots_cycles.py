import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from pyfrechet.metric_spaces import MetricData, Sphere, Spheroid
from pyfrechet.metrics import mse
from sklearn.model_selection import train_test_split
import itertools
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.d_trees import d_Tree
from sklearn.metrics import make_scorer
import time
from sklearn.model_selection import KFold
import itertools

np.random.seed(1000)
sign_level = np.array([0.01, 0.05, 0.1])

neg_mse = make_scorer(mse, greater_is_better=False) 

def custom_GCV(M_response, structure, X_train, y_train, param_grid, seed=5, n_splits=5):
    """
    Manual Grid Search CV for Fréchet forests.
    
    Parameters:
    - M: Metric space object
    - X_train: array-like, shape (n_samples, n_features)
    - y_train: array-like, shape (n_samples, 3)
    - param_grid: dict with keys 'min_split_size', 'mtry', 'lambda_'
    - seed: int, random seed for reproducibility
    - n_splits: int, number of CV folds

    Returns:
    - final_forest: fitted BaggedRegressor on full training data with best parameters
    - best: dict, best hyperparameters
    - cv_results: list of dicts with CV performance
    """
    grid = list(itertools.product(param_grid['min_split_size'],
                                  param_grid['mtry'])
                                  )

    cv_results = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for min_split_size, mtry in grid:
        fold_errors = []

        for train_index, val_index in kf.split(X_train):
            X_tr, X_val = X_train[train_index], X_train[val_index]
            y_tr_raw, y_val_raw = y_train[train_index], y_train[val_index]

            # Define metric and structure
            y_tr = MetricData(M_response, y_tr_raw.reshape(-1, 3))
            y_val = MetricData(M_response, y_val_raw.reshape(-1, 3))

            # Define forest
            base = d_Tree(split_type='2means', impurity_method='medoid', structure=structure,
                        min_split_size=min_split_size, mtry=mtry)
            forest = BaggedRegressor(estimator=base, n_estimators=200,
                                     bootstrap_fraction=1, bootstrap_replace=True,
                                     seed=seed, n_jobs=12)
            forest.fit(X_tr, y_tr)
            preds = forest.predict(X_val)
            error = mse(y_val, preds)
            fold_errors.append(error)

        avg_error = np.mean(fold_errors)
        cv_results.append({
            'min_split_size': min_split_size,
            'mtry': mtry,
            'cv_error': avg_error
        })
        print(f"Params: min_split_size={min_split_size}, mtry={mtry}, CV error={avg_error:.4f}")

    best = min(cv_results, key=lambda x: x['cv_error'])
    print("\nBest parameters:")
    print(best)

    # Final refit on full training data
    y_train_metric = MetricData(M_response, y_train.reshape(-1, 3))
    final_tree = d_Tree(split_type='2means', impurity_method='medoid',
                      structure=structure,
                      min_split_size=best['min_split_size'], mtry=best['mtry'])
    final_forest = BaggedRegressor(estimator=final_tree, n_estimators=200,
                                   bootstrap_fraction=1, bootstrap_replace=True,
                                   seed=seed, n_jobs=12)
    final_forest.fit(X_train, y_train_metric)

    return final_forest, best, cv_results

def area_pred_ball(M, radius, total_points):
    """
    Estimate the area of a prediction ball using the distance of M with given radius.
    The area is calculated by sampling points "uniformly" on the unit sphere and checking if they are within the radius.
    
    Parameters:
    - M: Metric space
    - radius: float, radius of the ball
    - total_points: int, number of points to sample

    Returns:
    - area: float, area of the ball
    """
    x, y, z = canonical_lattice(total_points)
    return 4*np.pi*np.sum((M.d(np.vstack((x,y,z)).T, np.array([np.sqrt(2)/2,np.sqrt(2)/2,0])) < radius))/total_points

def canonical_lattice(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a canonical lattice on the unit sphere.
    
    Parameters:
    - n: int, number of points to generate

    Returns:
    - x, y, z: coordinates of the points on the sphere
    """
    goldenRatio = (1 + 5**0.5)/2
    i = np.arange(0, n)
    theta = 2 * np.pi * i / goldenRatio
    phi = np.arccos(1 - 2*(i+0.5)/n)
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return x, y, z

def task(cyc):
    filename = f'sunspots_births_{cyc}_deaths.csv'
    print(f'Processing {filename}...')
    filepath = os.path.join(os.getcwd(), 'sunspots/data', filename)

    sample = pd.read_csv(filepath)

    X = np.vstack([sample['births_X.1'], sample['births_X.2'], sample['births_X.3']]).T
    y = np.vstack([sample['deaths_X.1'], sample['deaths_X.2'], sample['deaths_X.3']]).T
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1000)

    # ISOTROPIC RESPONSE - ISOTROPIC PREDICTORS
    # M = Sphere(dim=2)
    M = Spheroid(a=1, c=1)
    param_grid_iso = {'min_split_size': [1, 5, 10], 'mtry': [1]}
    start = time.time()
    iso_forest, _, _ = custom_GCV(M_response= Spheroid(a=1, c=1), structure=[(Sphere(dim=2), list(range(3)))], X_train=X_train, y_train=y_train, param_grid=param_grid_iso)
    end = time.time()
    elapsed_minutes = (end - start) / 60
    print(f"Isotropic response and predictors, cycle {cyc} time: {elapsed_minutes:.1f} minutes")
    iso_oob_quantile = np.percentile(iso_forest.oob_errors(), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')
    iso_preds = iso_forest.predict(X_test)
    iso_pb_ii_cov = np.sum(M.d(MetricData(M, y_test), iso_preds).reshape(X_test.shape[0],1) <= np.tile(iso_oob_quantile, X_test.shape[0]).reshape(-1,3), axis = 0) / X_test.shape[0]

    # Train data
    results = {}
    results['iso_train'] = y_train

    # Test data
    results['iso_test'] = y_test

    # Predictions
    results['iso_preds'] = iso_preds.data

    # MSEs
    results['mse_iso_iso_geo'] = mse(MetricData(Spheroid(a=1, c=1), y_test), MetricData(Spheroid(a=1, c=1), iso_preds.data))

    # OOB quantiles
    results['oob_quantile_iso_iso'] = iso_oob_quantile

    # PB coverages, Type II 
    results['pb_ii_cov_iso_iso'] = iso_pb_ii_cov

    # Areas
    M = Spheroid(a=1, c=1)
    results['area_iso_iso'] = [area_pred_ball(M, r, 5000) for r in iso_oob_quantile]

    # Save
    output_path = f'sunspots/results/new_results_cycle_{cyc}.npy'
    np.save(output_path, results)
    print("Sample saved to", output_path)

# with tqdm_joblib(tqdm(total=total_files)) as progress_bar:
#     Parallel(n_jobs=10)(delayed(task)(cyc) for cyc in file_list)

blocks = [
    [23],
    [21],
    [22],
    [17, 12],
    [16, 13],
    [15, 14],
    [18],
    [19],
    [20]
]

def process_block(block):
    # Process ONE block in parallel using 10 cores
    for cyc in block:
        task(cyc)

for block in blocks:
    print(f"Processing block: {block}")
    process_block(block)