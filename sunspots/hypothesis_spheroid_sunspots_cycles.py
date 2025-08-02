from joblib import Parallel, delayed
import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from pyfrechet.metric_spaces import MetricData, Sphere, Spheroid, two_euclidean, sphere_to_spheroid, spheroid_to_sphere
from pyfrechet.metrics import mse
from sklearn.model_selection import train_test_split
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.d_trees import d_Tree
from scipy import stats
from statsmodels.stats.multitest import multipletests
import time

np.random.seed(1000)

def train_and_predict(X_train, X_test, y_train, y_test, a, c, seed=5):
    """
    Train a model with specific spheroid parameters and return predictions
    """
    # For sphere case (a=1,c=1), use Sphere(2) directly, otherwise use heroid
    if a == 1 and c == 1:
        M = Sphere(dim=2)
        y_train_metric = MetricData(M, y_train)
    else:
        y_train_spheroid = sphere_to_spheroid(y_train, a, c)
        M = Spheroid(a=a, c=c)
        y_train_metric = MetricData(M, y_train_spheroid.reshape(-1, 3))
    
    # Define structure of predictors
    structure = [(Sphere(dim=2), list(range(3)))]
    # Use cart for sphere (a=c=1), medoid for spheroids
    impurity_method = 'cart' if a == 1 and c == 1 else 'medoid'
    base = d_Tree(split_type='2means', impurity_method=impurity_method, structure=structure,
                  min_split_size=1, mtry=1)
    forest = BaggedRegressor(estimator=base, n_estimators=75,
                           bootstrap_fraction=1, bootstrap_replace=True,
                           seed=seed, n_jobs=12)
    
    # Train model
    forest.fit(X_train, y_train_metric)
    
    # Get predictions
    preds = forest.predict(X_test)
    
    # For sphere case, predictions are already on sphere
    if a == 1 and c == 1:
        sphere_preds = preds.data
    else:
        sphere_preds = spheroid_to_sphere(preds.data, a, c, R=1)
    
    # Calculate point-wise squared errors
    sphere_preds_metric = MetricData(Sphere(2), sphere_preds)
    test_metric = MetricData(Sphere(2), y_test)
    point_errors = np.array([mse(test_metric[i:i+1], sphere_preds_metric[i:i+1])
                            for i in range(len(y_test))])
    
    # Calculate overall MSE
    overall_mse = mse(test_metric, sphere_preds_metric)
    print(f"Overall MSE for {('sphere' if a == 1 and c == 1 else f'spheroid (a={a}, c={c})')}: {overall_mse:.6f}")
    
    return sphere_preds, point_errors

def task(cyc):
    filename = f'sunspots_births_{cyc}_deaths.csv'
    print(f'Processing {filename}...')
    filepath = os.path.join(os.getcwd(), 'sunspots/data', filename)

    # Load data
    sample = pd.read_csv(filepath)
    X = np.vstack([sample['births_X.1'], sample['births_X.2'], sample['births_X.3']]).T
    y = np.vstack([sample['deaths_X.1'], sample['deaths_X.2'], sample['deaths_X.3']]).T
    
    # Same train/test split for all configurations
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1000)

    # Define spheroid configurations
    configs = [
        (0.75, 1),
        (0.8, 1),
        (0.9, 1),
        (1, 1)  # sphere case
    ]

    results = {
        'train_data': {'X': X_train, 'y': y_train},
        'test_data': {'X': X_test, 'y': y_test},
        'predictions': {},
        'point_errors': {},
        'p_values': [],
        'config_params': []
    }

    # Get sphere predictions (baseline)
    sphere_preds, sphere_errors = train_and_predict(X_train, X_test, y_train, y_test, 1, 1)
    results['predictions']['sphere'] = sphere_preds
    results['point_errors']['sphere'] = sphere_errors

    # For each spheroid configuration
    for a, c in configs[:-1]:  # exclude sphere case which we already did
        print(f"Processing spheroid with a={a}, c={c}")
        
        # Train and get predictions
        spheroid_preds, spheroid_errors = train_and_predict(X_train, X_test, y_train, y_test, a, c)
        
        # Store predictions and errors
        config_key = f'spheroid_{a}_{c}'
        results['predictions'][config_key] = spheroid_preds
        results['point_errors'][config_key] = spheroid_errors
        
        # Perform paired t-test
        # H0: mu_sphere = mu_spheroid vs H1: mu_sphere > mu_spheroid
        t_stat, p_val = stats.ttest_rel(sphere_errors, spheroid_errors, alternative='greater')
        results['p_values'].append(p_val)
        results['config_params'].append((a, c))

    # BY correction for multiple testing
    if len(results['p_values']) > 0:
        reject, p_adjusted, _, _ = multipletests(results['p_values'], alpha=0.01, method='fdr_by')
        results['reject_h0'] = reject
        results['p_adjusted'] = p_adjusted

    # Save results
    output_path = f'sunspots/results/hypothesis_results_cycle_{cyc}.npy'
    np.save(output_path, results)
    print("Results saved to", output_path)
    
    # Print results in a nice format
    print(f"\nResults for cycle {cyc}:")
    print("=" * 50)
    for i, ((a, c), p_adj, reject) in enumerate(zip(results['config_params'], 
                                                   results['p_adjusted'], 
                                                   results['reject_h0'])):
        result_str = f"Spheroid (a={a}, c={c}): p-value = {p_adj:.4f}"
        if reject:
            result_str = f"**{result_str}**"  # Mark significant results in bold
        print(result_str)

if __name__ == "__main__":
    blocks = [
        [22],
        [12],
        [16, 13],
        [15, 14],
        [23]
    ]

    def process_block(block):
        for cyc in block:
            task(cyc)

    for block in blocks:
        print(f"Processing block: {block}")
        process_block(block)
