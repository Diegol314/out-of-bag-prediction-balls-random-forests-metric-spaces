import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
import pickle
import numpy as np
import time
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from pyfrechet.metric_spaces import MetricData, two_euclidean

from sklearn.model_selection import train_test_split
from pyfrechet.metric_spaces import MetricData, Euclidean
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.trees import Tree
from pyfrechet.metrics import mse
from sklearn.metrics import make_scorer
from math import gamma

# Read block parameter from command line
n_samples=len(os.listdir(os.path.join(os.getcwd(), 'simulations_euc', 'volume_data')))
current_block = int(sys.argv[1])

data_dir = os.path.join(os.getcwd(), 'simulations_euc', 'volume_data')
results_dir = os.path.join(os.getcwd(), 'simulations_euc', 'conf_volume_results')
os.makedirs(results_dir, exist_ok=True)


# Define parameter grid for tuning
param_grid = {
    'estimator__min_split_size': [1, 5, 10],
    'estimator__mtry': [1, 2, 3]
    }

def tune_forest(X, y, param_grid):
    """Perform hyperparameter tuning using GridSearchCV."""
    base = Tree(split_type='2means', impurity_method='cart')
    forest = BaggedRegressor(estimator=base, n_estimators=200, bootstrap_fraction=1, bootstrap_replace=True, n_jobs=-1)
    grid_search = GridSearchCV(estimator=forest, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=1, verbose=0)
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def volume_of_n_ball(R, dim):
    """
    Calculate the volume of an n-dimensional ball.

    Parameters:
    radius (float): The radius of the ball.
    dimension (int): The dimension of the ball.

    Returns:
    float: The volume of the n-dimensional ball.
    """
    if dim < 0:
        raise ValueError("Dimension must be a non-negative integer.")
    elif dim == 0:
        return 1  # The volume of a 0-dimensional ball is 1 by definition
    else:
        return (np.pi ** (dim / 2)) / gamma(dim / 2 + 1) * (R ** dim)   
    

def task(file):    
    # Load data
    with open(os.path.join(data_dir, file), 'rb') as f:
        sample = pickle.load(f)
    
    dim = int(file.split('_')[3])
    M = two_euclidean(dim = dim)
    X = sample['X']
    X_train, X_test, y_train, y_test = train_test_split(X, sample['Y'], test_size=0.5, random_state=42)

    # Measure time for tuning and fitting
    
    # Perform hyperparameter tuning
    best_forest = tune_forest(X_train, MetricData(M, y_train), param_grid)
    # Fit the best forest
    # best_forest.fit(X, y_train)
    test_preds = best_forest.predict(X_test)
    quantile = np.percentile(M.d(MetricData(M, y_test), test_preds), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')
    volume = volume_of_n_ball(R = quantile, dim = dim)
    # Store results
    results = {
        'radius': quantile,
        'volume': volume
        }   
    filename = os.path.join(results_dir, file[:-4] + '_volume.npy')
    np.save(filename, results)

Parallel(n_jobs=12, verbose=2)(
    delayed(task)(file)
    for file in os.listdir(data_dir)
    if (file.endswith('.pkl') and 
        file.endswith(f'_block_{current_block}.pkl') and
        not os.path.exists(os.path.join(os.getcwd(), 'simulations_euc', 'conf_volume_results/' +  file[:-4]+ '_volume.npy' )))
)