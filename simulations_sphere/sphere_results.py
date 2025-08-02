import sys, os
from pathlib import Path

# Get the root directory (parent of simulations_sphere)
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
import numpy as np
import pickle
import joblib
from joblib import Parallel, delayed
from pyfrechet.metric_spaces import MetricData, Sphere
from pyfrechet.regression.bagged_regressor import BaggedRegressor
from pyfrechet.regression.trees import Tree
from pyfrechet.metrics import mse
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from scipy.stats import vonmises_fisher, vonmises_line
from tqdm import tqdm
import contextlib


np.random.seed(1000)

# Parameters
sample_sizes = [50, 100, 200, 500]  # Sample sizes
kappa_values = [50, 200]  # Concentration parameters
mu = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # Fixed unit vector in R^2

sign_level = np.array([0.01, 0.05, 0.1])
param_grid = {
    'estimator__min_split_size': [1, 5, 10]
}

# Custom scorer (negative mean squared error)
neg_mse = make_scorer(mse, greater_is_better=False)

# By-blocks execution
n_samples=len(os.listdir(os.path.join(os.getcwd(), 'simulations_sphere/' 'data')))
current_block = int(sys.argv[1])

base = Tree(split_type='2means', mtry=None, impurity_method='cart')
base_forest = BaggedRegressor(estimator=base, n_estimators=200, bootstrap_fraction=1, bootstrap_replace=True, n_jobs=-1, seed=5)

M = Sphere(2)

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# Function defining the regression mean m_0(theta) on S^2
def m_0(theta, mu):
    """
    Compute the regression mean on S^2.
    
    Parameters:
    theta : array-like
        Angles in [0, 2pi) that parameterize the great circle.
    mu : array-like, shape (2,)
        A unit vector defining the orientation of the great circle.
    
    Returns:
    array, shape (n, 3)
        The mean directions on S^2.
    """
    theta = np.asarray(theta)
    mu = np.asarray(mu)
    assert mu.shape == (2,) and np.isclose(np.linalg.norm(mu), 1), "mu must be a unit vector in R^2"
    
    x1 = np.cos(theta)
    x2 = np.sin(theta) * mu[0]
    x3 = np.sin(theta) * mu[1]
    
    return np.column_stack((x1, x2, x3))

# Function to generate vMF samples
def simulate_data(m_0, kappa, mu, theta_samples):
    """
    Generate samples from the von Mises-Fisher distribution on S^2.
    
    Parameters:
    sample_size : int
        Number of samples to generate.
    kappa : float
        Concentration parameter of the vMF distribution.
    mu : array-like, shape (2,)
        The unit vector defining the great circle.
    
    Returns:
    dict
        A dictionary containing input angles and generated samples.
    """
    mean_directions = m_0(theta_samples, mu)  # Compute means on S^2
    
    samples = [vonmises_fisher(mean, kappa).rvs() for mean in mean_directions]
    
    return theta_samples, np.array(samples)


def tune_forest(X, y, forest = base_forest, param_grid=param_grid):
    """ Perform hyperparameter tuning using GridSearchCV. """
    tuned_forest = GridSearchCV(estimator = forest, param_grid=param_grid, scoring=neg_mse, cv=5, n_jobs=-1, verbose=0)
    tuned_forest.fit(X, y)
    return tuned_forest.best_estimator_

# Main task
def task(file) -> None:
    """Processes a single file for sphere data regression."""
    # Check if results file already exists
    results_filename = os.path.join(os.getcwd(), 'simulations_sphere', 'results', f'{file[:-4]}' + '_results.npy')
    if os.path.exists(results_filename):
        print(f"Results file already exists for {file}, skipping...")
        return
    
    with open(os.path.join(os.getcwd(), 'simulations_sphere', 'data', file), 'rb') as f:
        sample = pickle.load(f)
    
    X = np.c_[sample['theta']]
    y = MetricData(M, sample['Y'].reshape(-1, 3))
    kappa = int(file.split('_')[3][5:])

    # Perform hyperparameter tuning
    forest = tune_forest(X, y, base_forest, param_grid)
    oob_quantile = np.percentile(forest.oob_errors(), (1 - np.array([0.01, 0.05, 0.1])) * 100, method='inverted_cdf')

    samp = int(file.split('_')[1][4:])
    N = int(file.split('_')[2][1:])  # Extract N from filename
    kappa = int(file.split('_')[3][5:])  # Extract kappa from filename


    seed = hash((samp, N, kappa)) % (2**32)
    np.random.seed(seed)  # Set seed based on the sample index

    ###########################################################################################################
    # TYPE I COVERAGE RESULTS
    type_i_filename = f'sphere_type_i_N{N}_kappa{kappa}.pkl'
    with open(os.path.join(os.getcwd(), 'simulations_sphere', 'type_i_data', type_i_filename), 'rb') as f:
        type_i_sample = pickle.load(f)
    
    # Randomly select rows from the dataframe
    thetas = type_i_sample['theta'].reshape(-1, 1)
    new_ys = type_i_sample['Y']
    # Predict the new observations
    pb_new_pred = forest.predict(thetas)
    pb_i_cov = (M.d(pb_new_pred, MetricData(M, new_ys)) <= oob_quantile)


############################################################################################################            
    # TYPE II COVERAGE RESULTS
    MC = 1000
    #Generate observations to estimate the probability
    theta_samples = np.array([vonmises_line(kappa = 1).rvs(MC)]).reshape(-1, 1)
    new_thetas, new_ys = simulate_data(m_0 = m_0, kappa = kappa, theta_samples = theta_samples, mu=mu)
    pb_new_pred = forest.predict(new_thetas.reshape(-1, 1))
    pb_ii_cov = np.sum(M.d(MetricData(M, new_ys), pb_new_pred) <= np.tile(oob_quantile, (MC, 1)), axis = 0) / MC
    
############################################################################################################
    # TYPE III COVERAGE RESULTS
    type_iii_filename = f'sphere_type_iii_N{N}_kappa{kappa}.pkl'
    with open(os.path.join(os.getcwd(), 'simulations_sphere', 'type_iii_data', type_iii_filename), 'rb') as f:
        type_iii_sample = pickle.load(f)

    # Use the pre-generated data
    thetas = type_iii_sample['theta'].reshape(-1, 1)
    new_ys = type_iii_sample['Y']

    # Predict the new observations
    pb_new_pred = forest.predict(thetas[0].reshape(-1,1))
    #pb_new_pred = np.tile(pb_new_pred, (MC, 1))  # Repeat the prediction for MC samples
    pb_iii_cov = np.repeat(M.d(pb_new_pred, MetricData(M, new_ys.squeeze()))[:, np.newaxis], 3, axis=1) <= np.tile(oob_quantile, (MC, 1))


###########################################################################################################
    # TYPE IV COVERAGE RESULTS
    theta = np.repeat(vonmises_line.ppf(q=0.25, kappa = 1), MC)
    theta, new_y = simulate_data(m_0 = m_0, kappa = kappa, theta_samples = theta, mu = mu)

    pb_new_pred = forest.predict(theta.reshape(-1,1))
    pb_iv_cov = np.sum(M.d(MetricData(M, new_y), pb_new_pred) <= np.tile(oob_quantile, (MC, 1)), axis = 0) / MC

    # Store results
    results = {
        'i_cov': pb_i_cov,
        'ii_cov': pb_ii_cov,
        'iii_cov': pb_iii_cov,
        'iv_cov': pb_iv_cov,
        'OOB_quantile': oob_quantile,
        }

    results_filename = os.path.join(os.getcwd(), 'simulations_sphere', 'results', f'{file[:-4]}' + '_results.npy')
    np.save(results_filename, results)

file_list = list(filter(
                lambda file: file.endswith(f'block_{current_block}.pkl'), 
                filter(lambda file: file.endswith('.pkl'), os.listdir(os.path.join(os.getcwd(), 'simulations_sphere', 'data/')))
                    )
                )  
total_files = len(file_list)

with tqdm_joblib(tqdm(desc="Percentage of tasks completed:", total = total_files)) as progress_bar:
    Parallel(n_jobs=-1, verbose=2)( delayed(task)(file) for file in file_list)
