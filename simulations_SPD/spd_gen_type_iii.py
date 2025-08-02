import sys, os; sys.path.append(os.path.dirname(os.getcwd())) 
import numpy as np
from scipy.special import digamma
from scipy.stats import wishart
import pickle
from scipy.stats import beta

np.random.seed(1000)

save_folder = os.path.join(os.getcwd(), 'simulations_SPD', 'type_ii_data')
os.makedirs(save_folder, exist_ok=True)

def Sigma_t(t_array, Sigma_array):
    """Provides an array with the matrices given by a regression model that interpolates between four matrices."""  
    """In [0,1], the regression starts with Sigma_1 and then goes to Sigma_2 and Sigma_3."""
    """In general, the regression oscillates between Sigma_1, Sigma_2 and Sigma_3."""
    
    # Define time intervals for interpolation
    t_array = np.array(t_array)
    t_array = t_array[:, None, None]
    # Return the interpolated matrices
    return np.where(np.floor(t_array + 1/2) % 2 == 0, np.cos(np.pi*t_array)**2 * Sigma_array[0] + (1 - np.cos(np.pi*t_array)**2) * Sigma_array[1], 0) + np.where(np.floor(t_array + 1/2) % 2 == 1, (1 - np.cos(np.pi*t_array)**2) * Sigma_array[1] + np.cos(np.pi*t_array)**2 * Sigma_array[2], 0)

def sim_regression_matrices(Sigmas: tuple,
                            t: np.array,
                            df: int=2):
    t = np.array(t)
    q = Sigmas[0].shape[0]
    c_dq = 2 * np.exp((1 / q) * sum( digamma((df - np.arange(1, q + 1) + 1 ) / 2) ))

    sigma_t = Sigma_t(t, Sigmas)
    sample_Y = [wishart( df=df, scale = sigma_t[k]/c_dq ).rvs( size=1 ) for k in range(t.shape[0])]
    return {'t': t, 'y': sample_Y}

# Define the matrices to interpolate 
Sigma_1 = np.array([[1, -0.6],
                  [-0.6, 0.5]])
Sigma_2 = np.array([[1, 0],
                  [0, 1]])
Sigma_3 = np.array([[0.5, 0.4],
                  [0.4, 1]])

M = 1000
sample_sizes = [50, 100, 200, 500]
sample_sizes = [size for size in sample_sizes]
dfs = [5, 15]

# Create directory for saving samples
save_folder = os.path.join(os.getcwd(), 'simulations_SPD', 'type_iii_data')
os.makedirs(save_folder, exist_ok=True)

# For each combination of sample size and degrees of freedom, generate n_samples samples
for sample_size in sample_sizes:
    for df in dfs:
        q_25 = 2*np.sqrt(5)*(beta(2,2).ppf(.25)-1/2)
        ts = np.repeat(q_25, M, axis=0)
        sample = sim_regression_matrices(Sigmas = (Sigma_1, Sigma_2, Sigma_3), 
                                           t = ts,  
                                           df = df)
        
        print(sample)
        filename = os.path.join(save_folder, f'SPD_type_iii_N{sample_size}_df{df}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(sample, f)