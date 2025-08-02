import os
import numpy as np
import pickle
from scipy.stats import vonmises_line, vonmises_fisher

np.random.seed(1000)

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
def simulate_data(kappa, mu, theta_samples):
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

# Parameters
M = 1000  
# Sample sizes
sample_sizes = [50, 100, 200, 500]  # Sample sizes

kappa_values = [50, 200]  # Concentration parameters
mu = np.array([1/np.sqrt(2), 1/np.sqrt(2)])  # Fixed unit vector in R^2

# Create directory for saving samples
save_folder = os.path.join(os.getcwd(), 'simulations_sphere', 'type_i_data')
os.makedirs(save_folder, exist_ok=True)
for sample_size in sample_sizes:
    for kappa in kappa_values:
        theta_samples = vonmises_line(kappa = 1).rvs(M) # Random angles
        theta, Y = simulate_data(kappa, mu, theta_samples)
        filename = os.path.join(save_folder, f'sphere_type_i_N{sample_size}_kappa{kappa}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump({'theta': theta, 'Y': Y}, f)