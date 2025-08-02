import sys, os; sys.path.append(os.path.dirname(os.getcwd())) 
import numpy as np
import pickle
from scipy.stats import beta

np.random.seed(1000)
# Function to simulate regression data
def simulate_data(sigma, X_design, betas):
    Ys = []
    sample_size = X_design.shape[0]

    # Generate the error term epsilon, which follows a normal distribution
    epsilon = np.random.normal(0, sigma, size = sample_size).reshape(sample_size, 1)

    # Step 2: Apply the model transformations
    X_1 = X_design[:, 0]  # X_1 corresponds to the first column (without intercept)
    X_2 = X_design[:, 1]  # X_2 corresponds to the second column
    X_3 = X_design[:, 2]  # X_3 corresponds to the third column

    # Calculate the response vector Y = beta_0 + beta_1*X_1 + beta_2*X_2 + beta_3*X_3 + epsilon
    Ys = betas[0] * X_1 + betas[1] * X_2 + betas[2] * X_3
    # Convert list to array for easier manipulation
    Ys = Ys.reshape(sample_size, 1) + epsilon
    
    return X_design, Ys

# Function to save the simulated regression data
def save_simulated_samples(n_samples, sample_sizes, n_predictors, sigma_values, betas):
    # Create the folder for saving simulations if it doesn't exist
    save_folder = os.path.join(os.getcwd(), 'simulations_euc', 'type_iii_data')
    os.makedirs(save_folder, exist_ok=True)

    M = 1000
    
    for sample_size in sample_sizes:
        for sigma in sigma_values:
            # Simulate the regression data for the given sample size and sigma
            q_25 = 2*np.sqrt(5)*(beta(2,2).ppf(.25)-1/2)
            new_X = np.tile(q_25, (M, n_predictors)).reshape(M, n_predictors)
            X, Y = simulate_data(sigma=sigma, X_design=new_X, betas=betas)
            
            # Define the filename for saving
            filename = os.path.join(save_folder, f'euc_type_iii_N{sample_size}_sigma{np.round(sigma, 1)}.pkl')
            
            # Save the sample using pickle
            with open(filename, 'wb') as f:
                pickle.dump({'X': X, 'Y': Y}, f)

betas = np.array([1, -1, 1])  # Define the true beta values

# Set parameters for the regression scenario
n_predictors = 3  # Number of predictors
# sigma_values = [1/np.sqrt(3), np.sqrt(3)/2, np.sqrt(3)]  # Different sigma values
sigma_values = [np.sqrt(3), np.sqrt(3)/2]
sample_sizes = [50, 100, 200, 500]  # Different sample sizes

# --- Saving the simulated regression samples ---
n_samples = 1000  # Number of samples to simulate and save

# Save the simulated data to files
save_simulated_samples(n_samples, sample_sizes, n_predictors, sigma_values, betas = betas)