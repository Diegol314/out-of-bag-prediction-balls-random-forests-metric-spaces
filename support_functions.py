import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from scipy import stats
from scipy.stats import vonmises_fisher, wishart
from scipy.special import digamma
from scipy.spatial import geometric_slerp
import os
from pathlib import Path
from matplotlib.cm import get_cmap
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# Import pyfrechet for metric spaces
try:
    from pyfrechet.metric_spaces import H2, Sphere, Spheroid, angles_to_spheroid, sphere_to_spheroid, spheroid_to_sphere
    PYFRECHET_AVAILABLE = True
except ImportError:
    PYFRECHET_AVAILABLE = False
    print("Warning: pyfrechet not available. Some plotting functions may not work.")

# Set the root directory
ROOT_DIR = Path(__file__).parent

# ================================
# EUCLIDEAN SPACE FUNCTIONS
# ================================

def all_coverage_results():
    """ Compute empirical OOB_quantile for different confidence levels. """
    coverage_df = pd.DataFrame(columns=['sample_index', 'train_size', 'sigma', 'OOB_quantile', 'pb_i_cov', 'pb_ii_cov', 'pb_iii_cov', 'pb_iv_cov',
     'conf_i_cov', 'conf_ii_cov', 'conf_iii_cov', 'conf_iv_cov'])
    i = 0
    results_path = ROOT_DIR / 'simulations_euc' / 'results'
    for file in os.listdir(results_path):
        if (file.endswith('.npy')):
            i+=1
            infile=open(results_path / file, 'rb')
            result=np.load(infile, allow_pickle=True).item()
            infile.close()
        else:
            continue
        coverage_df = pd.concat([coverage_df, pd.DataFrame({
            'sample_index': int(file.split('_')[1][4:]),
            'train_size': int(file.split('_')[2][1:]),
            'sigma': file.split('_')[3][5:],
            'pb_i_cov': [result['pb_i_cov']],
            'pb_ii_cov': [result['pb_ii_cov']],
            'pb_iii_cov': [result['pb_iii_cov']],
            'pb_iv_cov': [result['pb_iv_cov']],
            'conf_i_cov': [result['conf_i_cov']],
            'conf_ii_cov': [result['conf_ii_cov']],
            'conf_iii_cov': [result['conf_iii_cov']],
            'conf_iv_cov': [result['conf_iv_cov']],
            'pb_mse': [result['pb_mse']],
            'conf_mse': [result['conf_mse']],
            'OOB_quantile': [result['OOB_quantile']],
            'quantile': [result['quantile']],
            'pb_time': [result['pb_time']],
            'conf_time': [result['conf_time']]
        }, index=pd.RangeIndex(0, 1))], ignore_index=True)

    coverage_df['train_size'] = coverage_df['train_size'].astype('category')
    coverage_df['sigma'] = coverage_df.sigma.astype('category')
    return coverage_df


def calculate_type_i_coverage(coverage_df, sample_sizes, sigma_values, B=500, random_seed=1):
    """
    Calculate Type I coverage using bootstrap procedure.
    
    Parameters:
    -----------
    pb_coverage_df : DataFrame
        Coverage results dataframe
    sample_sizes : list
        List of sample sizes to analyze
    kappa_values : list
        List of kappa values to analyze
    B : int
        Number of bootstrap replicates (default: 1000)
    
    Returns:
    --------
    dict : Bootstrap results with means and standard deviations
    """

    if random_seed is not None:
        np.random.seed(random_seed)
    
    pb_diccionario_i = {
        'sigma_0.9': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '500': {'means': np.zeros(3), 'stds': np.zeros(3)}},
        'sigma_1.7': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '500': {'means': np.zeros(3), 'stds': np.zeros(3)}}
    }

    conf_diccionario_i = {
        'sigma_0.9': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '500': {'means': np.zeros(3), 'stds': np.zeros(3)}},
        'sigma_1.7': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '500': {'means': np.zeros(3), 'stds': np.zeros(3)}}
    }

    for N in sample_sizes:
        for sigma in sigma_values:
            # Filter data for current N and sigma
            coverage_df_N_sigma = coverage_df[
                (coverage_df['sigma'] == str(sigma)) & 
                (coverage_df['train_size'] == N)
            ]

            # Get M (number of samples for this N, sigma combination) and assert it is 1000
            M = len(coverage_df_N_sigma)
            #assert M == 1000, f"Expected M to be 1000, but got {M} for N={N}, sigma={sigma}"

            pb_coverages = []
            conf_coverages = []

            for m in range(M):
                # Extract row j_m^(b) of the dataset (training sample)
                training_sample_idx = m
                training_sample = coverage_df_N_sigma.iloc[training_sample_idx]

                # Extract element i_m^(b) from the i_cov column (test pair)
                test_pair_idx = m
                pb_coverage_for_this_pair = training_sample['pb_i_cov'][test_pair_idx, :]  # Shape: (3,)
                conf_coverage_for_this_pair = training_sample['conf_i_cov'][test_pair_idx, :]  # Shape: (3,)

                pb_coverages.append(pb_coverage_for_this_pair)
                conf_coverages.append(conf_coverage_for_this_pair)

            pb_coverages = np.array(pb_coverages)  # Shape: (M, 3)
            conf_coverages = np.array(conf_coverages)  # Shape: (M, 3)

            p_hat_M = np.mean(pb_coverages, axis=0)  # Shape: (3,)
            conf_hat_M = np.mean(conf_coverages, axis=0)  # Shape:


            # Bootstrap procedure
            pb_bootstrap_estimates = []  # Will store B bootstrap estimates
            conf_bootstrap_estimates = []
            
            for b in range(B):
                # Sample M indices for test pairs (i_1^(b), ..., i_M^(b))
                i_indices = np.random.choice(M, size=M, replace=True)
                
                # Sample M indices for training samples (j_1^(b), ..., j_M^(b))
                j_indices = np.random.choice(M, size=M, replace=True)
                
                # Extract bootstrap sample
                pb_bootstrap_coverages = []
                conf_bootstrap_coverages = []

                for m in range(M):
                    # Extract row j_m^(b) of the dataset (training sample)
                    training_sample_idx = j_indices[m]
                    training_sample = coverage_df_N_sigma.iloc[training_sample_idx]

                    # Extract element i_m^(b) from the i_cov column (test pair)
                    test_pair_idx = i_indices[m]
                    pb_coverage_for_this_pair = training_sample['pb_i_cov'][test_pair_idx, :]  # Shape: (3,)
                    conf_coverage_for_this_pair = training_sample['conf_i_cov'][test_pair_idx, :]  # Shape: (3,)

                    pb_bootstrap_coverages.append(pb_coverage_for_this_pair)
                    conf_bootstrap_coverages.append(conf_coverage_for_this_pair)

                # Convert to array and compute mean across the M bootstrap samples
                pb_bootstrap_coverages = np.array(pb_bootstrap_coverages)  # Shape: (M, 3)
                p_hat_M_b = np.mean(pb_bootstrap_coverages, axis=0)  # Shape: (3,)

                conf_bootstrap_coverages = np.array(conf_bootstrap_coverages)  # Shape: (M, 3)
                conf_hat_M_b = np.mean(conf_bootstrap_coverages, axis=0)
                
                pb_bootstrap_estimates.append(p_hat_M_b)
                conf_bootstrap_estimates.append(conf_hat_M_b)

            # Convert bootstrap estimates to array
            pb_bootstrap_estimates = np.array(pb_bootstrap_estimates)  # Shape: (B, 3)
            conf_bootstrap_estimates = np.array(conf_bootstrap_estimates)  # Shape: (B, 3


            # Compute overall mean and standard deviation
            #p_bar_M = np.mean(pb_bootstrap_estimates, axis=0)  # Shape: (3,)
            sigma_boot = np.std(pb_bootstrap_estimates, axis=0, ddof=1)  # Shape: (3,)

            #conf_bar_M = np.mean(conf_bootstrap_estimates, axis=0)  # Shape: (3,)
            conf_sigma_boot = np.std(conf_bootstrap_estimates, axis=0, ddof=1)  # Shape: (3,)

            # Store results
            sigma_key = f'sigma_{sigma}'
            N_key = str(N)
            pb_diccionario_i[sigma_key][N_key]['means'] = p_hat_M
            pb_diccionario_i[sigma_key][N_key]['stds'] = sigma_boot

            conf_diccionario_i[sigma_key][N_key]['means'] = conf_hat_M
            conf_diccionario_i[sigma_key][N_key]['stds'] = conf_sigma_boot
            # Print results
    
    return pb_diccionario_i, conf_diccionario_i


def calculate_type_iii_coverage(coverage_df, sample_sizes, sigma_values, B=500, random_seed=1):
    """
    Calculate Type III coverage using bootstrap procedure.

    Parameters:
    -----------
    pb_coverage_df : DataFrame
        Coverage results dataframe
    sample_sizes : list
        List of sample sizes to analyze
    kappa_values : list
        List of kappa values to analyze
    B : int
        Number of bootstrap replicates (default: 1000)
    
    Returns:
    --------
    dict : Bootstrap results with means and standard deviations
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    pb_diccionario_iii = {
        'sigma_0.9': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '500': {'means': np.zeros(3), 'stds': np.zeros(3)}},
        'sigma_1.7': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '500': {'means': np.zeros(3), 'stds': np.zeros(3)}}
    }

    conf_diccionario_iii = {
        'sigma_0.9': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '500': {'means': np.zeros(3), 'stds': np.zeros(3)}},
        'sigma_1.7': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '500': {'means': np.zeros(3), 'stds': np.zeros(3)}}
    }

    for N in sample_sizes:
        for sigma in sigma_values:
            # Filter data for current N and sigma
            coverage_df_N_sigma = coverage_df[
                (coverage_df['sigma'] == str(sigma)) & 
                (coverage_df['train_size'] == N)
            ]

            # Get M (number of samples for this N, sigma combination) and assert it is 1000
            M = len(coverage_df_N_sigma)
            #assert M == 1000, f"Expected M to be 1000, but got {M} for N={N}, sigma={sigma}"

            pb_coverages = []
            conf_coverages = []

            for m in range(M):
                # Extract row j_m^(b) of the dataset (training sample)
                training_sample_idx = m
                training_sample = coverage_df_N_sigma.iloc[training_sample_idx]

                # Extract element i_m^(b) from the i_cov column (test pair)
                test_pair_idx = m
                pb_coverage_for_this_pair = training_sample['pb_iii_cov'][test_pair_idx, :]  # Shape: (3,)
                conf_coverage_for_this_pair = training_sample['conf_iii_cov'][test_pair_idx, :]  # Shape: (3,)

                pb_coverages.append(pb_coverage_for_this_pair)
                conf_coverages.append(conf_coverage_for_this_pair)

            pb_coverages = np.array(pb_coverages)  # Shape: (M, 3)
            conf_coverages = np.array(conf_coverages)  # Shape: (M, 3)

            p_hat_M = np.mean(pb_coverages, axis=0)  # Shape: (3,)
            conf_hat_M = np.mean(conf_coverages, axis=0)  # Shape:


            # Bootstrap procedure
            pb_bootstrap_estimates = []  # Will store B bootstrap estimates
            conf_bootstrap_estimates = []
            
            for b in range(B):
                # Sample M indices for test pairs (i_1^(b), ..., i_M^(b))
                i_indices = np.random.choice(M, size=M, replace=True)
                
                # Sample M indices for training samples (j_1^(b), ..., j_M^(b))
                j_indices = np.random.choice(M, size=M, replace=True)
                
                # Extract bootstrap sample
                pb_bootstrap_coverages = []
                conf_bootstrap_coverages = []

                for m in range(M):
                    # Extract row j_m^(b) of the dataset (training sample)
                    training_sample_idx = j_indices[m]
                    training_sample = coverage_df_N_sigma.iloc[training_sample_idx]

                    # Extract element i_m^(b) from the i_cov column (test pair)
                    test_pair_idx = i_indices[m]
                    pb_coverage_for_this_pair = training_sample['pb_iii_cov'][test_pair_idx, :]  # Shape: (3,)
                    conf_coverage_for_this_pair = training_sample['conf_iii_cov'][test_pair_idx, :]  # Shape: (3,)

                    pb_bootstrap_coverages.append(pb_coverage_for_this_pair)
                    conf_bootstrap_coverages.append(conf_coverage_for_this_pair)

                # Convert to array and compute mean across the M bootstrap samples
                pb_bootstrap_coverages = np.array(pb_bootstrap_coverages)  # Shape: (M, 3)
                p_hat_M_b = np.mean(pb_bootstrap_coverages, axis=0)  # Shape: (3,)

                conf_bootstrap_coverages = np.array(conf_bootstrap_coverages)  # Shape: (M, 3)
                conf_hat_M_b = np.mean(conf_bootstrap_coverages, axis=0)
                
                pb_bootstrap_estimates.append(p_hat_M_b)
                conf_bootstrap_estimates.append(conf_hat_M_b)

            # Convert bootstrap estimates to array
            pb_bootstrap_estimates = np.array(pb_bootstrap_estimates)  # Shape: (B, 3)
            conf_bootstrap_estimates = np.array(conf_bootstrap_estimates)  # Shape: (B, 3


            # Compute overall mean and standard deviation
            #p_bar_M = np.mean(pb_bootstrap_estimates, axis=0)  # Shape: (3,)
            sigma_boot = np.std(pb_bootstrap_estimates, axis=0, ddof=1)  # Shape: (3,)

            #conf_bar_M = np.mean(conf_bootstrap_estimates, axis=0)  # Shape: (3,)
            conf_sigma_boot = np.std(conf_bootstrap_estimates, axis=0, ddof=1)  # Shape: (3,)

            # Store results
            sigma_key = f'sigma_{sigma}'
            N_key = str(N)
            pb_diccionario_iii[sigma_key][N_key]['means'] = p_hat_M
            pb_diccionario_iii[sigma_key][N_key]['stds'] = sigma_boot

            conf_diccionario_iii[sigma_key][N_key]['means'] = conf_hat_M
            conf_diccionario_iii[sigma_key][N_key]['stds'] = conf_sigma_boot

    return pb_diccionario_iii, conf_diccionario_iii

def format_cell(value, target_coverage=0.95, n_trials=1000, confidence_level=0.95):
    """
    Format cell with underline if mean coverage falls outside binomial proportion CI
    
    Parameters:
    value: string in format "mean (std)"
    target_coverage: expected coverage (0.99, 0.95, or 0.90)
    n_trials: number of trials (1000)
    confidence_level: confidence level for CI (0.95 for 95%)
    """
    pb_mean, pb_std = value.split(" ")
    mean_val = float(pb_mean) / 100.0  # Convert percentage back to proportion
    pb_mean = f"{float(pb_mean):.1f}"
    pb_std = pb_std.strip("()")
    pb_std = f"({float(pb_std):.2f})"
    
    # Calculate expected number of successes for target coverage
    expected_successes = int(target_coverage * n_trials)
    
    # Use scipy.stats.binomtest to get proportion confidence interval
    binom_result = stats.binomtest(expected_successes, n_trials)
    lower_bound, upper_bound = binom_result.proportion_ci(confidence_level=confidence_level)
    
    # Check if mean falls outside the confidence interval
    if mean_val < lower_bound or mean_val > upper_bound:
        # Underline the mean value in LaTeX
        pb_mean = f"\\underline{{{pb_mean}}}"
    
    return f"{pb_mean} {pb_std}"

def euclidean_type_analysis(coverage_df, coverage_type='i'):
    """Generate Type I or Type III tables"""
    sample_sizes = [50, 100, 200, 500]
    sigma_values = [0.9, 1.7]
    
    # Calculate bootstrap statistics based on coverage type
    if coverage_type == 'i':
        pb_diccionario, conf_diccionario = calculate_type_i_coverage(
            coverage_df, sample_sizes, sigma_values, B=500, random_seed=1
        )
    elif coverage_type == 'iii':
        pb_diccionario, conf_diccionario = calculate_type_iii_coverage(
            coverage_df, sample_sizes, sigma_values, B=500, random_seed=1
        )
    else:
        raise ValueError("coverage_type must be 'i' or 'iii'")
    
    # Create prediction balls table
    pb_rows = []
    index = []
    for sigma in sigma_values:
        for N in sample_sizes:
            pb_row = []
            pb_means = pb_diccionario[f'sigma_{sigma}'][str(N)]['means']
            pb_stds = pb_diccionario[f'sigma_{sigma}'][str(N)]['stds']
            # Format as "mean (std)"
            pb_formatted_values = [f"{100*pb_means[i]:.1f} ({100*pb_stds[i]:.2f})" for i in range(3)]
            pb_row.extend(pb_formatted_values)
            pb_rows.append(pb_row)
            index.append((f"{sigma}", f"{N}"))

    # MultiIndex for rows and columns
    row_index = pd.MultiIndex.from_tuples(index, names=["sigma", "N"])
    col_index = pd.MultiIndex.from_product(
        [["0.01", "0.05", "0.1"]],
        names=["Significance Level"]
    )

    # Create the DataFrame
    pb_df = pd.DataFrame(pb_rows, index=row_index, columns=col_index)
    
    # Apply formatting with column-specific target coverage
    target_coverages = [0.99, 0.95, 0.90]
    pb_latex = pb_df.copy()
    for col_idx, col in enumerate(pb_df.columns):
        target_coverage = target_coverages[col_idx]
        for row_idx in pb_df.index:
            pb_latex.loc[row_idx, col] = format_cell(
                pb_df.loc[row_idx, col], 
                target_coverage=target_coverage, 
                n_trials=1000
            )
    
    # Create split-conformal table
    conf_rows = []
    index = []
    for sigma in sigma_values:
        for N in sample_sizes:
            conf_row = []
            conf_means = conf_diccionario[f'sigma_{sigma}'][str(N)]['means']
            conf_stds = conf_diccionario[f'sigma_{sigma}'][str(N)]['stds']
            # Format as "mean (std)"
            conf_formatted_values = [f"{100*conf_means[i]:.1f} ({100*conf_stds[i]:.2f})" for i in range(3)]
            conf_row.extend(conf_formatted_values)
            conf_rows.append(conf_row)
            index.append((f"{sigma}", f"{N}"))

    conf_df = pd.DataFrame(conf_rows, index=row_index, columns=col_index)
    
    # Apply formatting for conformal
    conf_latex = conf_df.copy()
    for col_idx, col in enumerate(conf_df.columns):
        target_coverage = target_coverages[col_idx]
        for row_idx in conf_df.index:
            conf_latex.loc[row_idx, col] = format_cell(
                conf_df.loc[row_idx, col], 
                target_coverage=target_coverage, 
                n_trials=1000
            )
    
    return pb_latex, conf_latex

def euclidean_type_i_analysis(coverage_df):
    """Generate Type I tables"""
    return euclidean_type_analysis(coverage_df, coverage_type='i')

def euclidean_type_iii_analysis(coverage_df):
    """Generate Type III tables"""
    return euclidean_type_analysis(coverage_df, coverage_type='iii')

def euclidean_type_ii_analysis(coverage_df, save_individual=True):
    """Generate Type II plots"""
    import matplotlib.lines as mlines
    
    train_sizes = [50, 100, 200, 500]
    alpha_levels = [0.01, 0.05, 0.1]
    sigma_values = [0.9, 1.7]
    
    # Create 2x3 subplot figure
    
    for sigma in sigma_values:
        fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="white")

        if sigma == 0.9:
            print(" Sigma = √3/2")
        elif sigma == 1.7:
            print(" Sigma = √3")
        # Create separate dataframes for each alpha level as in original
        ii_coverage_dfs = {}
        
        for alpha_idx, alpha_level in enumerate(alpha_levels):
            sigma_data = coverage_df[coverage_df['sigma'] == str(sigma)].copy()
            sigma_data['pb_ii_cov_alpha'] = sigma_data['pb_ii_cov'].apply(lambda x: x[alpha_idx])
            sigma_data['conf_ii_cov_alpha'] = sigma_data['conf_ii_cov'].apply(lambda x: x[alpha_idx])
            ii_coverage_dfs[alpha_level] = sigma_data
        
        # Create plots for each alpha level 
        for alpha_idx, alpha_level in enumerate(alpha_levels):
            sigma_data = ii_coverage_dfs[alpha_level]
            
            ax = axes[alpha_idx]

            # Extract data for each training size
            pb_sigma_boxplot_data = [sigma_data[sigma_data['train_size'] == size]['pb_ii_cov_alpha'].values for size in train_sizes]
            conf_sigma_boxplot_data = [sigma_data[sigma_data['train_size'] == size]['conf_ii_cov_alpha'].values for size in train_sizes]

            # Create boxplots with adjusted positions
            pb_positions_sigma = np.array(range(len(train_sizes))) - 0.2
            conf_positions_sigma = np.array(range(len(train_sizes))) + 0.2

            ax.boxplot(pb_sigma_boxplot_data, positions=pb_positions_sigma, widths=0.3, notch=False, 
                       boxprops=dict(color="#000000", linestyle='-', linewidth=1.5), 
                       whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                       flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                       medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)
                       
            ax.boxplot(conf_sigma_boxplot_data, positions=conf_positions_sigma, widths=0.3, notch=False, 
                       boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                       whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                       flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                       medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)

            # Scatter plot
            pb_palette_sigma = ['#ee6100', 'g', 'b', 'y']
            conf_palette_sigma = ['#ee6100', 'g', 'b', 'y']

            for i, size in enumerate(train_sizes):
                pb_xs_sigma = np.random.normal(pb_positions_sigma[i], 0.04, len(pb_sigma_boxplot_data[i]))
                conf_xs_sigma = np.random.normal(conf_positions_sigma[i], 0.04, len(conf_sigma_boxplot_data[i]))

                ax.scatter(pb_xs_sigma, pb_sigma_boxplot_data[i], alpha=0.2, color = pb_palette_sigma[i], label='Prediction balls')
                ax.scatter(conf_xs_sigma, conf_sigma_boxplot_data[i], alpha=0.2, color = conf_palette_sigma[i], marker='^', label='Split-conformal')

            sns.despine(bottom=True)  # Remove right and top axis lines
            sns.set_style("whitegrid")
            ax.set_xticks(range(len(train_sizes)))
            ax.set_xticklabels([str(size) for size in train_sizes], fontsize=17)

            if alpha_level == 0.01:
                ax.set_ylim(0.86, 1.001)
            elif alpha_level == 0.05:
                ax.set_ylim(0.75, 1.001)
            else:
                ax.set_ylim(0.5, 1)

            ax.set_xlabel('Training sample size', fontsize=17)
            ax.set_ylabel('Coverage', fontsize=17)
            ax.tick_params(labelsize=17)
            ax.axhline(y=1-alpha_level, color='black', linestyle='dashed')
            ax.grid(False)

            legend_handles = []
            legend_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='none', markersize=10, label='OOB prediction balls'))
            legend_handles.append(mlines.Line2D([], [], color='gray', marker='^', linestyle='none', markersize=10, label='Split-conformal'))

            ax.legend(handles=legend_handles, loc='lower right', fontsize=13)

        plt.show()
        
        # Save individual plots if requested
        if save_individual:
            output_dir = ROOT_DIR / "results_plots"
            output_dir.mkdir(exist_ok=True)
            
            # Create individual plots for each alpha level
            for alpha_idx, alpha_level in enumerate(alpha_levels):
                sigma_data = ii_coverage_dfs[alpha_level]
                
                # Create individual figure
                fig_individual, ax = plt.subplots(1, 1, figsize=(7, 7), facecolor="white")
                
                # Extract data for each training size
                pb_sigma_boxplot_data = [sigma_data[sigma_data['train_size'] == size]['pb_ii_cov_alpha'].values for size in train_sizes]
                conf_sigma_boxplot_data = [sigma_data[sigma_data['train_size'] == size]['conf_ii_cov_alpha'].values for size in train_sizes]

                # Create boxplots with adjusted positions
                pb_positions_sigma = np.array(range(len(train_sizes))) - 0.2
                conf_positions_sigma = np.array(range(len(train_sizes))) + 0.2

                ax.boxplot(pb_sigma_boxplot_data, positions=pb_positions_sigma, widths=0.3, notch=False, 
                           boxprops=dict(color="#000000", linestyle='-', linewidth=1.5), 
                           whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                           flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                           medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)
                           
                ax.boxplot(conf_sigma_boxplot_data, positions=conf_positions_sigma, widths=0.3, notch=False, 
                           boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                           whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                           flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                           medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)

                # Scatter plot
                pb_palette_sigma = ['#ee6100', 'g', 'b', 'y']
                conf_palette_sigma = ['#ee6100', 'g', 'b', 'y']

                for i, size in enumerate(train_sizes):
                    pb_xs_sigma = np.random.normal(pb_positions_sigma[i], 0.04, len(pb_sigma_boxplot_data[i]))
                    conf_xs_sigma = np.random.normal(conf_positions_sigma[i], 0.04, len(conf_sigma_boxplot_data[i]))

                    ax.scatter(pb_xs_sigma, pb_sigma_boxplot_data[i], alpha=0.2, color = pb_palette_sigma[i])
                    ax.scatter(conf_xs_sigma, conf_sigma_boxplot_data[i], alpha=0.2, color = conf_palette_sigma[i], marker='^')

                sns.despine(bottom=True)  # Remove right and top axis lines
                sns.set_style("whitegrid")
                ax.set_xticks(range(len(train_sizes)))
                ax.set_xticklabels([str(size) for size in train_sizes], fontsize=17)

                if alpha_level == 0.01:
                    ax.set_ylim(0.86, 1.001)
                elif alpha_level == 0.05:
                    ax.set_ylim(0.75, 1.001)
                else:
                    ax.set_ylim(0.5, 1)

                ax.set_xlabel('Training sample size', fontsize=17)
                ax.set_ylabel('Coverage', fontsize=17)
                ax.tick_params(labelsize=17)
                ax.axhline(y=1-alpha_level, color='black', linestyle='dashed')
                ax.grid(False)

                legend_handles = []
                legend_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='none', markersize=10, label='OOB prediction balls'))
                legend_handles.append(mlines.Line2D([], [], color='gray', marker='^', linestyle='none', markersize=10, label='Split-conformal'))

                ax.legend(handles=legend_handles, loc='lower right', fontsize=13)
                
                fig_individual.tight_layout()
                
                # Save the individual plot
                filename = output_dir / f'euclidean_type_ii_coverage_sigma_{str(sigma).replace(".", "")}_alpha_{str(alpha_level)[2:]}.png'
                fig_individual.savefig(filename, bbox_inches='tight', format='png', dpi=75, transparent=True)
                plt.close(fig_individual)  # Close individual figure to free memory

    fig.tight_layout()


def euclidean_mse_analysis(coverage_df):
    """Calculate MSE comparison"""
    sample_sizes = [50, 100, 200, 500]
    sigma_values = [0.9, 1.7]
    
    results = []
    for sigma in sigma_values:
        for N in sample_sizes:
            subset = coverage_df[(coverage_df['sigma'] == str(sigma)) & (coverage_df['train_size'] == N)]
            
            if len(subset) > 0:
                pb_mse_mean = subset['pb_mse'].apply(lambda x: np.mean(x) if hasattr(x, '__len__') else x).mean()
                conf_mse_mean = subset['conf_mse'].apply(lambda x: np.mean(x) if hasattr(x, '__len__') else x).mean()
                
                relative_improvement = ((conf_mse_mean - pb_mse_mean) / conf_mse_mean) * 100
                
                results.append({
                    'train_size': N,
                    'sigma': sigma,
                    'pb_mse': pb_mse_mean,
                    'conf_mse': conf_mse_mean,
                    'relative_improvement': relative_improvement
                })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.set_index(['train_size', 'sigma'])
    return result_df

# ================================
# SPHERE SPACE FUNCTIONS
# ================================

def sphere_H2_coverage_results(space='sphere'):
    """ Compute empirical OOB_quantile for different confidence levels for Sphere data. """
    coverage_df = pd.DataFrame(columns=['sample_index', 'train_size', 'kappa', 'OOB_quantile', 'i_cov', 'ii_cov', 'iii_cov', 'iv_cov'])
    i = 0
    if space not in ['sphere', 'hyperboloid']:
        raise ValueError("space must be 'sphere' or 'hyperboloid'")
    elif space == 'sphere':
        results_path = ROOT_DIR / 'simulations_sphere' / 'results'
    else:  # space == 'hyperboloid'
        results_path = ROOT_DIR / 'simulations_H2' / 'results'
    for file in os.listdir(results_path):
        if (file.endswith('.npy')):
            i+=1
            infile=open(results_path / file, 'rb')
            result=np.load(infile, allow_pickle=True).item()
            infile.close()
        else:
            continue
        coverage_df = pd.concat([coverage_df, pd.DataFrame({
            'sample_index': int(file.split('_')[1][4:]),
            'train_size': int(file.split('_')[2][1:]),
            'kappa': file.split('_')[3][5:],
            'i_cov': [result['i_cov']],
            'ii_cov': [result['ii_cov']],
            'iii_cov': [result['iii_cov']],
            'iv_cov': [result['iv_cov']],
            'OOB_quantile': [result['OOB_quantile']],
        }, index=pd.RangeIndex(0, 1))], ignore_index=True)

    coverage_df['train_size'] = coverage_df['train_size'].astype('category')
    coverage_df['kappa'] = coverage_df.kappa.astype('category')
    return coverage_df

def calculate_type_i_coverage_sphere_H2(pb_coverage_df, sample_sizes, kappa_values, B=500, random_seed=1):
    """
    Calculate Type I coverage using bootstrap procedure for sphere or hyperboloid.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    pb_diccionario_i = {
        'kappa_50': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '500': {'means': np.zeros(3), 'stds': np.zeros(3)}},
        'kappa_200': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '500': {'means': np.zeros(3), 'stds': np.zeros(3)}}
    }

    for N in sample_sizes:
        for kappa in kappa_values:
            # Filter data for current N and kappa
            pb_coverage_df_N_kappa = pb_coverage_df[
                (pb_coverage_df['kappa'] == str(kappa)) & 
                (pb_coverage_df['train_size'] == N)
            ]

            # Get M (number of samples for this N, kappa combination) and assert it is 1000
            M = len(pb_coverage_df_N_kappa)

            # ONLY FOR STANDARD DEVIATIONS
            # Bootstrap procedure
            pb_bootstrap_estimates = []  # Will store B bootstrap estimates
            for b in range(B):
                # Sample M indices for test pairs (i_1^(b), ..., i_M^(b))
                i_indices = np.random.choice(M, size=M, replace=True)
                
                # Sample M indices for training samples (j_1^(b), ..., j_M^(b))
                j_indices = np.random.choice(M, size=M, replace=True)
                
                # Extract bootstrap sample
                pb_bootstrap_coverages = []

                for m in range(M):
                    # Extract row j_m^(b) of the dataset (training sample)
                    training_sample_idx = j_indices[m]
                    training_sample = pb_coverage_df_N_kappa.iloc[training_sample_idx]

                    # Extract element i_m^(b) from the i_cov column (test pair)
                    test_pair_idx = i_indices[m]
                    pb_coverage_for_this_pair = training_sample['i_cov'][test_pair_idx, :]  # Shape: (3,)

                    pb_bootstrap_coverages.append(pb_coverage_for_this_pair)

                # Convert to array and compute mean across the M bootstrap samples
                pb_bootstrap_coverages = np.array(pb_bootstrap_coverages)  # Shape: (M, 3)
                p_hat_M_b = np.mean(pb_bootstrap_coverages, axis=0)  # Shape: (3,)
                # Compute overall mean and standard deviation
                #p_bar_M = np.mean(pb_bootstrap_estimates, axis=0)  # Shape: (3,)
                pb_bootstrap_estimates.append(p_hat_M_b)

            pb_bootstrap_estimates = np.array(pb_bootstrap_estimates)  # Shape: (B, 3)

            sigma_boot = np.std(pb_bootstrap_estimates, axis=0, ddof=1)  # Shape: (3,)

            pb_coverages = []  # Will store bootstrap coverages
            for m in range(M):
                # Extract row j_m^(b) of the dataset (training sample)
                training_sample_idx = m
                training_sample = pb_coverage_df_N_kappa.iloc[training_sample_idx]

                # Extract element i_m^(b) from the i_cov column (test pair)
                test_pair_idx = m
                pb_coverage_for_this_pair = training_sample['i_cov'][test_pair_idx, :]  # Shape: (3,)

                pb_coverages.append(pb_coverage_for_this_pair)

            # Convert to array and compute mean across the M bootstrap samples
            pb_coverages = np.array(pb_coverages)  # Shape: (M, 3)
            p_hat_M = np.mean(pb_coverages, axis=0)  # Shape: (3,)

            # Store results
            kappa_key = f'kappa_{kappa}'
            N_key = str(N)
            pb_diccionario_i[kappa_key][N_key]['means'] = p_hat_M
            pb_diccionario_i[kappa_key][N_key]['stds'] = sigma_boot

    
    return pb_diccionario_i

def calculate_type_iii_coverage_sphere_H2(pb_coverage_df, sample_sizes, kappa_values, B=500, random_seed=1):
    """
    Calculate Type III coverage using bootstrap procedure for sphere or hyperboloid.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    pb_diccionario_iii = {
        'kappa_50': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '500': {'means': np.zeros(3), 'stds': np.zeros(3)}},
        'kappa_200': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '500': {'means': np.zeros(3), 'stds': np.zeros(3)}}
    }

    for N in sample_sizes:
        for kappa in kappa_values:
            # Filter data for current N and kappa
            pb_coverage_df_N_kappa = pb_coverage_df[
                (pb_coverage_df['kappa'] == str(kappa)) & 
                (pb_coverage_df['train_size'] == N)
            ]

            # Get M (number of samples for this N, kappa combination)
            M = len(pb_coverage_df_N_kappa)

            # Bootstrap procedure
            pb_bootstrap_estimates = []  # Will store B bootstrap estimates
            
            for b in range(B):
                # Sample M indices for test pairs (i_1^(b), ..., i_M^(b))
                i_indices = np.random.choice(M, size=M, replace=True)
                
                # Sample M indices for training samples (j_1^(b), ..., j_M^(b))
                j_indices = np.random.choice(M, size=M, replace=True)
                
                # Extract bootstrap sample
                pb_bootstrap_coverages = []

                for m in range(M):
                    # Extract row j_m^(b) of the dataset (training sample)
                    training_sample_idx = j_indices[m]
                    training_sample = pb_coverage_df_N_kappa.iloc[training_sample_idx]

                    # Extract element i_m^(b) from the iii_cov column (test pair)
                    test_pair_idx = i_indices[m]
                    pb_coverage_for_this_pair = training_sample['iii_cov'][test_pair_idx, :]  # Shape: (3,)

                    pb_bootstrap_coverages.append(pb_coverage_for_this_pair)

                # Convert to array and compute mean across the M bootstrap samples
                pb_bootstrap_coverages = np.array(pb_bootstrap_coverages)  # Shape: (M, 3)
                p_hat_M_b = np.mean(pb_bootstrap_coverages, axis=0)  # Shape: (3,)
                pb_bootstrap_estimates.append(p_hat_M_b)

            pb_bootstrap_estimates = np.array(pb_bootstrap_estimates)  # Shape: (B, 3)
            sigma_boot = np.std(pb_bootstrap_estimates, axis=0, ddof=1)  # Shape: (3,)

            # Calculate actual coverage for this combination
            pb_coverages = []
            for m in range(M):
                # Extract row m of the dataset (training sample)
                training_sample_idx = m
                training_sample = pb_coverage_df_N_kappa.iloc[training_sample_idx]

                # Extract element m from the iii_cov column (test pair)
                test_pair_idx = m
                pb_coverage_for_this_pair = training_sample['iii_cov'][test_pair_idx, :]  # Shape: (3,)

                pb_coverages.append(pb_coverage_for_this_pair)

            # Convert to array and compute mean across the M samples
            pb_coverages = np.array(pb_coverages)  # Shape: (M, 3)
            p_hat_M = np.mean(pb_coverages, axis=0)  # Shape: (3,)

            # Store results
            kappa_key = f'kappa_{kappa}'
            N_key = str(N)
            pb_diccionario_iii[kappa_key][N_key]['means'] = p_hat_M
            pb_diccionario_iii[kappa_key][N_key]['stds'] = sigma_boot

    return pb_diccionario_iii

def sphere_H2_type_i_analysis(pb_coverage_df):
    """Generate Type I tables"""
    sample_sizes = [50, 100, 200, 500]
    kappa_values = [50, 200]
    
    # Calculate bootstrap statistics
    pb_diccionario_i = calculate_type_i_coverage_sphere_H2(
        pb_coverage_df, sample_sizes, kappa_values, B=500, random_seed=1
    )
    
    # Create prediction balls table
    pb_rows = []
    index = []
    for kappa in kappa_values:
        for N in sample_sizes:
            pb_row = []
            pb_means = pb_diccionario_i[f'kappa_{kappa}'][str(N)]['means']
            pb_stds = pb_diccionario_i[f'kappa_{kappa}'][str(N)]['stds']
            # Format as "mean (std)"
            pb_formatted_values = [f"{100*pb_means[i]:.1f} ({100*pb_stds[i]:.2f})" for i in range(3)]
            pb_row.extend(pb_formatted_values)
            pb_rows.append(pb_row)
            index.append((f"{kappa}", f"{N}"))

    # MultiIndex for rows and columns
    row_index = pd.MultiIndex.from_tuples(index, names=["kappa", "N"])
    col_index = pd.MultiIndex.from_product(
        [["0.01", "0.05", "0.1"]],
        names=["Significance Level"]
    )

    # Create the DataFrame
    pb_df = pd.DataFrame(pb_rows, index=row_index, columns=col_index)
    
    # Apply formatting with column-specific target coverage
    target_coverages = [0.99, 0.95, 0.90]
    pb_latex = pb_df.copy()
    for col_idx, col in enumerate(pb_df.columns):
        target_coverage = target_coverages[col_idx]
        for row_idx in pb_df.index:
            pb_latex.loc[row_idx, col] = format_cell(
                pb_df.loc[row_idx, col], 
                target_coverage=target_coverage, 
                n_trials=1000
            )
    
    return pb_latex

def sphere_H2_type_iii_analysis(pb_coverage_df):
    """Generate Type III tables with proper formatting"""
    sample_sizes = [50, 100, 200, 500]
    kappa_values = [50, 200]
    
    # Calculate bootstrap statistics
    pb_diccionario_iii = calculate_type_iii_coverage_sphere_H2(
        pb_coverage_df, sample_sizes, kappa_values, B=500, random_seed=1
    )
    
    # Create prediction balls table
    pb_rows = []
    index = []
    for kappa in kappa_values:
        for N in sample_sizes:
            pb_row = []
            pb_means = pb_diccionario_iii[f'kappa_{kappa}'][str(N)]['means']
            pb_stds = pb_diccionario_iii[f'kappa_{kappa}'][str(N)]['stds']
            # Format as "mean (std)" - multiply by 100 for percentage
            pb_formatted_values = [f"{100*pb_means[i]:.1f} ({100*pb_stds[i]:.2f})" for i in range(3)]
            pb_row.extend(pb_formatted_values)
            pb_rows.append(pb_row)
            index.append((f"{kappa}", f"{N}"))

    # MultiIndex for rows and columns
    row_index = pd.MultiIndex.from_tuples(index, names=["kappa", "N"])
    col_index = pd.MultiIndex.from_product(
        [["0.01", "0.05", "0.1"]],
        names=["Significance Level"]
    )

    # Create the DataFrame
    pb_df = pd.DataFrame(pb_rows, index=row_index, columns=col_index)
    
    # Apply formatting with column-specific target coverage for Type III
    # Type III uses different target coverages - need to check what these should be
    target_coverages = [0.99, 0.95, 0.90]  # Same as Type I for now
    pb_latex = pb_df.copy()
    for col_idx, col in enumerate(pb_df.columns):
        target_coverage = target_coverages[col_idx]
        for row_idx in pb_df.index:
            pb_latex.loc[row_idx, col] = format_cell(
                pb_df.loc[row_idx, col], 
                target_coverage=target_coverage, 
                n_trials=1000
            )
    
    return pb_latex
    """Generate Type I tables"""
    sample_sizes = [50, 100, 200, 500]
    kappa_values = [50, 200]
    
    # Calculate bootstrap statistics
    pb_diccionario_i = calculate_type_i_coverage_sphere_H2(
        pb_coverage_df, sample_sizes, kappa_values, B=500, random_seed=1
    )
    
    # Create prediction balls table
    pb_rows = []
    index = []
    for kappa in kappa_values:
        for N in sample_sizes:
            pb_row = []
            pb_means = pb_diccionario_i[f'kappa_{kappa}'][str(N)]['means']
            pb_stds = pb_diccionario_i[f'kappa_{kappa}'][str(N)]['stds']
            # Format as "mean (std)"
            pb_formatted_values = [f"{100*pb_means[i]:.1f} ({100*pb_stds[i]:.2f})" for i in range(3)]
            pb_row.extend(pb_formatted_values)
            pb_rows.append(pb_row)
            index.append((f"{kappa}", f"{N}"))

    # MultiIndex for rows and columns
    row_index = pd.MultiIndex.from_tuples(index, names=["kappa", "N"])
    col_index = pd.MultiIndex.from_product(
        [["0.01", "0.05", "0.1"]],
        names=["Significance Level"]
    )

    # Create the DataFrame
    pb_df = pd.DataFrame(pb_rows, index=row_index, columns=col_index)
    
    # Apply formatting with column-specific target coverage
    target_coverages = [0.99, 0.95, 0.90]
    pb_latex = pb_df.copy()
    for col_idx, col in enumerate(pb_df.columns):
        target_coverage = target_coverages[col_idx]
        for row_idx in pb_df.index:
            pb_latex.loc[row_idx, col] = format_cell(
                pb_df.loc[row_idx, col], 
                target_coverage=target_coverage, 
                n_trials=1000
            )
    
    return pb_latex

def sphere_H2_type_ii_analysis(pb_coverage_df, save_individual=True):
    """Generate Type II plots exactly"""
    train_sizes = [50, 100, 200, 500]
    alpha_levels = [0.01, 0.05, 0.1]
    kappa_values = ['50', '200']
    
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    
    # Create separate dataframes for each kappa and alpha level
    ii_pb_coverage_df_kappa_50_alpha_01 = pb_coverage_df[pb_coverage_df['kappa'] == '50'].copy()
    ii_pb_coverage_df_kappa_50_alpha_05 = pb_coverage_df[pb_coverage_df['kappa'] == '50'].copy()
    ii_pb_coverage_df_kappa_50_alpha_1  = pb_coverage_df[pb_coverage_df['kappa'] == '50'].copy()

    ii_pb_coverage_df_kappa_50_alpha_01['ii_cov']  = pb_coverage_df[pb_coverage_df['kappa'] == '50']['ii_cov'].apply(lambda x: x[0])
    ii_pb_coverage_df_kappa_50_alpha_05['ii_cov']  = pb_coverage_df[pb_coverage_df['kappa'] == '50']['ii_cov'].apply(lambda x: x[1])
    ii_pb_coverage_df_kappa_50_alpha_1['ii_cov']  = pb_coverage_df[pb_coverage_df['kappa'] == '50']['ii_cov'].apply(lambda x: x[2])

    ii_pb_coverage_df_kappa_200_alpha_01 = pb_coverage_df[pb_coverage_df['kappa'] == '200'].copy()
    ii_pb_coverage_df_kappa_200_alpha_05 = pb_coverage_df[pb_coverage_df['kappa'] == '200'].copy()
    ii_pb_coverage_df_kappa_200_alpha_1  = pb_coverage_df[pb_coverage_df['kappa'] == '200'].copy()

    ii_pb_coverage_df_kappa_200_alpha_01['ii_cov'] = pb_coverage_df[pb_coverage_df['kappa'] == '200']['ii_cov'].apply(lambda x: x[0])
    ii_pb_coverage_df_kappa_200_alpha_05['ii_cov'] = pb_coverage_df[pb_coverage_df['kappa'] == '200']['ii_cov'].apply(lambda x: x[1])
    ii_pb_coverage_df_kappa_200_alpha_1['ii_cov'] = pb_coverage_df[pb_coverage_df['kappa'] == '200']['ii_cov'].apply(lambda x: x[2])

    # Create subplot figure with 1 row, 3 columns (one for each alpha level)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="white")

    for alpha_idx, (kappa_50_data, kappa_200_data, alpha_level) in enumerate(zip(
        [ii_pb_coverage_df_kappa_50_alpha_01, ii_pb_coverage_df_kappa_50_alpha_05, ii_pb_coverage_df_kappa_50_alpha_1], 
        [ii_pb_coverage_df_kappa_200_alpha_01, ii_pb_coverage_df_kappa_200_alpha_05, ii_pb_coverage_df_kappa_200_alpha_1], 
        [0.01, 0.05, 0.1]
    )):
        ax = axes[alpha_idx]

        # Extract data for each training size
        kappa_50_boxplot_data = [kappa_50_data[kappa_50_data['train_size'] == size]['ii_cov'].values for size in train_sizes]
        kappa_200_boxplot_data = [kappa_200_data[kappa_200_data['train_size'] == size]['ii_cov'].values for size in train_sizes]

        # Create boxplots with adjusted positions
        positions_kappa_50 = np.array(range(len(train_sizes))) - 0.2
        positions_kappa_200 = np.array(range(len(train_sizes))) + 0.2

        ax.boxplot(kappa_50_boxplot_data, positions=positions_kappa_50, widths=0.3, notch=False, 
                   boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                   whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                   flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                   medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)
                   
        ax.boxplot(kappa_200_boxplot_data, positions=positions_kappa_200, widths=0.3, notch=False, 
                   boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                   whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                   flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                   medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)
        
        sns.set_style("whitegrid")
        # Scatter plot
        palette_kappa_50 = ['#ee6100', 'g', 'b', 'y']
        palette_kappa_200 = ['#ee6100', 'g', 'b', 'y']

        for i, size in enumerate(train_sizes):
            xs_kappa_50 = np.random.normal(positions_kappa_50[i], 0.04, len(kappa_50_boxplot_data[i]))
            xs_kappa_200 = np.random.normal(positions_kappa_200[i], 0.04, len(kappa_200_boxplot_data[i]))

            ax.scatter(xs_kappa_50, kappa_50_boxplot_data[i], alpha=0.2, color=palette_kappa_50[i])
            ax.scatter(xs_kappa_200, kappa_200_boxplot_data[i], alpha=0.2, color=palette_kappa_200[i], marker='^')

        sns.despine(bottom=True)  # Remove right and top axis lines

        ax.set_xticks(range(len(train_sizes)))
        ax.set_xticklabels([str(size) for size in train_sizes], fontsize=17)

        if alpha_level == 0.01:
            ax.set_ylim(0.825, 1.001)
        elif alpha_level == 0.05:
            ax.set_ylim(0.75, 1.001)
        else:
            ax.set_ylim(0.5, 1.001)

        ax.set_xlabel('Training sample size', fontsize=17)
        ax.set_ylabel('Coverage', fontsize=17)
        ax.tick_params(labelsize=17)
        ax.axhline(y=1-alpha_level, color='black', linestyle='dashed')
        ax.grid(False)

        # Custom legend 
        legend_handles = []
        legend_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='none', markersize=10, label=r'$\kappa = 50$'))
        legend_handles.append(mlines.Line2D([], [], color='gray', marker='^', linestyle='none', markersize=10, label=r'$\kappa = 200$'))
        ax.legend(handles=legend_handles, loc='lower right', fontsize=13)
    
    plt.show()
    fig.tight_layout()
    
    # Save individual plots if requested
    if save_individual:
        output_dir = ROOT_DIR / "results_plots"
        output_dir.mkdir(exist_ok=True)
        
        # Create individual plots for each alpha level
        for alpha_idx, (kappa_50_data, kappa_200_data, alpha_level) in enumerate(zip(
            [ii_pb_coverage_df_kappa_50_alpha_01, ii_pb_coverage_df_kappa_50_alpha_05, ii_pb_coverage_df_kappa_50_alpha_1], 
            [ii_pb_coverage_df_kappa_200_alpha_01, ii_pb_coverage_df_kappa_200_alpha_05, ii_pb_coverage_df_kappa_200_alpha_1], 
            [0.01, 0.05, 0.1]
        )):
            # Create individual figure
            fig_individual, ax = plt.subplots(1, 1, figsize=(7, 7), facecolor="white")
            
            # Extract data for each training size
            kappa_50_boxplot_data = [kappa_50_data[kappa_50_data['train_size'] == size]['ii_cov'].values for size in train_sizes]
            kappa_200_boxplot_data = [kappa_200_data[kappa_200_data['train_size'] == size]['ii_cov'].values for size in train_sizes]

            # Create boxplots with adjusted positions
            positions_kappa_50 = np.array(range(len(train_sizes))) - 0.2
            positions_kappa_200 = np.array(range(len(train_sizes))) + 0.2

            ax.boxplot(kappa_50_boxplot_data, positions=positions_kappa_50, widths=0.3, notch=False, 
                       boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                       whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                       flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                       medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)
                       
            ax.boxplot(kappa_200_boxplot_data, positions=positions_kappa_200, widths=0.3, notch=False, 
                       boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                       whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                       flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                       medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)
            
            sns.set_style("whitegrid")
            # Scatter plot
            palette_kappa_50 = ['#ee6100', 'g', 'b', 'y']
            palette_kappa_200 = ['#ee6100', 'g', 'b', 'y']

            for i, size in enumerate(train_sizes):
                xs_kappa_50 = np.random.normal(positions_kappa_50[i], 0.04, len(kappa_50_boxplot_data[i]))
                xs_kappa_200 = np.random.normal(positions_kappa_200[i], 0.04, len(kappa_200_boxplot_data[i]))

                ax.scatter(xs_kappa_50, kappa_50_boxplot_data[i], alpha=0.2, color=palette_kappa_50[i])
                ax.scatter(xs_kappa_200, kappa_200_boxplot_data[i], alpha=0.2, color=palette_kappa_200[i], marker='^')

            sns.despine(bottom=True)  # Remove right and top axis lines

            ax.set_xticks(range(len(train_sizes)))
            ax.set_xticklabels([str(size) for size in train_sizes], fontsize=17)

            if alpha_level == 0.01:
                ax.set_ylim(0.825, 1.001)
            elif alpha_level == 0.05:
                ax.set_ylim(0.75, 1.001)
            else:
                ax.set_ylim(0.5, 1.001)

            ax.set_xlabel('Training sample size', fontsize=17)
            ax.set_ylabel('Coverage', fontsize=17)
            ax.tick_params(labelsize=17)
            ax.axhline(y=1-alpha_level, color='black', linestyle='dashed')
            ax.grid(False)

            # Custom legend 
            legend_handles = []
            legend_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='none', markersize=10, label=r'$\kappa = 50$'))
            legend_handles.append(mlines.Line2D([], [], color='gray', marker='^', linestyle='none', markersize=10, label=r'$\kappa = 200$'))
            ax.legend(handles=legend_handles, loc='lower right', fontsize=13)
            
            fig_individual.tight_layout()
            
            # Save the individual plot
            filename = output_dir / f'sphere_hyperboloid_type_ii_coverage_{str(alpha_level)[2:]}.png'
            fig_individual.savefig(filename, bbox_inches='tight', format='png', dpi=75, transparent=True)
            plt.close(fig_individual)  # Close individual figure to free memory

def sphere_H2_type_iv_analysis(pb_coverage_df, save_individual=True):
    """Generate Type IV plots exactly"""
    train_sizes = [50, 100, 200, 500]
    alpha_levels = [0.01, 0.05, 0.1]
    kappa_values = ['50', '200']
    
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    
    # Create separate dataframes for each kappa and alpha level
    iv_pb_coverage_df_kappa_50_alpha_01 = pb_coverage_df[pb_coverage_df['kappa'] == '50'].copy()
    iv_pb_coverage_df_kappa_50_alpha_05 = pb_coverage_df[pb_coverage_df['kappa'] == '50'].copy()
    iv_pb_coverage_df_kappa_50_alpha_1  = pb_coverage_df[pb_coverage_df['kappa'] == '50'].copy()

    iv_pb_coverage_df_kappa_50_alpha_01['iv_cov']  = pb_coverage_df[pb_coverage_df['kappa'] == '50']['iv_cov'].apply(lambda x: x[0])
    iv_pb_coverage_df_kappa_50_alpha_05['iv_cov']  = pb_coverage_df[pb_coverage_df['kappa'] == '50']['iv_cov'].apply(lambda x: x[1])
    iv_pb_coverage_df_kappa_50_alpha_1['iv_cov']  = pb_coverage_df[pb_coverage_df['kappa'] == '50']['iv_cov'].apply(lambda x: x[2])

    iv_pb_coverage_df_kappa_200_alpha_01 = pb_coverage_df[pb_coverage_df['kappa'] == '200'].copy()
    iv_pb_coverage_df_kappa_200_alpha_05 = pb_coverage_df[pb_coverage_df['kappa'] == '200'].copy()
    iv_pb_coverage_df_kappa_200_alpha_1  = pb_coverage_df[pb_coverage_df['kappa'] == '200'].copy()

    iv_pb_coverage_df_kappa_200_alpha_01['iv_cov'] = pb_coverage_df[pb_coverage_df['kappa'] == '200']['iv_cov'].apply(lambda x: x[0])
    iv_pb_coverage_df_kappa_200_alpha_05['iv_cov'] = pb_coverage_df[pb_coverage_df['kappa'] == '200']['iv_cov'].apply(lambda x: x[1])
    iv_pb_coverage_df_kappa_200_alpha_1['iv_cov'] = pb_coverage_df[pb_coverage_df['kappa'] == '200']['iv_cov'].apply(lambda x: x[2])

    # Create subplot figure with 1 row, 3 columns (one for each alpha level)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="white")

    for alpha_idx, (kappa_50_data, kappa_200_data, alpha_level) in enumerate(zip(
        [iv_pb_coverage_df_kappa_50_alpha_01, iv_pb_coverage_df_kappa_50_alpha_05, iv_pb_coverage_df_kappa_50_alpha_1], 
        [iv_pb_coverage_df_kappa_200_alpha_01, iv_pb_coverage_df_kappa_200_alpha_05, iv_pb_coverage_df_kappa_200_alpha_1], 
        [0.01, 0.05, 0.1]
    )):
        ax = axes[alpha_idx]

        # Extract data for each training size
        kappa_50_boxplot_data = [kappa_50_data[kappa_50_data['train_size'] == size]['iv_cov'].values for size in train_sizes]
        kappa_200_boxplot_data = [kappa_200_data[kappa_200_data['train_size'] == size]['iv_cov'].values for size in train_sizes]

        # Create boxplots with adjusted positions
        positions_kappa_50 = np.array(range(len(train_sizes))) - 0.2
        positions_kappa_200 = np.array(range(len(train_sizes))) + 0.2

        ax.boxplot(kappa_50_boxplot_data, positions=positions_kappa_50, widths=0.3, notch=False, 
                   boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                   whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                   flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                   medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)
                   
        ax.boxplot(kappa_200_boxplot_data, positions=positions_kappa_200, widths=0.3, notch=False, 
                   boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                   whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                   flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                   medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)

        # Scatter plot
        palette_kappa_50 = ['#ee6100', 'g', 'b', 'y']
        palette_kappa_200 = ['#ee6100', 'g', 'b', 'y']

        for i, size in enumerate(train_sizes):
            xs_kappa_50 = np.random.normal(positions_kappa_50[i], 0.04, len(kappa_50_boxplot_data[i]))
            xs_kappa_200 = np.random.normal(positions_kappa_200[i], 0.04, len(kappa_200_boxplot_data[i]))

            ax.scatter(xs_kappa_50, kappa_50_boxplot_data[i], alpha=0.2, color=palette_kappa_50[i])
            ax.scatter(xs_kappa_200, kappa_200_boxplot_data[i], alpha=0.2, color=palette_kappa_200[i], marker='^')

        sns.despine(bottom=True)  # Remove right and top axis lines

        ax.set_xticks(range(len(train_sizes)))
        ax.set_xticklabels([str(size) for size in train_sizes], fontsize=17)

        if alpha_level == 0.01:
            ax.set_ylim(0.825, 1.001)
        elif alpha_level == 0.05:
            ax.set_ylim(0.75, 1.001)
        else:
            ax.set_ylim(0.5, 1.001)

        ax.set_xlabel('Training sample size', fontsize=17)
        ax.set_ylabel('Coverage', fontsize=17)
        ax.tick_params(labelsize=17)
        ax.axhline(y=1-alpha_level, color='black', linestyle='dashed')
        ax.grid(False)

        # Custom legend 
        legend_handles = []
        legend_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='none', markersize=10, label=r'$\kappa = 50$'))
        legend_handles.append(mlines.Line2D([], [], color='gray', marker='^', linestyle='none', markersize=10, label=r'$\kappa = 200$'))
        ax.legend(handles=legend_handles, loc='lower right', fontsize=13)
    
    plt.show()
    fig.tight_layout()
    
    # Save individual plots if requested
    if save_individual:
        output_dir = ROOT_DIR / "results_plots"
        output_dir.mkdir(exist_ok=True)
        
        # Create individual plots for each alpha level
        for alpha_idx, (kappa_50_data, kappa_200_data, alpha_level) in enumerate(zip(
            [iv_pb_coverage_df_kappa_50_alpha_01, iv_pb_coverage_df_kappa_50_alpha_05, iv_pb_coverage_df_kappa_50_alpha_1], 
            [iv_pb_coverage_df_kappa_200_alpha_01, iv_pb_coverage_df_kappa_200_alpha_05, iv_pb_coverage_df_kappa_200_alpha_1], 
            [0.01, 0.05, 0.1]
        )):
            # Create individual figure
            fig_individual, ax = plt.subplots(1, 1, figsize=(7, 7), facecolor="white")
            
            # Extract data for each training size
            kappa_50_boxplot_data = [kappa_50_data[kappa_50_data['train_size'] == size]['iv_cov'].values for size in train_sizes]
            kappa_200_boxplot_data = [kappa_200_data[kappa_200_data['train_size'] == size]['iv_cov'].values for size in train_sizes]

            # Create boxplots with adjusted positions
            positions_kappa_50 = np.array(range(len(train_sizes))) - 0.2
            positions_kappa_200 = np.array(range(len(train_sizes))) + 0.2

            ax.boxplot(kappa_50_boxplot_data, positions=positions_kappa_50, widths=0.3, notch=False, 
                       boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                       whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                       flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                       medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)
                       
            ax.boxplot(kappa_200_boxplot_data, positions=positions_kappa_200, widths=0.3, notch=False, 
                       boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                       whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                       flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                       medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)

            # Scatter plot
            palette_kappa_50 = ['#ee6100', 'g', 'b', 'y']
            palette_kappa_200 = ['#ee6100', 'g', 'b', 'y']

            for i, size in enumerate(train_sizes):
                xs_kappa_50 = np.random.normal(positions_kappa_50[i], 0.04, len(kappa_50_boxplot_data[i]))
                xs_kappa_200 = np.random.normal(positions_kappa_200[i], 0.04, len(kappa_200_boxplot_data[i]))

                ax.scatter(xs_kappa_50, kappa_50_boxplot_data[i], alpha=0.2, color=palette_kappa_50[i])
                ax.scatter(xs_kappa_200, kappa_200_boxplot_data[i], alpha=0.2, color=palette_kappa_200[i], marker='^')

            sns.despine(bottom=True)  # Remove right and top axis lines

            ax.set_xticks(range(len(train_sizes)))
            ax.set_xticklabels([str(size) for size in train_sizes], fontsize=17)

            if alpha_level == 0.01:
                ax.set_ylim(0.825, 1.001)
            elif alpha_level == 0.05:
                ax.set_ylim(0.75, 1.001)
            else:
                ax.set_ylim(0.5, 1.001)

            ax.set_xlabel('Training sample size', fontsize=17)
            ax.set_ylabel('Coverage', fontsize=17)
            ax.tick_params(labelsize=17)
            ax.axhline(y=1-alpha_level, color='black', linestyle='dashed')
            ax.grid(False)

            # Custom legend 
            legend_handles = []
            legend_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='none', markersize=10, label=r'$\kappa = 50$'))
            legend_handles.append(mlines.Line2D([], [], color='gray', marker='^', linestyle='none', markersize=10, label=r'$\kappa = 200$'))
            ax.legend(handles=legend_handles, loc='lower right', fontsize=13)
            
            fig_individual.tight_layout()
            
            # Save the individual plot
            filename = output_dir / f'sphere_hyperboloid_type_iv_coverage_{str(alpha_level)[2:]}.png'
            fig_individual.savefig(filename, bbox_inches='tight', format='png', dpi=75, transparent=True)
            plt.close(fig_individual)  # Close individual figure to free memory

def sphere_H2_radius_analysis(pb_coverage_df, space='sphere', save_individual=True):
    """Create radius boxplots for sphere/hyperboloid data."""
    import matplotlib.patches as mpatches
    
    # Split the OOB_quantile array by alpha levels
    radius_alpha_01 = pb_coverage_df.copy()
    radius_alpha_05 = pb_coverage_df.copy()
    radius_alpha_1 = pb_coverage_df.copy()
    
    radius_alpha_01['OOB_quantile'] = pb_coverage_df['OOB_quantile'].apply(lambda x: x[0])
    radius_alpha_05['OOB_quantile'] = pb_coverage_df['OOB_quantile'].apply(lambda x: x[1])
    radius_alpha_1['OOB_quantile'] = pb_coverage_df['OOB_quantile'].apply(lambda x: x[2])
    
    # Define plotting style
    boxprops = dict(linestyle='-', linewidth=1.5, color='#000000')
    flierprops = dict(marker='o', markersize=1, linestyle='none')
    whiskerprops = dict(color='#000000')
    capprops = dict(color='#000000')
    medianprops = dict(linewidth=1.5, linestyle='-', color='#ff0808')
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="white")
    count = 0

    for data, alpha_level in zip(
        [radius_alpha_01, radius_alpha_05, radius_alpha_1], 
        [0.01, 0.05, 0.1]):
        
        coverage_df = data
        ax = axes[count]
        count += 1
        
        # Extract unique train sizes and kappa values
        train_sizes = sorted(coverage_df['train_size'].unique())
        kappas = ['50', '200'] 
        
        # Prepare the data for boxplots
        grouped_data = [
            [coverage_df.loc[(coverage_df['kappa'] == kappa) & (coverage_df['train_size'] == N), 'OOB_quantile']
                for N in train_sizes]
            for kappa in kappas ]
            
        sns.set_style("whitegrid")
        # Plotting

        palette = ['#ee6100', 'g', 'b', 'y']  # Generate unique colors

        for i, group in enumerate(grouped_data):
            # In this loop, select the kappa values
            base_position = 1 + i * (len(train_sizes) + 1)  # spacing between groups

            for j, ts_data in enumerate(group):
                # In this loop, select the train sizes. ts_data is the dataset for train size and kappa
                pos = base_position + j
                ax.boxplot(ts_data, positions=[pos], widths=.9, notch=False, whiskerprops=whiskerprops,
                          capprops=capprops, flierprops=flierprops, medianprops=medianprops,
                          showmeans=False, showfliers=False) 
        
                palette = ['#ee6100', 'g', 'b', 'y']
                for x, val in zip(np.random.normal(pos, 0.14, ts_data.shape[0]), ts_data):
                    ax.scatter(x, val, alpha=0.2, color=palette[j])

        sns.despine(bottom=True)  # removes right and top axis lines

        # Formatting
        ax.set_xticks(
            ticks=[1 + i * (len(train_sizes) + 1) + (len(train_sizes) - 1) / 2 for i in range(len(kappas))],
            labels=kappas
        )
        ax.set_xlim(0, len(kappas) * (len(train_sizes) + 1))

        if space == 'sphere':
            if alpha_level == 0.01:
                ax.set_ylim(0.2, 1.75)
            elif alpha_level == 0.05:
                ax.set_ylim(0.15, 0.85)
            else:
                ax.set_ylim(0.10, 0.6)

        elif space == 'hyperboloid':
            if alpha_level == 0.01:
                ax.set_ylim(0.1, 0.8)
            elif alpha_level == 0.05:
                ax.set_ylim(0.1, 0.5)
            else:
                ax.set_ylim(0.1, 0.4)
            
        ax.set_xlabel(r'$\kappa$', fontsize=17)
        ax.set_ylabel('Radius', fontsize=17)
        ax.tick_params(axis='x', labelsize=17)
        ax.tick_params(axis='y', labelsize=17)

        legend_handles = [
            mpatches.Patch(color=palette[j], label=f'Train size: {train_sizes[j]}') 
            for j in range(len(train_sizes))
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=13)
        ax.grid(False)
        fig.tight_layout()
        
    plt.show()
    
    # Save individual plots if requested
    if save_individual:
        output_dir = ROOT_DIR / "results_plots"
        output_dir.mkdir(exist_ok=True)
        
        # Create individual plots for each alpha level
        for data, alpha_level in zip(
            [radius_alpha_01, radius_alpha_05, radius_alpha_1], 
            [0.01, 0.05, 0.1]):
            
            coverage_df = data
            # Create individual figure
            fig_individual, ax = plt.subplots(1, 1, figsize=(7, 7), facecolor="white")
            
            # Define plotting style for individual plot
            boxprops = dict(linestyle='-', linewidth=1.5, color='#000000')
            flierprops = dict(marker='o', markersize=1, linestyle='none')
            whiskerprops = dict(color='#000000')
            capprops = dict(color='#000000')
            medianprops = dict(linewidth=1.5, linestyle='-', color='#ff0808')
            
            # Extract unique train sizes and kappa values
            train_sizes = sorted(coverage_df['train_size'].unique())
            kappas = ['50', '200'] 
            
            # Prepare the data for boxplots
            grouped_data = [
                [coverage_df.loc[(coverage_df['kappa'] == kappa) & (coverage_df['train_size'] == N), 'OOB_quantile']
                    for N in train_sizes]
                for kappa in kappas ]
                
            sns.set_style("whitegrid")
            # Plotting

            palette = ['#ee6100', 'g', 'b', 'y']  # Generate unique colors

            for i, group in enumerate(grouped_data):
                # In this loop, select the kappa values
                base_position = 1 + i * (len(train_sizes) + 1)  # spacing between groups

                for j, ts_data in enumerate(group):
                    # In this loop, select the train sizes. ts_data is the dataset for train size and kappa
                    pos = base_position + j
                    ax.boxplot(ts_data, positions=[pos], widths=.9, notch=False, whiskerprops=whiskerprops,
                              capprops=capprops, flierprops=flierprops, medianprops=medianprops,
                              showmeans=False, showfliers=False) 
            
                    palette = ['#ee6100', 'g', 'b', 'y']
                    for x, val in zip(np.random.normal(pos, 0.14, ts_data.shape[0]), ts_data):
                        ax.scatter(x, val, alpha=0.2, color=palette[j])

            sns.despine(bottom=True)  # removes right and top axis lines

            # Formatting
            ax.set_xticks(
                ticks=[1 + i * (len(train_sizes) + 1) + (len(train_sizes) - 1) / 2 for i in range(len(kappas))],
                labels=kappas
            )
            ax.set_xlim(0, len(kappas) * (len(train_sizes) + 1))

            if space == 'sphere':
                if alpha_level == 0.01:
                    ax.set_ylim(0.2, 1.75)
                elif alpha_level == 0.05:
                    ax.set_ylim(0.15, 0.85)
                else:
                    ax.set_ylim(0.15, 0.6)

            elif space == 'hyperboloid':
                if alpha_level == 0.01:
                    ax.set_ylim(0.1, 0.8)
                elif alpha_level == 0.05:
                    ax.set_ylim(0.1, 0.5)
                else:
                    ax.set_ylim(0.1, 0.4)
                
            ax.set_xlabel(r'$\kappa$', fontsize=17)
            ax.set_ylabel('Radius', fontsize=17)
            ax.tick_params(axis='x', labelsize=17)
            ax.tick_params(axis='y', labelsize=17)
            
            # Create proper legend handles for individual plots
            import matplotlib.patches as mpatches
            legend_handles = [
                mpatches.Patch(color=palette[j], label=f'Train size: {train_sizes[j]}') 
                for j in range(len(train_sizes))
            ]
            ax.legend(handles=legend_handles, loc='upper right', fontsize=13)
            ax.grid(False)
            fig_individual.tight_layout()
            
            # Save the individual plot
            filename = output_dir / f'{space}_radius_vs_kappa_{str(alpha_level)[2:]}.png'
            fig_individual.savefig(filename, bbox_inches='tight', format='png', dpi=75, transparent=True)
            plt.close(fig_individual)  # Close individual figure to free memory

# ================================
# SPD SPACE FUNCTIONS
# ================================

def spd_coverage_results():
    """ Compute empirical OOB_quantile for different confidence levels. """
    coverage_df = pd.DataFrame(columns=['sample_index', 'train_size', 'df', 'ai_i_cov', 'ai_ii_cov', 'ai_iii_cov', 'ai_iv_cov',
    'lc_i_cov', 'lc_ii_cov', 'lc_iii_cov', 'lc_iv_cov', 'le_i_cov', 'le_ii_cov', 'le_iii_cov', 'le_iv_cov',
    'ai_OOB_quantile', 'lc_OOB_quantile', 'le_OOB_quantile'])
    i = 0
    results_path = ROOT_DIR / 'simulations_SPD' / 'results'
    for file in os.listdir(results_path):
        if (file.endswith('.npy')):# and file.split('_')[3][5:] == '100'):
            i+=1
            infile=open(results_path / file, 'rb')
            result=np.load(infile, allow_pickle=True).item()
            infile.close()
        else:
            continue
        coverage_df = pd.concat([coverage_df, pd.DataFrame({
            'sample_index': int(file.split('_')[1][4:]),
            'train_size': int(file.split('_')[2][1:]),
            'df': file.split('_')[3][2:],
            'ai_i_cov': [result['ai_i_cov']],
            'ai_ii_cov': [result['ai_ii_cov']],
            'ai_iii_cov': [result['ai_iii_cov']],
            'ai_iv_cov': [result['ai_iv_cov']],
            'lc_i_cov': [result['lc_i_cov']],
            'lc_ii_cov': [result['lc_ii_cov']],
            'lc_iii_cov': [result['lc_iii_cov']],
            'lc_iv_cov': [result['lc_iv_cov']],
            'le_i_cov': [result['le_i_cov']],
            'le_ii_cov': [result['le_ii_cov']],
            'le_iii_cov': [result['le_iii_cov']],
            'le_iv_cov': [result['le_iv_cov']],
            'ai_OOB_quantile': [result['ai_OOB_quantile']],
            'lc_OOB_quantile': [result['lc_OOB_quantile']],
            'le_OOB_quantile': [result['le_OOB_quantile']]
        }, index=pd.RangeIndex(0, 1))], ignore_index=True)

    coverage_df['train_size'] = coverage_df['train_size'].astype('category')
    coverage_df['df'] = coverage_df.df.astype('category')
    return coverage_df

def calculate_type_i_coverage_spd(coverage_df, sample_sizes, df_values, B=500, random_seed=1):
    """
    Calculate Type I coverage using bootstrap procedure.
    
    Parameters:
    -----------
    pb_coverage_df : DataFrame
        Coverage results dataframe
    sample_sizes : list
        List of sample sizes to analyze
    df_values : list
        List of df values to analyze
    B : int
        Number of bootstrap replicates (default: 1000)
    
    Returns:
    --------
    dict : Bootstrap results with means and standard deviations
    """

    if random_seed is not None:
        np.random.seed(random_seed)
    
    ai_diccionario_i = {
        'df_5': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '500': {'means': np.zeros(3), 'stds': np.zeros(3)}},
        'df_15': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '500': {'means': np.zeros(3), 'stds': np.zeros(3)}}
    }

    lc_diccionario_i = {
        'df_5': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '500': {'means': np.zeros(3), 'stds': np.zeros(3)}},
        'df_15': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '500': {'means': np.zeros(3), 'stds': np.zeros(3)}}
    }

    le_diccionario_i = {
        'df_5': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '500': {'means': np.zeros(3), 'stds': np.zeros(3)}},
        'df_15': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '500': {'means': np.zeros(3), 'stds': np.zeros(3)}}
    }

    for N in sample_sizes:
        for df in df_values:            
            # Filter data for current N and df
            coverage_df_N_df = coverage_df[
                (coverage_df['df'] == str(df)) & 
                (coverage_df['train_size'] == N)
            ]
            
            # Get M (number of samples for this N, df combination) and assert it is 1000
            M = len(coverage_df_N_df)
            #assert M == 1000, f"Expected M to be 1000, but got {M} for N={N}, df={df}"
            
            # Bootstrap procedure
            ai_bootstrap_estimates = []  # Will store B bootstrap estimates
            lc_bootstrap_estimates = []  # Will store B bootstrap estimates
            le_bootstrap_estimates = []  # Will store B bootstrap estimates

            ai_coverages = []  # Will store B bootstrap estimates
            lc_coverages = []  # Will store B bootstrap estimates
            le_coverages = []  # Will store B bootstrap estimates

            for m in range(M):
                # Extract row j_m^(b) of the dataset (training sample)
                training_sample_idx = m
                training_sample = coverage_df_N_df.iloc[training_sample_idx]
                
                # Extract element i_m^(b) from the i_cov column (test pair)
                test_pair_idx = m
                ai_coverage_for_this_pair = training_sample['ai_i_cov'][test_pair_idx, :]  # Shape: (3,)
                lc_coverage_for_this_pair = training_sample['lc_i_cov'][test_pair_idx, :]  # Shape: (3,)
                le_coverage_for_this_pair = training_sample['le_i_cov'][test_pair_idx, :]  # Shape: (3,)
                
                ai_coverages.append(ai_coverage_for_this_pair)
                lc_coverages.append(lc_coverage_for_this_pair)
                le_coverages.append(le_coverage_for_this_pair)
            

            # Convert to array and compute mean across the M bootstrap samples
            ai_coverages = np.array(ai_coverages)  # Shape: (M, 3)
            ai_p_hat_M = np.mean(ai_coverages, axis=0)  # Shape: (3,)
            lc_coverages = np.array(lc_coverages)  # Shape: (M, 3)
            lc_p_hat_M = np.mean(lc_coverages, axis=0)  # Shape: (3,)
            le_coverages = np.array(le_coverages)  # Shape: (M, 3)
            le_p_hat_M = np.mean(le_coverages, axis=0)  # Shape: (3,)

            for b in range(B):
                # Sample M indices for test pairs (i_1^(b), ..., i_M^(b))
                i_indices = np.random.choice(M, size=M, replace=True)
                
                # Sample M indices for training samples (j_1^(b), ..., j_M^(b))
                j_indices = np.random.choice(M, size=M, replace=True)
                
                # Extract bootstrap sample
                ai_bootstrap_coverages = []
                lc_bootstrap_coverages = []
                le_bootstrap_coverages = []
                
                for m in range(M):
                    # Extract row j_m^(b) of the dataset (training sample)
                    training_sample_idx = j_indices[m]
                    training_sample = coverage_df_N_df.iloc[training_sample_idx]
                    
                    # Extract element i_m^(b) from the i_cov column (test pair)
                    test_pair_idx = i_indices[m]
                    ai_coverage_for_this_pair = training_sample['ai_i_cov'][test_pair_idx, :]  # Shape: (3,)
                    lc_coverage_for_this_pair = training_sample['lc_i_cov'][test_pair_idx, :]  # Shape: (3,)
                    le_coverage_for_this_pair = training_sample['le_i_cov'][test_pair_idx, :]  # Shape: (3,)
                    
                    ai_bootstrap_coverages.append(ai_coverage_for_this_pair)
                    lc_bootstrap_coverages.append(lc_coverage_for_this_pair)
                    le_bootstrap_coverages.append(le_coverage_for_this_pair)
                
                # Convert to array and compute mean across the M bootstrap samples
                ai_bootstrap_coverages = np.array(ai_bootstrap_coverages)  # Shape: (M, 3)
                ai_p_hat_M_b = np.mean(ai_bootstrap_coverages, axis=0)  # Shape: (3,)

                lc_bootstrap_coverages = np.array(lc_bootstrap_coverages)  # Shape: (M, 3)
                lc_p_hat_M_b = np.mean(lc_bootstrap_coverages, axis=0)  # Shape: (3,)

                le_bootstrap_coverages = np.array(le_bootstrap_coverages)  # Shape: (M, 3)
                le_p_hat_M_b = np.mean(le_bootstrap_coverages, axis=0)  # Shape: (3,)
                
                ai_bootstrap_estimates.append(ai_p_hat_M_b)
                lc_bootstrap_estimates.append(lc_p_hat_M_b)
                le_bootstrap_estimates.append(le_p_hat_M_b)
            
            # Convert bootstrap estimates to array
            ai_bootstrap_estimates = np.array(ai_bootstrap_estimates)  # Shape: (B, 3)
            lc_bootstrap_estimates = np.array(lc_bootstrap_estimates)  # Shape: (B, 3)
            le_bootstrap_estimates = np.array(le_bootstrap_estimates)  # Shape: (B, 3)
            
            # Compute overall mean and standard deviation
            #ai_p_bar_M = np.mean(ai_bootstrap_estimates, axis=0)  # Shape: (3,)
            ai_sigma_boot = np.std(ai_bootstrap_estimates, axis=0, ddof=1)  # Shape: (3,)
            #lc_p_bar_M = np.mean(lc_bootstrap_estimates, axis=0)  # Shape: (3,)
            lc_sigma_boot = np.std(lc_bootstrap_estimates, axis=0, ddof=1)
            #le_p_bar_M = np.mean(le_bootstrap_estimates, axis=0)  # Shape: (3,)
            le_sigma_boot = np.std(le_bootstrap_estimates, axis=0, ddof=1)
            
            # Store results
            df_key = f'df_{df}'
            N_key = str(N)
            ai_diccionario_i[df_key][N_key]['means'] = ai_p_hat_M
            ai_diccionario_i[df_key][N_key]['stds'] = ai_sigma_boot
            lc_diccionario_i[df_key][N_key]['means'] = lc_p_hat_M
            lc_diccionario_i[df_key][N_key]['stds'] = lc_sigma_boot
            le_diccionario_i[df_key][N_key]['means'] = le_p_hat_M
            le_diccionario_i[df_key][N_key]['stds'] = le_sigma_boot
    
    return ai_diccionario_i, lc_diccionario_i, le_diccionario_i

def spd_type_i_analysis(SPD_coverage_df):
    """Generate Type I tables exactly"""
    sample_sizes = [50, 100, 200, 500]
    df_values = [5, 15]
    
    # Calculate bootstrap statistics
    ai_diccionario_i, lc_diccionario_i, le_diccionario_i = calculate_type_i_coverage_spd(
        coverage_df=SPD_coverage_df, 
        sample_sizes=sample_sizes, 
        df_values=df_values,
        B=500,  # Number of bootstrap replicates
        random_seed=1  # Optional: set a random seed for reproducibility
    )
    
    # Prepare data for the DataFrame
    ai_rows = []
    le_rows = []
    lc_rows = []
    index = []

    for df in [5, 15]:
        for N in [50, 100, 200, 500]:
            ai_row = []
            le_row = []
            lc_row = []
            ai_means = ai_diccionario_i[f'df_{df}'][str(N)]['means']
            ai_stds = ai_diccionario_i[f'df_{df}'][str(N)]['stds']
            le_means = le_diccionario_i[f'df_{df}'][str(N)]['means']
            le_stds = le_diccionario_i[f'df_{df}'][str(N)]['stds']
            lc_means = lc_diccionario_i[f'df_{df}'][str(N)]['means']
            lc_stds = lc_diccionario_i[f'df_{df}'][str(N)]['stds']
            # Format as "mean (std)"
            ai_formatted_values = [f"{100*ai_means[i]:.1f} ({100*ai_stds[i]:.2f})" for i in range(3)]
            ai_row.extend(ai_formatted_values)
            ai_rows.append(ai_row)
            le_formatted_values = [f"{100*le_means[i]:.1f} ({100*le_stds[i]:.2f})" for i in range(3)]
            le_row.extend(le_formatted_values)
            le_rows.append(le_row)
            lc_formatted_values = [f"{100*lc_means[i]:.1f} ({100*lc_stds[i]:.2f})" for i in range(3)]
            lc_row.extend(lc_formatted_values)
            lc_rows.append(lc_row)
            index.append((f"{df}", f"{N}"))

    # MultiIndex for rows and columns
    row_index = pd.MultiIndex.from_tuples(index, names=["d", "N"])
    col_index = pd.MultiIndex.from_product(
        [["0.01", "0.05", "0.1"]],
        names=[r"Significance Level"]
    )

    # Create the DataFrame
    ai_df = pd.DataFrame(ai_rows, index=row_index, columns=col_index)
    le_df = pd.DataFrame(le_rows, index=row_index, columns=col_index)
    lc_df = pd.DataFrame(lc_rows, index=row_index, columns=col_index)

    # Define target coverage values that correspond to each column
    # Column 0 (0.01): 0.99, Column 1 (0.05): 0.95, Column 2 (0.1): 0.90
    target_coverages = [0.99, 0.95, 0.90]

    # Apply formatting with column-specific target coverage
    ai_latex = ai_df.copy()
    le_latex = le_df.copy()
    lc_latex = lc_df.copy()
    for col_idx, col in enumerate(ai_df.columns):
        # Use the target coverage corresponding to this column
        target_coverage = target_coverages[col_idx]

        for row_idx in ai_df.index:
            ai_latex.loc[row_idx, col] = format_cell(
                ai_df.loc[row_idx, col], 
                target_coverage=target_coverage, 
                n_trials=1000
            )
            le_latex.loc[row_idx, col] = format_cell(
                le_df.loc[row_idx, col], 
                target_coverage=target_coverage, 
                n_trials=1000
            )
            lc_latex.loc[row_idx, col] = format_cell(
                lc_df.loc[row_idx, col], 
                target_coverage=target_coverage, 
                n_trials=1000
            )

    return ai_latex, le_latex, lc_latex

def spd_type_ii_analysis(SPD_coverage_df, save_individual=True):
    """Generate Type II plots"""
    ai_SPD_coverage_df = SPD_coverage_df[['train_size', 'df', 'ai_i_cov', 'ai_ii_cov', 'ai_iii_cov', 'ai_iv_cov', 'ai_OOB_quantile']]
    lc_SPD_coverage_df = SPD_coverage_df[['train_size', 'df', 'lc_i_cov', 'lc_ii_cov', 'lc_iii_cov', 'lc_iv_cov', 'lc_OOB_quantile']]
    le_SPD_coverage_df = SPD_coverage_df[['train_size', 'df', 'le_i_cov', 'le_ii_cov', 'le_iii_cov', 'le_iv_cov', 'le_OOB_quantile']]
    
    # DF 5 alpha level analysis
    ai_ii_SPD_coverage_df_df_5_alpha_01 = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '5'].copy()
    ai_ii_SPD_coverage_df_df_5_alpha_05 = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '5'].copy()
    ai_ii_SPD_coverage_df_df_5_alpha_1  = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '5'].copy()

    lc_ii_SPD_coverage_df_df_5_alpha_01 = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '5'].copy()
    lc_ii_SPD_coverage_df_df_5_alpha_05 = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '5'].copy()
    lc_ii_SPD_coverage_df_df_5_alpha_1  = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '5'].copy()

    le_ii_SPD_coverage_df_df_5_alpha_01 = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '5'].copy()
    le_ii_SPD_coverage_df_df_5_alpha_05 = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '5'].copy()
    le_ii_SPD_coverage_df_df_5_alpha_1  = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '5'].copy()

    ai_ii_SPD_coverage_df_df_5_alpha_01['ai_ii_cov'] = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '5']['ai_ii_cov'].apply(lambda x: x[0])
    ai_ii_SPD_coverage_df_df_5_alpha_01['ai_OOB_quantile'] = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '5']['ai_OOB_quantile'].apply(lambda x: x[0])

    lc_ii_SPD_coverage_df_df_5_alpha_01['lc_ii_cov'] = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '5']['lc_ii_cov'].apply(lambda x: x[0])
    lc_ii_SPD_coverage_df_df_5_alpha_01['lc_OOB_quantile'] = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '5']['lc_OOB_quantile'].apply(lambda x: x[0])

    le_ii_SPD_coverage_df_df_5_alpha_01['le_ii_cov'] = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '5']['le_ii_cov'].apply(lambda x: x[0])
    le_ii_SPD_coverage_df_df_5_alpha_01['le_OOB_quantile'] = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '5']['le_OOB_quantile'].apply(lambda x: x[0])

    ai_ii_SPD_coverage_df_df_5_alpha_05['ai_ii_cov'] = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '5']['ai_ii_cov'].apply(lambda x: x[1])
    ai_ii_SPD_coverage_df_df_5_alpha_05['ai_OOB_quantile'] = SPD_coverage_df[SPD_coverage_df['df'] == '5']['ai_OOB_quantile'].apply(lambda x: x[1])

    lc_ii_SPD_coverage_df_df_5_alpha_05['lc_ii_cov'] = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '5']['lc_ii_cov'].apply(lambda x: x[1])
    lc_ii_SPD_coverage_df_df_5_alpha_05['lc_OOB_quantile'] = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '5']['lc_OOB_quantile'].apply(lambda x: x[1])

    le_ii_SPD_coverage_df_df_5_alpha_05['le_ii_cov'] = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '5']['le_ii_cov'].apply(lambda x: x[1])
    le_ii_SPD_coverage_df_df_5_alpha_05['le_OOB_quantile'] = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '5']['le_OOB_quantile'].apply(lambda x: x[1])

    ai_ii_SPD_coverage_df_df_5_alpha_1['ai_ii_cov'] = SPD_coverage_df[SPD_coverage_df['df'] == '5']['ai_ii_cov'].apply(lambda x: x[2])
    ai_ii_SPD_coverage_df_df_5_alpha_1['ai_OOB_quantile'] = SPD_coverage_df[SPD_coverage_df['df'] == '5']['ai_OOB_quantile'].apply(lambda x: x[2])

    lc_ii_SPD_coverage_df_df_5_alpha_1['lc_ii_cov'] = SPD_coverage_df[SPD_coverage_df['df'] == '5']['lc_ii_cov'].apply(lambda x: x[2])
    lc_ii_SPD_coverage_df_df_5_alpha_1['lc_OOB_quantile'] = SPD_coverage_df[SPD_coverage_df['df'] == '5']['lc_OOB_quantile'].apply(lambda x: x[2])

    le_ii_SPD_coverage_df_df_5_alpha_1['le_ii_cov'] = SPD_coverage_df[SPD_coverage_df['df'] == '5']['le_ii_cov'].apply(lambda x: x[2])
    le_ii_SPD_coverage_df_df_5_alpha_1['le_OOB_quantile'] = SPD_coverage_df[SPD_coverage_df['df'] == '5']['le_OOB_quantile'].apply(lambda x: x[2])

    # DF 15 alpha level analysis
    ai_ii_SPD_coverage_df_df_15_alpha_01 = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '15'].copy()
    ai_ii_SPD_coverage_df_df_15_alpha_05 = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '15'].copy()
    ai_ii_SPD_coverage_df_df_15_alpha_1  = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '15'].copy()

    lc_ii_SPD_coverage_df_df_15_alpha_01 = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '15'].copy()
    lc_ii_SPD_coverage_df_df_15_alpha_05 = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '15'].copy()
    lc_ii_SPD_coverage_df_df_15_alpha_1  = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '15'].copy()

    le_ii_SPD_coverage_df_df_15_alpha_01 = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '15'].copy()
    le_ii_SPD_coverage_df_df_15_alpha_05 = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '15'].copy()
    le_ii_SPD_coverage_df_df_15_alpha_1  = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '15'].copy()

    ai_ii_SPD_coverage_df_df_15_alpha_01['ai_ii_cov'] = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '15']['ai_ii_cov'].apply(lambda x: x[0])
    ai_ii_SPD_coverage_df_df_15_alpha_01['ai_OOB_quantile'] = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '15']['ai_OOB_quantile'].apply(lambda x: x[0])

    lc_ii_SPD_coverage_df_df_15_alpha_01['lc_ii_cov'] = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '15']['lc_ii_cov'].apply(lambda x: x[0])
    lc_ii_SPD_coverage_df_df_15_alpha_01['lc_OOB_quantile'] = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '15']['lc_OOB_quantile'].apply(lambda x: x[0])

    le_ii_SPD_coverage_df_df_15_alpha_01['le_ii_cov'] = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '15']['le_ii_cov'].apply(lambda x: x[0])
    le_ii_SPD_coverage_df_df_15_alpha_01['le_OOB_quantile'] = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '15']['le_OOB_quantile'].apply(lambda x: x[0])

    ai_ii_SPD_coverage_df_df_15_alpha_05['ai_ii_cov'] = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '15']['ai_ii_cov'].apply(lambda x: x[1])
    ai_ii_SPD_coverage_df_df_15_alpha_05['ai_OOB_quantile'] = SPD_coverage_df[SPD_coverage_df['df'] == '15']['ai_OOB_quantile'].apply(lambda x: x[1])

    lc_ii_SPD_coverage_df_df_15_alpha_05['lc_ii_cov'] = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '15']['lc_ii_cov'].apply(lambda x: x[1])
    lc_ii_SPD_coverage_df_df_15_alpha_05['lc_OOB_quantile'] = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '15']['lc_OOB_quantile'].apply(lambda x: x[1])

    le_ii_SPD_coverage_df_df_15_alpha_05['le_ii_cov'] = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '15']['le_ii_cov'].apply(lambda x: x[1])
    le_ii_SPD_coverage_df_df_15_alpha_05['le_OOB_quantile'] = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '15']['le_OOB_quantile'].apply(lambda x: x[1])

    ai_ii_SPD_coverage_df_df_15_alpha_1['ai_ii_cov'] = SPD_coverage_df[SPD_coverage_df['df'] == '15']['ai_ii_cov'].apply(lambda x: x[2])
    ai_ii_SPD_coverage_df_df_15_alpha_1['ai_OOB_quantile'] = SPD_coverage_df[SPD_coverage_df['df'] == '15']['ai_OOB_quantile'].apply(lambda x: x[2])

    lc_ii_SPD_coverage_df_df_15_alpha_1['lc_ii_cov'] = SPD_coverage_df[SPD_coverage_df['df'] == '15']['lc_ii_cov'].apply(lambda x: x[2])
    lc_ii_SPD_coverage_df_df_15_alpha_1['lc_OOB_quantile'] = SPD_coverage_df[SPD_coverage_df['df'] == '15']['lc_OOB_quantile'].apply(lambda x: x[2])

    le_ii_SPD_coverage_df_df_15_alpha_1['le_ii_cov'] = SPD_coverage_df[SPD_coverage_df['df'] == '15']['le_ii_cov'].apply(lambda x: x[2])
    le_ii_SPD_coverage_df_df_15_alpha_1['le_OOB_quantile'] = SPD_coverage_df[SPD_coverage_df['df'] == '15']['le_OOB_quantile'].apply(lambda x: x[2])

    # AI plots - Create 1x3 subplot figure
    print(f"\n Affine-invariant metric")
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="white")
    
    for alpha_idx, (df_5_data, df_15_data, alpha_level) in enumerate(zip(
        [ai_ii_SPD_coverage_df_df_5_alpha_01, ai_ii_SPD_coverage_df_df_5_alpha_05, ai_ii_SPD_coverage_df_df_5_alpha_1], 
        [ai_ii_SPD_coverage_df_df_15_alpha_01, ai_ii_SPD_coverage_df_df_15_alpha_05, ai_ii_SPD_coverage_df_df_15_alpha_1], 
        [0.01, 0.05, 0.1]
    )):
        ax = axes[alpha_idx]

        # Extract data for each training size
        train_sizes = [50, 100, 200, 500]
        df_5_boxplot_data = [df_5_data[df_5_data['train_size'] == size]['ai_ii_cov'].values for size in train_sizes]
        df_15_boxplot_data = [df_15_data[df_15_data['train_size'] == size]['ai_ii_cov'].values for size in train_sizes]

        # Create boxplots with adjusted positions
        positions_df_5 = np.array(range(len(train_sizes))) - 0.2
        positions_df_15 = np.array(range(len(train_sizes))) + 0.2

        ax.boxplot(df_5_boxplot_data, positions=positions_df_5, widths=0.3, notch=False, 
                   boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                   whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                   showfliers=False,
                   medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False)
                   
        ax.boxplot(df_15_boxplot_data, positions=positions_df_15, widths=0.3, notch=False, 
                   boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                   whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                   showfliers=False,
                   medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False)

        # Scatter plot
        palette_df_5 = ['#ee6100', 'g', 'b', 'y']
        palette_df_15 = ['#ee6100', 'g', 'b', 'y']

        for i, size in enumerate(train_sizes):
            xs_df_5 = np.random.normal(positions_df_5[i], 0.04, len(df_5_boxplot_data[i]))
            xs_df_15 = np.random.normal(positions_df_15[i], 0.04, len(df_15_boxplot_data[i]))

            ax.scatter(xs_df_5, df_5_boxplot_data[i], alpha=0.2, color=palette_df_5[i], label='Prediction balls')
            ax.scatter(xs_df_15, df_15_boxplot_data[i], alpha=0.2, color=palette_df_15[i], marker='^', label='Split-conformal')

        sns.despine(bottom=True)  # Remove right and top axis lines
        sns.set_style("whitegrid")

        ax.set_xticks(range(len(train_sizes)))
        ax.set_xticklabels([str(size) for size in train_sizes], fontsize=17)

        if alpha_level == 0.01:
            ax.set_ylim(0.825, 1.001)
        elif alpha_level == 0.05:
            ax.set_ylim(0.75, 1.001)
        else:
            ax.set_ylim(0.5, 1.001)

        ax.set_xlabel('Training sample size', fontsize=17)
        ax.set_ylabel('Coverage', fontsize=17)
        ax.tick_params(labelsize=17)
        ax.axhline(y=1-alpha_level, color='black', linestyle='dashed')
        ax.grid(False)

        # Custom legend
        legend_handles = []
        legend_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='none', markersize=10, label=r'$d = 5$'))
        legend_handles.append(mlines.Line2D([], [], color='gray', marker='^', linestyle='none', markersize=10, label=r'$d = 15$'))

        ax.legend(handles=legend_handles, loc='lower right', fontsize=13)
    
    plt.show()
    fig.tight_layout()

    # LC plots - Create 1x3 subplot figure
    print(f"\n Log-Cholesky metric")
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="white")
    
    for alpha_idx, (df_5_data, df_15_data, alpha_level) in enumerate(zip(
        [lc_ii_SPD_coverage_df_df_5_alpha_01, lc_ii_SPD_coverage_df_df_5_alpha_05, lc_ii_SPD_coverage_df_df_5_alpha_1], 
        [lc_ii_SPD_coverage_df_df_15_alpha_01, lc_ii_SPD_coverage_df_df_15_alpha_05, lc_ii_SPD_coverage_df_df_15_alpha_1], 
        [0.01, 0.05, 0.1]
    )):
        ax = axes[alpha_idx]

        # Extract data for each training size
        train_sizes = [50, 100, 200, 500]
        df_5_boxplot_data = [df_5_data[df_5_data['train_size'] == size]['lc_ii_cov'].values for size in train_sizes]
        df_15_boxplot_data = [df_15_data[df_15_data['train_size'] == size]['lc_ii_cov'].values for size in train_sizes]

        # Create boxplots with adjusted positions
        positions_df_5 = np.array(range(len(train_sizes))) - 0.2
        positions_df_15 = np.array(range(len(train_sizes))) + 0.2

        ax.boxplot(df_5_boxplot_data, positions=positions_df_5, widths=0.3, notch=False, 
                   boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                   whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                   showfliers=False,
                   medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False)
                   
        ax.boxplot(df_15_boxplot_data, positions=positions_df_15, widths=0.3, notch=False, 
                   boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                   whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                   showfliers=False,
                   medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False)

        # Scatter plot
        palette_df_5 = ['#ee6100', 'g', 'b', 'y']
        palette_df_15 = ['#ee6100', 'g', 'b', 'y']

        for i, size in enumerate(train_sizes):
            xs_df_5 = np.random.normal(positions_df_5[i], 0.04, len(df_5_boxplot_data[i]))
            xs_df_15 = np.random.normal(positions_df_15[i], 0.04, len(df_15_boxplot_data[i]))

            ax.scatter(xs_df_5, df_5_boxplot_data[i], alpha=0.2, color=palette_df_5[i], label='Prediction balls')
            ax.scatter(xs_df_15, df_15_boxplot_data[i], alpha=0.2, color=palette_df_15[i], marker='^', label='Split-conformal')

        sns.despine(bottom=True)  # Remove right and top axis lines
        sns.set_style("whitegrid")

        ax.set_xticks(range(len(train_sizes)))
        ax.set_xticklabels([str(size) for size in train_sizes], fontsize=17)

        if alpha_level == 0.01:
            ax.set_ylim(0.825, 1.001)
        elif alpha_level == 0.05:
            ax.set_ylim(0.75, 1.001)
        else:
            ax.set_ylim(0.5, 1.001)

        ax.set_xlabel('Training sample size', fontsize=17)
        ax.set_ylabel('Coverage', fontsize=17)
        ax.tick_params(labelsize=17)
        ax.axhline(y=1-alpha_level, color='black', linestyle='dashed')
        ax.grid(False)

        # Custom legend
        legend_handles = []
        legend_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='none', markersize=10, label=r'$d = 5$'))
        legend_handles.append(mlines.Line2D([], [], color='gray', marker='^', linestyle='none', markersize=10, label=r'$d = 15$'))

        ax.legend(handles=legend_handles, loc='lower right', fontsize=13)
    
    plt.show()
    fig.tight_layout()

    # LE plots - Create 1x3 subplot figure
    print(f"\n Log-Euclidean metric")
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="white")
    
    for alpha_idx, (df_5_data, df_15_data, alpha_level) in enumerate(zip(
        [le_ii_SPD_coverage_df_df_5_alpha_01, le_ii_SPD_coverage_df_df_5_alpha_05, le_ii_SPD_coverage_df_df_5_alpha_1], 
        [le_ii_SPD_coverage_df_df_15_alpha_01, le_ii_SPD_coverage_df_df_15_alpha_05, le_ii_SPD_coverage_df_df_15_alpha_1], 
        [0.01, 0.05, 0.1]
    )):
        ax = axes[alpha_idx]

        # Extract data for each training size
        train_sizes = [50, 100, 200, 500]
        df_5_boxplot_data = [df_5_data[df_5_data['train_size'] == size]['le_ii_cov'].values for size in train_sizes]
        df_15_boxplot_data = [df_15_data[df_15_data['train_size'] == size]['le_ii_cov'].values for size in train_sizes]

        # Create boxplots with adjusted positions
        positions_df_5 = np.array(range(len(train_sizes))) - 0.2
        positions_df_15 = np.array(range(len(train_sizes))) + 0.2

        ax.boxplot(df_5_boxplot_data, positions=positions_df_5, widths=0.3, notch=False, 
                   boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                   whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                   showfliers=False,
                   medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False)
                   
        ax.boxplot(df_15_boxplot_data, positions=positions_df_15, widths=0.3, notch=False, 
                   boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                   whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                   showfliers=False,
                   medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False)

        # Scatter plot
        palette_df_5 = ['#ee6100', 'g', 'b', 'y']
        palette_df_15 = ['#ee6100', 'g', 'b', 'y']

        for i, size in enumerate(train_sizes):
            xs_df_5 = np.random.normal(positions_df_5[i], 0.04, len(df_5_boxplot_data[i]))
            xs_df_15 = np.random.normal(positions_df_15[i], 0.04, len(df_15_boxplot_data[i]))

            ax.scatter(xs_df_5, df_5_boxplot_data[i], alpha=0.2, color=palette_df_5[i], label='Prediction balls')
            ax.scatter(xs_df_15, df_15_boxplot_data[i], alpha=0.2, color=palette_df_15[i], marker='^', label='Split-conformal')

        sns.despine(bottom=True)  # Remove right and top axis lines
        sns.set_style("whitegrid")

        ax.set_xticks(range(len(train_sizes)))
        ax.set_xticklabels([str(size) for size in train_sizes], fontsize=17)

        if alpha_level == 0.01:
            ax.set_ylim(0.825, 1.001)
        elif alpha_level == 0.05:
            ax.set_ylim(0.75, 1.001)
        else:
            ax.set_ylim(0.5, 1.001)

        ax.set_xlabel('Training sample size', fontsize=17)
        ax.set_ylabel('Coverage', fontsize=17)
        ax.tick_params(labelsize=17)
        ax.axhline(y=1-alpha_level, color='black', linestyle='dashed')
        ax.grid(False)

        # Custom legend
        legend_handles = []
        legend_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='none', markersize=10, label=r'$d = 5$'))
        legend_handles.append(mlines.Line2D([], [], color='gray', marker='^', linestyle='none', markersize=10, label=r'$d = 15$'))

        ax.legend(handles=legend_handles, loc='lower right', fontsize=13)
    
    plt.show()
    fig.tight_layout()

# ================================
# TYPE IV AND MISSING TYPE III ANALYSIS FUNCTIONS
# ================================

def euclidean_type_iv_analysis(coverage_df, sigma_value='0.9', save_individual=True):
    """Generate Type IV plots for Euclidean distance"""
    train_sizes = [50, 100, 200, 500]
    alpha_levels = [0.01, 0.05, 0.1]
    sigma_values = [0.9, 1.7]
    
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    
    
    for sigma in sigma_values:
        if sigma == 0.9:
            print(" Sigma = √3/2")
        elif sigma == 1.7:
            print(" Sigma = √3")
        fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="white")
        
        # Store data for individual plots
        iv_coverage_dfs = {}
        
        # Create plots for each alpha level 
        for alpha_idx, alpha_level in enumerate(alpha_levels):
            # Create separate dataframes for each sigma and alpha level
            sigma_data = coverage_df[coverage_df['sigma'] == str(sigma)].copy()
            
            sigma_data['pb_iv_cov_alpha'] = sigma_data['pb_iv_cov'].apply(lambda x: x[alpha_idx])
            sigma_data['conf_iv_cov_alpha'] = sigma_data['conf_iv_cov'].apply(lambda x: x[alpha_idx])
            iv_coverage_dfs[alpha_level] = sigma_data

            ax = axes[alpha_idx]

            # Extract data for each training size
            pb_sigma_boxplot_data = [sigma_data[sigma_data['train_size'] == size]['pb_iv_cov_alpha'].values for size in train_sizes]
            conf_sigma_boxplot_data = [sigma_data[sigma_data['train_size'] == size]['conf_iv_cov_alpha'].values for size in train_sizes]

            # Create boxplots with adjusted positions
            pb_positions_sigma = np.array(range(len(train_sizes))) - 0.2
            conf_positions_sigma = np.array(range(len(train_sizes))) + 0.2

            ax.boxplot(pb_sigma_boxplot_data, positions=pb_positions_sigma, widths=0.3, notch=False, 
                       boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                       whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                       flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                       medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)
                       
            ax.boxplot(conf_sigma_boxplot_data, positions=conf_positions_sigma, widths=0.3, notch=False, 
                       boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                       whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                       flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                       medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)

            # Scatter plot
            pb_palette_sigma = ['#ee6100', 'g', 'b', 'y']
            conf_palette_sigma = ['#ee6100', 'g', 'b', 'y']

            for i, size in enumerate(train_sizes):
                pb_xs_sigma = np.random.normal(pb_positions_sigma[i], 0.04, len(pb_sigma_boxplot_data[i]))
                conf_xs_sigma = np.random.normal(conf_positions_sigma[i], 0.04, len(conf_sigma_boxplot_data[i]))

                ax.scatter(pb_xs_sigma, pb_sigma_boxplot_data[i], alpha=0.2, color = pb_palette_sigma[i], label='Prediction balls')
                ax.scatter(conf_xs_sigma, conf_sigma_boxplot_data[i], alpha=0.2, color = conf_palette_sigma[i], marker='^', label='Split-conformal')

            sns.despine(bottom=True)  # Remove right and top axis lines
            sns.set_style("whitegrid")
            ax.set_xticks(range(len(train_sizes)))
            ax.set_xticklabels([str(size) for size in train_sizes], fontsize=17)

            if alpha_level == 0.01:
                ax.set_ylim(0.86, 1.001)
            elif alpha_level == 0.05:
                ax.set_ylim(0.75, 1.001)
            else:
                ax.set_ylim(0.5, 1)

            ax.set_xlabel('Training sample size', fontsize=17)
            ax.set_ylabel('Coverage', fontsize=17)
            ax.tick_params(labelsize=17)
            ax.axhline(y=1-alpha_level, color='black', linestyle='dashed')
            ax.grid(False)

            legend_handles = []
            legend_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='none', markersize=10, label='OOB prediction balls'))
            legend_handles.append(mlines.Line2D([], [], color='gray', marker='^', linestyle='none', markersize=10, label='Split-conformal'))

            ax.legend(handles=legend_handles, loc='lower right', fontsize=13)
        plt.show()
        
        # Save individual plots if requested
        if save_individual:
            output_dir = ROOT_DIR / "results_plots"
            output_dir.mkdir(exist_ok=True)
            
            # Create individual plots for each alpha level
            for alpha_idx, alpha_level in enumerate(alpha_levels):
                sigma_data = iv_coverage_dfs[alpha_level]
                
                # Create individual figure
                fig_individual, ax = plt.subplots(1, 1, figsize=(7, 7), facecolor="white")
                
                # Extract data for each training size
                pb_sigma_boxplot_data = [sigma_data[sigma_data['train_size'] == size]['pb_iv_cov_alpha'].values for size in train_sizes]
                conf_sigma_boxplot_data = [sigma_data[sigma_data['train_size'] == size]['conf_iv_cov_alpha'].values for size in train_sizes]

                # Create boxplots with adjusted positions
                pb_positions_sigma = np.array(range(len(train_sizes))) - 0.2
                conf_positions_sigma = np.array(range(len(train_sizes))) + 0.2

                ax.boxplot(pb_sigma_boxplot_data, positions=pb_positions_sigma, widths=0.3, notch=False, 
                           boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                           whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                           flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                           medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)
                           
                ax.boxplot(conf_sigma_boxplot_data, positions=conf_positions_sigma, widths=0.3, notch=False, 
                           boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                           whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                           flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                           medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)

                # Scatter plot
                pb_palette_sigma = ['#ee6100', 'g', 'b', 'y']
                conf_palette_sigma = ['#ee6100', 'g', 'b', 'y']

                for i, size in enumerate(train_sizes):
                    pb_xs_sigma = np.random.normal(pb_positions_sigma[i], 0.04, len(pb_sigma_boxplot_data[i]))
                    conf_xs_sigma = np.random.normal(conf_positions_sigma[i], 0.04, len(conf_sigma_boxplot_data[i]))

                    ax.scatter(pb_xs_sigma, pb_sigma_boxplot_data[i], alpha=0.2, color = pb_palette_sigma[i])
                    ax.scatter(conf_xs_sigma, conf_sigma_boxplot_data[i], alpha=0.2, color = conf_palette_sigma[i], marker='^')

                sns.despine(bottom=True)  # Remove right and top axis lines
                sns.set_style("whitegrid")
                ax.set_xticks(range(len(train_sizes)))
                ax.set_xticklabels([str(size) for size in train_sizes], fontsize=17)

                if alpha_level == 0.01:
                    ax.set_ylim(0.86, 1.001)
                elif alpha_level == 0.05:
                    ax.set_ylim(0.75, 1.001)
                else:
                    ax.set_ylim(0.5, 1)

                ax.set_xlabel('Training sample size', fontsize=17)
                ax.set_ylabel('Coverage', fontsize=17)
                ax.tick_params(labelsize=17)
                ax.axhline(y=1-alpha_level, color='black', linestyle='dashed')
                ax.grid(False)

                legend_handles = []
                legend_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='none', markersize=10, label='OOB prediction balls'))
                legend_handles.append(mlines.Line2D([], [], color='gray', marker='^', linestyle='none', markersize=10, label='Split-conformal'))

                ax.legend(handles=legend_handles, loc='lower right', fontsize=13)
                
                fig_individual.tight_layout()
                
                # Save the individual plot
                filename = output_dir / f'euclidean_type_iv_coverage_sigma_{str(sigma).replace(".", "")}_alpha_{str(alpha_level)[2:]}.png'
                fig_individual.savefig(filename, bbox_inches='tight', format='png', dpi=75, transparent=True)
                plt.close(fig_individual)  # Close individual figure to free memory

    fig.tight_layout()


def sphere_type_iii_analysis(coverage_df, kappa_value='50'):
    """
    Extract and analyze Type III coverage data for Sphere/H2 space
    
    """
    # Filter data for the specified kappa value
    filtered_df = coverage_df[coverage_df['kappa'] == kappa_value].copy()
    
    if len(filtered_df) == 0:
        raise ValueError(f"No data found for kappa={kappa_value}. Available kappa values: {coverage_df['kappa'].unique()}")
    
    # Initialize results for each alpha level
    iii_coverage_df_alpha_01 = filtered_df.copy()
    iii_coverage_df_alpha_05 = filtered_df.copy()
    iii_coverage_df_alpha_1 = filtered_df.copy()
    
    # Extract Type III coverage for each alpha level
    # iii_cov contains arrays of shape (1000, 3) where columns are [alpha_0.01, alpha_0.05, alpha_0.1]
    alpha_01_coverage = []
    alpha_05_coverage = []
    alpha_1_coverage = []
    
    for idx, row in filtered_df.iterrows():
        iii_cov_data = row['iii_cov']  # Shape: (1000, 3)
        # Calculate mean coverage across the 1000 test samples for each alpha
        alpha_01_coverage.append(np.mean(iii_cov_data[:, 0]))  # Alpha 0.01
        alpha_05_coverage.append(np.mean(iii_cov_data[:, 1]))  # Alpha 0.05
        alpha_1_coverage.append(np.mean(iii_cov_data[:, 2]))   # Alpha 0.1
    
    iii_coverage_df_alpha_01['iii_cov'] = alpha_01_coverage
    iii_coverage_df_alpha_05['iii_cov'] = alpha_05_coverage
    iii_coverage_df_alpha_1['iii_cov'] = alpha_1_coverage
    
    return [iii_coverage_df_alpha_01, iii_coverage_df_alpha_05, iii_coverage_df_alpha_1]

def create_sphere_type_iii_table(coverage_df, kappa_value='50'):
    """
    Create Type III coverage table for Sphere space using proper formatting
    """
    return sphere_H2_type_iii_analysis(coverage_df)

def create_hyperboloid_type_iii_table(coverage_df, kappa_value='50'):
    """
    Create Type III coverage table for Hyperboloid space using proper formatting
    """
    return sphere_H2_type_iii_analysis(coverage_df)

def sphere_type_iv_analysis(coverage_df, kappa_value='0.9'):
    """
    Extract and analyze Type IV coverage data for Sphere/H2 space
    
    """
    # Create separate dataframes for each alpha level
    iv_coverage_df_alpha_01 = coverage_df[coverage_df['kappa'] == kappa_value].copy()
    iv_coverage_df_alpha_05 = coverage_df[coverage_df['kappa'] == kappa_value].copy()
    iv_coverage_df_alpha_1 = coverage_df[coverage_df['kappa'] == kappa_value].copy()
    
    # Extract Type IV coverage for each alpha level
    iv_coverage_df_alpha_01['iv_cov'] = coverage_df[coverage_df['kappa'] == kappa_value]['iv_cov'].apply(lambda x: x[0])
    iv_coverage_df_alpha_05['iv_cov'] = coverage_df[coverage_df['kappa'] == kappa_value]['iv_cov'].apply(lambda x: x[1])
    iv_coverage_df_alpha_1['iv_cov'] = coverage_df[coverage_df['kappa'] == kappa_value]['iv_cov'].apply(lambda x: x[2])
    
    return [iv_coverage_df_alpha_01, iv_coverage_df_alpha_05, iv_coverage_df_alpha_1]

def spd_type_iii_analysis(coverage_df, df_value='3'):
    """
    Extract and analyze Type III coverage data for SPD space (all three metrics)
    
    """
    df_data = coverage_df[coverage_df['df'] == df_value]
    
    # AI metric
    ai_iii_coverage_df_alpha_01 = df_data.copy()
    ai_iii_coverage_df_alpha_05 = df_data.copy()
    ai_iii_coverage_df_alpha_1 = df_data.copy()
    
    ai_iii_coverage_df_alpha_01['ai_iii_cov'] = df_data['ai_iii_cov'].apply(lambda x: x[0])
    ai_iii_coverage_df_alpha_05['ai_iii_cov'] = df_data['ai_iii_cov'].apply(lambda x: x[1])
    ai_iii_coverage_df_alpha_1['ai_iii_cov'] = df_data['ai_iii_cov'].apply(lambda x: x[2])
    
    # LC metric
    lc_iii_coverage_df_alpha_01 = df_data.copy()
    lc_iii_coverage_df_alpha_05 = df_data.copy()
    lc_iii_coverage_df_alpha_1 = df_data.copy()
    
    lc_iii_coverage_df_alpha_01['lc_iii_cov'] = df_data['lc_iii_cov'].apply(lambda x: x[0])
    lc_iii_coverage_df_alpha_05['lc_iii_cov'] = df_data['lc_iii_cov'].apply(lambda x: x[1])
    lc_iii_coverage_df_alpha_1['lc_iii_cov'] = df_data['lc_iii_cov'].apply(lambda x: x[2])
    
    # LE metric
    le_iii_coverage_df_alpha_01 = df_data.copy()
    le_iii_coverage_df_alpha_05 = df_data.copy()
    le_iii_coverage_df_alpha_1 = df_data.copy()
    
    le_iii_coverage_df_alpha_01['le_iii_cov'] = df_data['le_iii_cov'].apply(lambda x: x[0])
    le_iii_coverage_df_alpha_05['le_iii_cov'] = df_data['le_iii_cov'].apply(lambda x: x[1])
    le_iii_coverage_df_alpha_1['le_iii_cov'] = df_data['le_iii_cov'].apply(lambda x: x[2])
    
    return {
        'ai': [ai_iii_coverage_df_alpha_01, ai_iii_coverage_df_alpha_05, ai_iii_coverage_df_alpha_1],
        'lc': [lc_iii_coverage_df_alpha_01, lc_iii_coverage_df_alpha_05, lc_iii_coverage_df_alpha_1],
        'le': [le_iii_coverage_df_alpha_01, le_iii_coverage_df_alpha_05, le_iii_coverage_df_alpha_1]
    }

def calculate_type_iii_coverage_spd(coverage_df, sample_sizes, df_values, B=500, random_seed=1):
    """
    Calculate Type III coverage using bootstrap procedure.
    
    Parameters:
    -----------
    coverage_df : DataFrame
        Coverage results dataframe
    sample_sizes : list
        List of sample sizes to analyze
    df_values : list
        List of df values to analyze
    B : int
        Number of bootstrap replicates (default: 500)
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Bootstrap results with means and standard deviations
    """

    if random_seed is not None:
        np.random.seed(random_seed)
    
    ai_diccionario_iii = {
        'df_5': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '500': {'means': np.zeros(3), 'stds': np.zeros(3)}},
        'df_15': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '500': {'means': np.zeros(3), 'stds': np.zeros(3)}}
    }

    lc_diccionario_iii = {
        'df_5': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '500': {'means': np.zeros(3), 'stds': np.zeros(3)}},
        'df_15': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '500': {'means': np.zeros(3), 'stds': np.zeros(3)}}
    }

    le_diccionario_iii = {
        'df_5': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                     '500': {'means': np.zeros(3), 'stds': np.zeros(3)}},
        'df_15': {'50': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '100': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '200': {'means': np.zeros(3), 'stds': np.zeros(3)}, 
                      '500': {'means': np.zeros(3), 'stds': np.zeros(3)}}
    }

    for N in sample_sizes:
        for df in df_values:            
            # Filter data for current N and df
            coverage_df_N_df = coverage_df[
                (coverage_df['df'] == str(df)) & 
                (coverage_df['train_size'] == N)
            ]
            
            # Get M (number of samples for this N, df combination)
            M = len(coverage_df_N_df)
            
            # Calculate original estimates
            ai_coverages = []
            lc_coverages = []
            le_coverages = []

            for m in range(M):
                training_sample = coverage_df_N_df.iloc[m]
                test_pair_idx = m
                ai_coverage_for_this_pair = training_sample['ai_iii_cov'][test_pair_idx, :]
                lc_coverage_for_this_pair = training_sample['lc_iii_cov'][test_pair_idx, :]
                le_coverage_for_this_pair = training_sample['le_iii_cov'][test_pair_idx, :]
                
                ai_coverages.append(ai_coverage_for_this_pair)
                lc_coverages.append(lc_coverage_for_this_pair)
                le_coverages.append(le_coverage_for_this_pair)
            
            # Convert to array and compute mean across the M bootstrap samples
            ai_coverages = np.array(ai_coverages)
            ai_p_hat_M = np.mean(ai_coverages, axis=0)
            lc_coverages = np.array(lc_coverages)
            lc_p_hat_M = np.mean(lc_coverages, axis=0)
            le_coverages = np.array(le_coverages)
            le_p_hat_M = np.mean(le_coverages, axis=0)

            # Bootstrap procedure
            ai_bootstrap_estimates = []
            lc_bootstrap_estimates = []
            le_bootstrap_estimates = []

            for b in range(B):
                # Sample M indices for test pairs and training samples
                i_indices = np.random.choice(M, size=M, replace=True)
                j_indices = np.random.choice(M, size=M, replace=True)
                
                # Extract bootstrap sample
                ai_bootstrap_coverages = []
                lc_bootstrap_coverages = []
                le_bootstrap_coverages = []
                
                for m in range(M):
                    training_sample_idx = j_indices[m]
                    training_sample = coverage_df_N_df.iloc[training_sample_idx]
                    test_pair_idx = i_indices[m]
                    
                    ai_coverage_for_this_pair = training_sample['ai_iii_cov'][test_pair_idx, :]
                    lc_coverage_for_this_pair = training_sample['lc_iii_cov'][test_pair_idx, :]
                    le_coverage_for_this_pair = training_sample['le_iii_cov'][test_pair_idx, :]
                    
                    ai_bootstrap_coverages.append(ai_coverage_for_this_pair)
                    lc_bootstrap_coverages.append(lc_coverage_for_this_pair)
                    le_bootstrap_coverages.append(le_coverage_for_this_pair)
                
                # Convert to array and compute mean
                ai_bootstrap_coverages = np.array(ai_bootstrap_coverages)
                ai_p_hat_M_b = np.mean(ai_bootstrap_coverages, axis=0)
                lc_bootstrap_coverages = np.array(lc_bootstrap_coverages)
                lc_p_hat_M_b = np.mean(lc_bootstrap_coverages, axis=0)
                le_bootstrap_coverages = np.array(le_bootstrap_coverages)
                le_p_hat_M_b = np.mean(le_bootstrap_coverages, axis=0)
                
                ai_bootstrap_estimates.append(ai_p_hat_M_b)
                lc_bootstrap_estimates.append(lc_p_hat_M_b)
                le_bootstrap_estimates.append(le_p_hat_M_b)
            
            # Convert bootstrap estimates to array and compute std
            ai_bootstrap_estimates = np.array(ai_bootstrap_estimates)
            ai_sigma_boot = np.std(ai_bootstrap_estimates, axis=0, ddof=1)
            lc_bootstrap_estimates = np.array(lc_bootstrap_estimates)
            lc_sigma_boot = np.std(lc_bootstrap_estimates, axis=0, ddof=1)
            le_bootstrap_estimates = np.array(le_bootstrap_estimates)
            le_sigma_boot = np.std(le_bootstrap_estimates, axis=0, ddof=1)
            
            # Store results
            df_key = f'df_{df}'
            N_key = str(N)
            ai_diccionario_iii[df_key][N_key]['means'] = ai_p_hat_M
            ai_diccionario_iii[df_key][N_key]['stds'] = ai_sigma_boot
            lc_diccionario_iii[df_key][N_key]['means'] = lc_p_hat_M
            lc_diccionario_iii[df_key][N_key]['stds'] = lc_sigma_boot
            le_diccionario_iii[df_key][N_key]['means'] = le_p_hat_M
            le_diccionario_iii[df_key][N_key]['stds'] = le_sigma_boot
    
    return ai_diccionario_iii, lc_diccionario_iii, le_diccionario_iii

def create_spd_type_iii_tables(coverage_df, df_value='3'):
    """
    Create Type III coverage tables for SPD space (all three metrics)
    """
    sample_sizes = [50, 100, 200, 500]
    df_values = [5, 15]
    
    # Calculate bootstrap statistics
    ai_diccionario_iii, lc_diccionario_iii, le_diccionario_iii = calculate_type_iii_coverage_spd(
        coverage_df=coverage_df, 
        sample_sizes=sample_sizes, 
        df_values=df_values,
        B=500,  # Number of bootstrap replicates
        random_seed=1  # Optional: set a random seed for reproducibility
    )
    
    # Prepare data for the DataFrame
    ai_rows = []
    le_rows = []
    lc_rows = []
    index = []

    for df in [5, 15]:
        for N in [50, 100, 200, 500]:
            ai_row = []
            le_row = []
            lc_row = []
            ai_means = ai_diccionario_iii[f'df_{df}'][str(N)]['means']
            ai_stds = ai_diccionario_iii[f'df_{df}'][str(N)]['stds']
            le_means = le_diccionario_iii[f'df_{df}'][str(N)]['means']
            le_stds = le_diccionario_iii[f'df_{df}'][str(N)]['stds']
            lc_means = lc_diccionario_iii[f'df_{df}'][str(N)]['means']
            lc_stds = lc_diccionario_iii[f'df_{df}'][str(N)]['stds']
            # Format as "mean (std)"
            ai_formatted_values = [f"{100*ai_means[i]:.1f} ({100*ai_stds[i]:.2f})" for i in range(3)]
            ai_row.extend(ai_formatted_values)
            ai_rows.append(ai_row)
            le_formatted_values = [f"{100*le_means[i]:.1f} ({100*le_stds[i]:.2f})" for i in range(3)]
            le_row.extend(le_formatted_values)
            le_rows.append(le_row)
            lc_formatted_values = [f"{100*lc_means[i]:.1f} ({100*lc_stds[i]:.2f})" for i in range(3)]
            lc_row.extend(lc_formatted_values)
            lc_rows.append(lc_row)
            index.append((f"{df}", f"{N}"))

    # MultiIndex for rows and columns
    row_index = pd.MultiIndex.from_tuples(index, names=["d", "N"])
    col_index = pd.MultiIndex.from_product(
        [["0.01", "0.05", "0.1"]],
        names=[r"Significance Level"]
    )

    # Create the DataFrame
    ai_df = pd.DataFrame(ai_rows, index=row_index, columns=col_index)
    le_df = pd.DataFrame(le_rows, index=row_index, columns=col_index)
    lc_df = pd.DataFrame(lc_rows, index=row_index, columns=col_index)

    # Define target coverage values that correspond to each column
    # Column 0 (0.01): 0.99, Column 1 (0.05): 0.95, Column 2 (0.1): 0.90
    target_coverages = [0.99, 0.95, 0.90]

    # Apply formatting with column-specific target coverage
    ai_latex = ai_df.copy()
    le_latex = le_df.copy()
    lc_latex = lc_df.copy()
    for col_idx, col in enumerate(ai_df.columns):
        # Use the target coverage corresponding to this column
        target_coverage = target_coverages[col_idx]

        for row_idx in ai_df.index:
            ai_latex.loc[row_idx, col] = format_cell(
                ai_df.loc[row_idx, col], 
                target_coverage=target_coverage, 
                n_trials=1000
            )
            le_latex.loc[row_idx, col] = format_cell(
                le_df.loc[row_idx, col], 
                target_coverage=target_coverage, 
                n_trials=1000
            )
            lc_latex.loc[row_idx, col] = format_cell(
                lc_df.loc[row_idx, col], 
                target_coverage=target_coverage, 
                n_trials=1000
            )

    return ai_latex, le_latex, lc_latex
    
    # LE metric
    le_iii_coverage_df_alpha_01 = df_data.copy()
    le_iii_coverage_df_alpha_05 = df_data.copy()
    le_iii_coverage_df_alpha_1 = df_data.copy()
    
    le_iii_coverage_df_alpha_01['le_iii_cov'] = df_data['le_iii_cov'].apply(lambda x: x[0])
    le_iii_coverage_df_alpha_05['le_iii_cov'] = df_data['le_iii_cov'].apply(lambda x: x[1])
    le_iii_coverage_df_alpha_1['le_iii_cov'] = df_data['le_iii_cov'].apply(lambda x: x[2])
    
    return {
        'ai': [ai_iii_coverage_df_alpha_01, ai_iii_coverage_df_alpha_05, ai_iii_coverage_df_alpha_1],
        'lc': [lc_iii_coverage_df_alpha_01, lc_iii_coverage_df_alpha_05, lc_iii_coverage_df_alpha_1],
        'le': [le_iii_coverage_df_alpha_01, le_iii_coverage_df_alpha_05, le_iii_coverage_df_alpha_1]
    }

def spd_type_iv_analysis(SPD_coverage_df, save_individual=True):
    """Generate Type IV plots"""
    ai_SPD_coverage_df = SPD_coverage_df[['train_size', 'df', 'ai_i_cov', 'ai_ii_cov', 'ai_iii_cov', 'ai_iv_cov', 'ai_OOB_quantile']]
    lc_SPD_coverage_df = SPD_coverage_df[['train_size', 'df', 'lc_i_cov', 'lc_ii_cov', 'lc_iii_cov', 'lc_iv_cov', 'lc_OOB_quantile']]
    le_SPD_coverage_df = SPD_coverage_df[['train_size', 'df', 'le_i_cov', 'le_ii_cov', 'le_iii_cov', 'le_iv_cov', 'le_OOB_quantile']]
    
    # DF 5 alpha level analysis
    ai_iv_SPD_coverage_df_df_5_alpha_01 = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '5'].copy()
    ai_iv_SPD_coverage_df_df_5_alpha_05 = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '5'].copy()
    ai_iv_SPD_coverage_df_df_5_alpha_1  = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '5'].copy()

    lc_iv_SPD_coverage_df_df_5_alpha_01 = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '5'].copy()
    lc_iv_SPD_coverage_df_df_5_alpha_05 = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '5'].copy()
    lc_iv_SPD_coverage_df_df_5_alpha_1  = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '5'].copy()

    le_iv_SPD_coverage_df_df_5_alpha_01 = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '5'].copy()
    le_iv_SPD_coverage_df_df_5_alpha_05 = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '5'].copy()
    le_iv_SPD_coverage_df_df_5_alpha_1  = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '5'].copy()

    ai_iv_SPD_coverage_df_df_5_alpha_01['ai_iv_cov'] = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '5']['ai_iv_cov'].apply(lambda x: x[0])
    ai_iv_SPD_coverage_df_df_5_alpha_01['ai_OOB_quantile'] = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '5']['ai_OOB_quantile'].apply(lambda x: x[0])

    lc_iv_SPD_coverage_df_df_5_alpha_01['lc_iv_cov'] = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '5']['lc_iv_cov'].apply(lambda x: x[0])
    lc_iv_SPD_coverage_df_df_5_alpha_01['lc_OOB_quantile'] = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '5']['lc_OOB_quantile'].apply(lambda x: x[0])

    le_iv_SPD_coverage_df_df_5_alpha_01['le_iv_cov'] = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '5']['le_iv_cov'].apply(lambda x: x[0])
    le_iv_SPD_coverage_df_df_5_alpha_01['le_OOB_quantile'] = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '5']['le_OOB_quantile'].apply(lambda x: x[0])

    ai_iv_SPD_coverage_df_df_5_alpha_05['ai_iv_cov'] = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '5']['ai_iv_cov'].apply(lambda x: x[1])
    ai_iv_SPD_coverage_df_df_5_alpha_05['ai_OOB_quantile'] = SPD_coverage_df[SPD_coverage_df['df'] == '5']['ai_OOB_quantile'].apply(lambda x: x[1])

    lc_iv_SPD_coverage_df_df_5_alpha_05['lc_iv_cov'] = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '5']['lc_iv_cov'].apply(lambda x: x[1])
    lc_iv_SPD_coverage_df_df_5_alpha_05['lc_OOB_quantile'] = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '5']['lc_OOB_quantile'].apply(lambda x: x[1])

    le_iv_SPD_coverage_df_df_5_alpha_05['le_iv_cov'] = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '5']['le_iv_cov'].apply(lambda x: x[1])
    le_iv_SPD_coverage_df_df_5_alpha_05['le_OOB_quantile'] = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '5']['le_OOB_quantile'].apply(lambda x: x[1])

    ai_iv_SPD_coverage_df_df_5_alpha_1['ai_iv_cov'] = SPD_coverage_df[SPD_coverage_df['df'] == '5']['ai_iv_cov'].apply(lambda x: x[2])
    ai_iv_SPD_coverage_df_df_5_alpha_1['ai_OOB_quantile'] = SPD_coverage_df[SPD_coverage_df['df'] == '5']['ai_OOB_quantile'].apply(lambda x: x[2])

    lc_iv_SPD_coverage_df_df_5_alpha_1['lc_iv_cov'] = SPD_coverage_df[SPD_coverage_df['df'] == '5']['lc_iv_cov'].apply(lambda x: x[2])
    lc_iv_SPD_coverage_df_df_5_alpha_1['lc_OOB_quantile'] = SPD_coverage_df[SPD_coverage_df['df'] == '5']['lc_OOB_quantile'].apply(lambda x: x[2])

    le_iv_SPD_coverage_df_df_5_alpha_1['le_iv_cov'] = SPD_coverage_df[SPD_coverage_df['df'] == '5']['le_iv_cov'].apply(lambda x: x[2])
    le_iv_SPD_coverage_df_df_5_alpha_1['le_OOB_quantile'] = SPD_coverage_df[SPD_coverage_df['df'] == '5']['le_OOB_quantile'].apply(lambda x: x[2])

    # DF 15 alpha level analysis
    ai_iv_SPD_coverage_df_df_15_alpha_01 = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '15'].copy()
    ai_iv_SPD_coverage_df_df_15_alpha_05 = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '15'].copy()
    ai_iv_SPD_coverage_df_df_15_alpha_1  = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '15'].copy()

    lc_iv_SPD_coverage_df_df_15_alpha_01 = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '15'].copy()
    lc_iv_SPD_coverage_df_df_15_alpha_05 = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '15'].copy()
    lc_iv_SPD_coverage_df_df_15_alpha_1  = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '15'].copy()

    le_iv_SPD_coverage_df_df_15_alpha_01 = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '15'].copy()
    le_iv_SPD_coverage_df_df_15_alpha_05 = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '15'].copy()
    le_iv_SPD_coverage_df_df_15_alpha_1  = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '15'].copy()

    ai_iv_SPD_coverage_df_df_15_alpha_01['ai_iv_cov'] = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '15']['ai_iv_cov'].apply(lambda x: x[0])
    ai_iv_SPD_coverage_df_df_15_alpha_01['ai_OOB_quantile'] = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '15']['ai_OOB_quantile'].apply(lambda x: x[0])

    lc_iv_SPD_coverage_df_df_15_alpha_01['lc_iv_cov'] = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '15']['lc_iv_cov'].apply(lambda x: x[0])
    lc_iv_SPD_coverage_df_df_15_alpha_01['lc_OOB_quantile'] = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '15']['lc_OOB_quantile'].apply(lambda x: x[0])

    le_iv_SPD_coverage_df_df_15_alpha_01['le_iv_cov'] = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '15']['le_iv_cov'].apply(lambda x: x[0])
    le_iv_SPD_coverage_df_df_15_alpha_01['le_OOB_quantile'] = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '15']['le_OOB_quantile'].apply(lambda x: x[0])

    ai_iv_SPD_coverage_df_df_15_alpha_05['ai_iv_cov'] = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '15']['ai_iv_cov'].apply(lambda x: x[1])
    ai_iv_SPD_coverage_df_df_15_alpha_05['ai_OOB_quantile'] = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '15']['ai_OOB_quantile'].apply(lambda x: x[1])

    lc_iv_SPD_coverage_df_df_15_alpha_05['lc_iv_cov'] = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '15']['lc_iv_cov'].apply(lambda x: x[1])
    lc_iv_SPD_coverage_df_df_15_alpha_05['lc_OOB_quantile'] = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '15']['lc_OOB_quantile'].apply(lambda x: x[1])

    le_iv_SPD_coverage_df_df_15_alpha_05['le_iv_cov'] = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '15']['le_iv_cov'].apply(lambda x: x[1])
    le_iv_SPD_coverage_df_df_15_alpha_05['le_OOB_quantile'] = le_SPD_coverage_df[le_SPD_coverage_df['df'] == '15']['le_OOB_quantile'].apply(lambda x: x[1])

    ai_iv_SPD_coverage_df_df_15_alpha_1['ai_iv_cov'] = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '15']['ai_iv_cov'].apply(lambda x: x[2])
    ai_iv_SPD_coverage_df_df_15_alpha_1['ai_OOB_quantile'] = ai_SPD_coverage_df[ai_SPD_coverage_df['df'] == '15']['ai_OOB_quantile'].apply(lambda x: x[2])

    lc_iv_SPD_coverage_df_df_15_alpha_1['lc_iv_cov'] = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '15']['lc_iv_cov'].apply(lambda x: x[2])
    lc_iv_SPD_coverage_df_df_15_alpha_1['lc_OOB_quantile'] = lc_SPD_coverage_df[lc_SPD_coverage_df['df'] == '15']['lc_OOB_quantile'].apply(lambda x: x[2])

    le_iv_SPD_coverage_df_df_15_alpha_1['le_iv_cov'] = SPD_coverage_df[SPD_coverage_df['df'] == '15']['le_iv_cov'].apply(lambda x: x[2])
    le_iv_SPD_coverage_df_df_15_alpha_1['le_OOB_quantile'] = SPD_coverage_df[SPD_coverage_df['df'] == '15']['le_OOB_quantile'].apply(lambda x: x[2])

    # AI plots - Create 1x3 subplot figure
    print(f"\n Affine-invariant metric")
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="white")
    
    for alpha_idx, (df_5_data, df_15_data, alpha_level) in enumerate(zip(
        [ai_iv_SPD_coverage_df_df_5_alpha_01, ai_iv_SPD_coverage_df_df_5_alpha_05, ai_iv_SPD_coverage_df_df_5_alpha_1], 
        [ai_iv_SPD_coverage_df_df_15_alpha_01, ai_iv_SPD_coverage_df_df_15_alpha_05, ai_iv_SPD_coverage_df_df_15_alpha_1], 
        [0.01, 0.05, 0.1]
    )):
        ax = axes[alpha_idx]

        # Extract data for each training size
        train_sizes = [50, 100, 200, 500]
        df_5_boxplot_data = [df_5_data[df_5_data['train_size'] == size]['ai_iv_cov'].values for size in train_sizes]
        df_15_boxplot_data = [df_15_data[df_15_data['train_size'] == size]['ai_iv_cov'].values for size in train_sizes]

        # Create boxplots with adjusted positions
        positions_df_5 = np.array(range(len(train_sizes))) - 0.2
        positions_df_15 = np.array(range(len(train_sizes))) + 0.2

        ax.boxplot(df_5_boxplot_data, positions=positions_df_5, widths=0.3, notch=False, 
                   boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                   whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                   showfliers=False,
                   medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False)
                   
        ax.boxplot(df_15_boxplot_data, positions=positions_df_15, widths=0.3, notch=False, 
                   boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                   whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                   showfliers=False,
                   medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False)

        # Scatter plot
        palette_df_5 = ['#ee6100', 'g', 'b', 'y']
        palette_df_15 = ['#ee6100', 'g', 'b', 'y']

        for i, size in enumerate(train_sizes):
            xs_df_5 = np.random.normal(positions_df_5[i], 0.04, len(df_5_boxplot_data[i]))
            xs_df_15 = np.random.normal(positions_df_15[i], 0.04, len(df_15_boxplot_data[i]))

            ax.scatter(xs_df_5, df_5_boxplot_data[i], alpha=0.2, color=palette_df_5[i], label='Prediction balls')
            ax.scatter(xs_df_15, df_15_boxplot_data[i], alpha=0.2, color=palette_df_15[i], marker='^', label='Split-conformal')

        sns.despine(bottom=True)  # Remove right and top axis lines
        sns.set_style("whitegrid")
        ax.set_xticks(range(len(train_sizes)))
        ax.set_xticklabels([str(size) for size in train_sizes], fontsize=17)

        if alpha_level == 0.01:
            ax.set_ylim(0.86, 1.001)
        elif alpha_level == 0.05:
            ax.set_ylim(0.75, 1.001)
        else:
            ax.set_ylim(0.5, 1)

        ax.set_xlabel('Training sample size', fontsize=17)
        ax.set_ylabel('Coverage', fontsize=17)
        ax.tick_params(labelsize=17)
        ax.axhline(y=1-alpha_level, color='black', linestyle='dashed')
        ax.grid(False)

        # Custom legend
        legend_handles = []
        legend_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='none', markersize=10, label=r'$d = 5$'))
        legend_handles.append(mlines.Line2D([], [], color='gray', marker='^', linestyle='none', markersize=10, label=r'$d = 15$'))

        ax.legend(handles=legend_handles, loc='lower right', fontsize=13)
    
    plt.show()
    fig.tight_layout()

    # LC plots - Create 1x3 subplot figure
    print(f"\n Log-Cholesky metric")
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="white")
    
    for alpha_idx, (df_5_data, df_15_data, alpha_level) in enumerate(zip(
        [lc_iv_SPD_coverage_df_df_5_alpha_01, lc_iv_SPD_coverage_df_df_5_alpha_05, lc_iv_SPD_coverage_df_df_5_alpha_1], 
        [lc_iv_SPD_coverage_df_df_15_alpha_01, lc_iv_SPD_coverage_df_df_15_alpha_05, lc_iv_SPD_coverage_df_df_15_alpha_1], 
        [0.01, 0.05, 0.1]
    )):
        ax = axes[alpha_idx]

        # Extract data for each training size
        train_sizes = [50, 100, 200, 500]
        df_5_boxplot_data = [df_5_data[df_5_data['train_size'] == size]['lc_iv_cov'].values for size in train_sizes]
        df_15_boxplot_data = [df_15_data[df_15_data['train_size'] == size]['lc_iv_cov'].values for size in train_sizes]

        # Create boxplots with adjusted positions
        positions_df_5 = np.array(range(len(train_sizes))) - 0.2
        positions_df_15 = np.array(range(len(train_sizes))) + 0.2

        ax.boxplot(df_5_boxplot_data, positions=positions_df_5, widths=0.3, notch=False, 
                   boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                   whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                   flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                   medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)
                   
        ax.boxplot(df_15_boxplot_data, positions=positions_df_15, widths=0.3, notch=False, 
                   boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                   whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                   flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                   medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)

        # Scatter plot
        palette_df_5 = ['#ee6100', 'g', 'b', 'y']
        palette_df_15 = ['#ee6100', 'g', 'b', 'y']

        for i, size in enumerate(train_sizes):
            xs_df_5 = np.random.normal(positions_df_5[i], 0.04, len(df_5_boxplot_data[i]))
            xs_df_15 = np.random.normal(positions_df_15[i], 0.04, len(df_15_boxplot_data[i]))

            ax.scatter(xs_df_5, df_5_boxplot_data[i], alpha=0.2, color=palette_df_5[i], label='Prediction balls')
            ax.scatter(xs_df_15, df_15_boxplot_data[i], alpha=0.2, color=palette_df_15[i], marker='^', label='Split-conformal')

        sns.despine(bottom=True)  # Remove right and top axis lines
        sns.set_style("whitegrid")
        ax.set_xticks(range(len(train_sizes)))
        ax.set_xticklabels([str(size) for size in train_sizes], fontsize=17)

        if alpha_level == 0.01:
            ax.set_ylim(0.86, 1.001)
        elif alpha_level == 0.05:
            ax.set_ylim(0.75, 1.001)
        else:
            ax.set_ylim(0.5, 1)

        ax.set_xlabel('Training sample size', fontsize=17)
        ax.set_ylabel('Coverage', fontsize=17)
        ax.tick_params(labelsize=17)
        ax.axhline(y=1-alpha_level, color='black', linestyle='dashed')
        ax.grid(False)

        # Custom legend
        legend_handles = []
        legend_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='none', markersize=10, label=r'$d = 5$'))
        legend_handles.append(mlines.Line2D([], [], color='gray', marker='^', linestyle='none', markersize=10, label=r'$d = 15$'))

        ax.legend(handles=legend_handles, loc='lower right', fontsize=13)
    
    plt.show()
    fig.tight_layout()

    # LE plots - Create 1x3 subplot figure
    print(f"\n Log-Euclidean metric")
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="white")
    
    for alpha_idx, (df_5_data, df_15_data, alpha_level) in enumerate(zip(
        [le_iv_SPD_coverage_df_df_5_alpha_01, le_iv_SPD_coverage_df_df_5_alpha_05, le_iv_SPD_coverage_df_df_5_alpha_1], 
        [le_iv_SPD_coverage_df_df_15_alpha_01, le_iv_SPD_coverage_df_df_15_alpha_05, le_iv_SPD_coverage_df_df_15_alpha_1], 
        [0.01, 0.05, 0.1]
    )):
        ax = axes[alpha_idx]

        # Extract data for each training size
        train_sizes = [50, 100, 200, 500]
        df_5_boxplot_data = [df_5_data[df_5_data['train_size'] == size]['le_iv_cov'].values for size in train_sizes]
        df_15_boxplot_data = [df_15_data[df_15_data['train_size'] == size]['le_iv_cov'].values for size in train_sizes]

        # Create boxplots with adjusted positions
        positions_df_5 = np.array(range(len(train_sizes))) - 0.2
        positions_df_15 = np.array(range(len(train_sizes))) + 0.2

        ax.boxplot(df_5_boxplot_data, positions=positions_df_5, widths=0.3, notch=False, 
                   boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                   whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                   flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                   medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)
                   
        ax.boxplot(df_15_boxplot_data, positions=positions_df_15, widths=0.3, notch=False, 
                   boxprops=dict(color='#000000', linestyle='-', linewidth=1.5), 
                   whiskerprops=dict(color='#000000'), capprops=dict(color='#000000'), 
                   flierprops=dict(marker='o', markersize=1, linestyle='none'), 
                   medianprops=dict(linewidth=1.5, linestyle='-', color='#ff0808'), showmeans=False, showfliers=False)

        # Scatter plot
        palette_df_5 = ['#ee6100', 'g', 'b', 'y']
        palette_df_15 = ['#ee6100', 'g', 'b', 'y']

        for i, size in enumerate(train_sizes):
            xs_df_5 = np.random.normal(positions_df_5[i], 0.04, len(df_5_boxplot_data[i]))
            xs_df_15 = np.random.normal(positions_df_15[i], 0.04, len(df_15_boxplot_data[i]))

            ax.scatter(xs_df_5, df_5_boxplot_data[i], alpha=0.2, color=palette_df_5[i], label='Prediction balls')
            ax.scatter(xs_df_15, df_15_boxplot_data[i], alpha=0.2, color=palette_df_15[i], marker='^', label='Split-conformal')

        sns.despine(bottom=True)  # Remove right and top axis lines
        sns.set_style("whitegrid")
        ax.set_xticks(range(len(train_sizes)))
        ax.set_xticklabels([str(size) for size in train_sizes], fontsize=17)

        if alpha_level == 0.01:
            ax.set_ylim(0.86, 1.001)
        elif alpha_level == 0.05:
            ax.set_ylim(0.75, 1.001)
        else:
            ax.set_ylim(0.5, 1)

        ax.set_xlabel('Training sample size', fontsize=17)
        ax.set_ylabel('Coverage', fontsize=17)
        ax.tick_params(labelsize=17)
        ax.axhline(y=1-alpha_level, color='black', linestyle='dashed')
        ax.grid(False)

        # Custom legend
        legend_handles = []
        legend_handles.append(mlines.Line2D([], [], color='gray', marker='o', linestyle='none', markersize=10, label=r'$d = 5$'))
        legend_handles.append(mlines.Line2D([], [], color='gray', marker='^', linestyle='none', markersize=10, label=r'$d = 15$'))

        ax.legend(handles=legend_handles, loc='lower right', fontsize=13)
    
    plt.show()
    fig.tight_layout()

def spd_radius_analysis(SPD_coverage_df, save_individual=True):
    """Create radius boxplots for SPD data for all three metrics."""
    print("=== SPD Radius Analysis ===")
    
    # Define plotting style
    boxprops = dict(linestyle='-', linewidth=1.5, color='#000000')
    whiskerprops = dict(color='#000000')
    capprops = dict(color='#000000')
    medianprops = dict(linewidth=1.5, linestyle='-', color='#ff0808')
    
    # Process each metric
    for metric, quantile_col in [('ai', 'ai_OOB_quantile'), ('le', 'le_OOB_quantile'), ('lc', 'lc_OOB_quantile')]:
        print(f"\n{metric.upper()} metric")
        
        # Split the OOB_quantile array by alpha levels
        radius_alpha_01 = SPD_coverage_df.copy()
        radius_alpha_05 = SPD_coverage_df.copy()
        radius_alpha_1 = SPD_coverage_df.copy()
        
        radius_alpha_01['OOB_quantile'] = SPD_coverage_df[quantile_col].apply(lambda x: x[0])
        radius_alpha_05['OOB_quantile'] = SPD_coverage_df[quantile_col].apply(lambda x: x[1])
        radius_alpha_1['OOB_quantile'] = SPD_coverage_df[quantile_col].apply(lambda x: x[2])

        fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor="white")
        count = 0

        for data, alpha_level in zip(
            [radius_alpha_01, radius_alpha_05, radius_alpha_1], 
            [0.01, 0.05, 0.1]):
            
            coverage_df = data
            
            # Extract unique train sizes and degrees of freedom
            train_sizes = sorted(coverage_df['train_size'].unique())
            dfs = coverage_df['df'].unique()
            
            # Prepare the data for boxplots
            grouped_data = [
                [coverage_df.loc[(coverage_df['df'] == df) & (coverage_df['train_size'] == N), 'OOB_quantile']
                    for N in train_sizes]
                for df in dfs ]

            # Plotting
            ax = axes[count]
            count += 1

            palette = ['#ee6100', 'g', 'b', 'y']  # Generate unique colors

            for i, group in enumerate(grouped_data):
                # In this loop, select the degrees of freedom
                base_position = 1 + i * (len(train_sizes) + 1)  # spacing between groups

                for j, ts_data in enumerate(group):
                    # In this loop, select the train sizes. ts_data is the dataset for train size and df
                    pos = base_position + j
                    ax.boxplot(ts_data, positions=[pos], widths=.9, notch=False, whiskerprops=whiskerprops,
                              capprops=capprops, showfliers=False, medianprops=medianprops, showmeans=False) 

                    for x, val in zip(np.random.normal(pos, 0.14, ts_data.shape[0]), ts_data):
                        ax.scatter(x, val, alpha=0.2, color=palette[j])

            sns.despine(bottom=True)  # removes right and top axis lines
            sns.set_style("whitegrid")

            # Formatting
            ax.set_xticks(
                ticks=[1 + i * (len(train_sizes) + 1) + (len(train_sizes) - 1) / 2 for i in range(len(dfs))],
                labels=dfs
            )
            ax.set_xlim(0, len(dfs) * (len(train_sizes) + 1))
                
            ax.set_xlabel(r'$d$', fontsize=17)
            ax.set_ylabel('Radius', fontsize=17)
            ax.tick_params(axis='x', labelsize=17)
            ax.tick_params(axis='y', labelsize=17)
            legend_handles = [
                mpatches.Patch(color=palette[j], label=f'Train size: {train_sizes[j]}') 
                for j in range(len(train_sizes))
            ]
            ax.legend(handles=legend_handles, loc='upper right', fontsize=13)
            ax.grid(False)
            fig.tight_layout()
            
            # Save the plot
            output_dir = ROOT_DIR / 'results_plots'
            output_dir.mkdir(exist_ok=True)
            filename = output_dir / f'{metric}_SPD_radius_vs_df_{alpha_level:.2f}.png'
            fig.savefig(filename, bbox_inches='tight', format='png', dpi=75, transparent=True)

        plt.show()

# ================================
# ADDITIONAL PLOTTING FUNCTIONS FROM plots.ipynb
# ================================

def plot_ellipse(mat, ax, xy=(0,0), scale_factor=1, edgecolor='red', 
                facecolor='None', linewidth=2, alpha=1):
    """Plot ellipse from 2x2 covariance matrix."""
    eigenvalues, eigenvectors = np.linalg.eig(mat)
    theta = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    ellipse = Ellipse(xy=xy,
                      width=scale_factor*np.sqrt(eigenvalues[0]),
                      height=scale_factor*np.sqrt(eigenvalues[1]),
                      angle=theta,
                      edgecolor=edgecolor,
                      facecolor=facecolor,
                      lw=linewidth,
                      alpha=alpha)
    ax.add_patch(ellipse)

def Sigma_t(t_array, Sigma_array):
    """Provides an array with the matrices given by a regression model that interpolates between matrices."""  
    t_array = np.array(t_array)
    t_array = t_array[:, None, None]
    return np.where(np.floor(t_array + 1/2) % 2 == 0, 
                    np.cos(np.pi*t_array)**2 * Sigma_array[0] + (1 - np.cos(np.pi*t_array)**2) * Sigma_array[1], 0) + \
           np.where(np.floor(t_array + 1/2) % 2 == 1, 
                    (1 - np.cos(np.pi*t_array)**2) * Sigma_array[1] + np.cos(np.pi*t_array)**2 * Sigma_array[2], 0)

def sim_regression_matrices(Sigmas, t, df=2):
    """Simulate regression matrices using Wishart distribution."""
    t = np.array(t)
    q = Sigmas[0].shape[0]
    c_dq = 2 * np.exp((1 / q) * sum(digamma((df - np.arange(1, q + 1) + 1) / 2)))
    
    sigma_t = Sigma_t(t, Sigmas)
    sample_Y = [wishart(df=df, scale=sigma_t[k]/c_dq).rvs(size=1) for k in range(t.shape[0])]
    return {'t': t, 'y': sample_Y}

def m_0_sphere(theta, mu):
    """Compute the regression mean on S^2."""
    theta = np.asarray(theta)
    mu = np.asarray(mu)
    assert mu.shape == (2,) and np.isclose(np.linalg.norm(mu), 1), "mu must be a unit vector in R^2"
    
    x1 = np.cos(theta)
    x2 = np.sin(theta) * mu[0]
    x3 = np.sin(theta) * mu[1]
    
    return np.column_stack((x1, x2, x3))

def simulate_sphere_data(kappa, mu, theta_samples):
    """Generate samples from the von Mises-Fisher distribution on S^2."""
    mean_directions = m_0_sphere(theta_samples, mu)
    samples = [vonmises_fisher(mean, kappa).rvs(200) for mean in mean_directions]
    return theta_samples, np.array(samples)

def rotate_north_pole_to(q):
    """Rotate north pole to target point q."""
    p = np.array([0, 0, 1])
    q = q / np.linalg.norm(q)

    v = np.cross(p, q)
    c = np.dot(p, q)
    
    if np.linalg.norm(v) < 1e-10:
        if c > 0:
            return np.eye(3)
        else:
            
            return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

    s = np.linalg.norm(v)
    v_hat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    
    R = np.eye(3) + v_hat + v_hat @ v_hat * ((1 - c) / s**2)
    return R

def create_hyperboloid_visualization():
    """Create hyperboloid visualization with regression curve and data points."""
        
    # Read hyperboloid points
    try:
        hyp_points_path = ROOT_DIR / 'simulations_H2' / 'H2_dibujo_repst200_kappa200.csv'
        if not hyp_points_path.exists():
            print(f"Warning: Hyperboloid data file not found at {hyp_points_path}")
            return
            
        hyp_points = pd.read_csv(hyp_points_path)
        if 'Unnamed: 0' in hyp_points.columns:
            hyp_points.drop(columns=['Unnamed: 0'], inplace=True)
        hyp_points.columns = ['t', 'V1', 'V2', 'V3']
    except Exception as e:
        print(f"Warning: Could not load hyperboloid data: {e}")
        return

    # Create hyperboloid grid
    u = np.linspace(0, .8, 1000)
    v = np.linspace(0, 2*np.pi, 1000)
    u, v = np.meshgrid(u, v)
    
    x_grid = np.cosh(np.abs(u))
    y_grid = np.sinh(np.abs(u)) * np.sign(u) * np.cos(v)
    z_grid = np.sinh(np.abs(u)) * np.sign(u) * np.sin(v)
    
    # Define regression curve
    theta = np.linspace(-.8, .8, 10000)
    mu = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    
    fig = plt.figure(constrained_layout=True, figsize=(7, 7))
    ax = plt.axes(projection='3d', computed_zorder=False)
    
    x_curve = np.cosh(np.abs(theta))
    y_curve = np.sinh(np.abs(theta)) * np.sign(theta) * mu[0]
    z_curve = np.sinh(np.abs(theta)) * np.sign(theta) * mu[1]
    
    ax.plot_wireframe(x_grid, y_grid, z_grid, color='lightblue', alpha=0.3, zorder=0, linewidth=0.85)
    ax.plot(x_curve, y_curve, z_curve, color='black', linewidth=0.75, label=r'$\boldsymbol{m}_0(\theta)$', zorder=1)
    
    # Plot data points
    cmap = plt.cm.rainbow
    for i in range(5):
        color = cmap(i/4)
        if i == 0:
            p = 0.01
        elif i == 1:
            p = 0.25
        elif i == 2:
            p = 0.5
        elif i == 3:
            p = 0.75
        else:
            p = 0.99
        ax.plot(hyp_points['V1'][i*200:(i+1)*200], 
                hyp_points['V2'][i*200:(i+1)*200], 
                hyp_points['V3'][i*200:(i+1)*200], 
                marker='.', color=color, lw=0, markersize=3, 
                label=f'$\\theta=\\frac{{1}}{{4}} \\Phi^{{-1}}({p})$', alpha=0.25, zorder=0)
    
    # Add center points
    center_values = [0.5815870, 0.1686224, 0, -0.1686224, -0.5815870]
    for val in center_values:
        ax.plot(np.cosh(abs(val)), np.sign(val) * np.sinh(abs(val)) * mu[0], 
                np.sign(val) * np.sinh(abs(val)) * mu[1], 
                color='black', marker='.', lw=0, markersize=5, zorder=1)
    
    ax.set_axis_off()
    ax.set_ylim(-1, 1)
    ax.view_init(elev=15, azim=-10, roll=0)
    ax.grid(False)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = ROOT_DIR / 'results_plots'
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / 'hyperboloid_curve_visualization.png'
    fig.savefig(filename, bbox_inches='tight', dpi=200, format='png', transparent=True)
    plt.show()

def create_sphere_visualization():
    """Create sphere visualization with regression curve and data points."""
    # Setup parameters
    mu = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    np.random.seed(1000)
    sample_size = 1000
    kappa = 200
    theta_centers = np.array([0, 2*np.pi/5, 4*np.pi/5, 6*np.pi/5, 8*np.pi/5])
    
    # Generate data
    theta, Y = simulate_sphere_data(kappa, mu, theta_centers)
    Y = Y.reshape(-1, 3)
    
    theta_samples = np.linspace(0, 2*np.pi, 1000)
    
    fig = plt.figure(constrained_layout=True, figsize=(7, 7))
    ax = plt.axes(projection='3d', computed_zorder=False)
    
    # Create sphere wireframe
    u = np.linspace(0, np.pi, 100)
    v = np.linspace(0, 2 * np.pi, 100)
    u, v = np.meshgrid(u, v)
    
    x_sphere = np.sin(u) * np.cos(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(u)
    
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.3, zorder=0)
    
    # Plot regression curve
    curve = m_0_sphere(theta_samples, mu)
    ax.plot(curve[:,0], curve[:,1], curve[:,2], color='black', label=r'$\boldsymbol{m}_0(\theta)$', zorder=1)
    
    # Plot data points
    cmap = plt.cm.rainbow
    theta_labels = ['$\\theta=0$', '$\\theta=\\frac{2\pi}{5}$', '$\\theta=\\frac{4\pi}{5}$', 
                    '$\\theta=\\frac{6\pi}{5}$', '$\\theta=\\frac{8\pi}{5}$']
    
    for i in range(5):
        color = cmap(i/4)
        ax.plot(Y[i*200:(i+1)*200, 0], Y[i*200:(i+1)*200, 1], Y[i*200:(i+1)*200, 2], 
                marker='.', color=color, lw=0, markersize=5, label=theta_labels[i], alpha=0.25, zorder=0)
    
    # Plot center points
    ax.plot(np.cos(theta), np.sin(theta) * mu[0], np.sin(theta) * mu[1], 
            color='black', marker='.', lw=0, markersize=10, zorder=1)
    
    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=90)
    ax.set_xlim(-0.3, .3)
    ax.set_ylim(0, 1)
    ax.set_zlim(-.75, 1.2)
    Axes3D.set_aspect(ax, 'equal')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = ROOT_DIR / 'results_plots'
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / 'sphere_curve_visualization.png'
    fig.savefig(filename, bbox_inches='tight', format='png', dpi=75, transparent=True)
    plt.show()

def create_spd_interpolation_plot():
    """Create SPD matrices interpolation visualization."""
    np.random.seed(1000)
    
    # Define the matrices to interpolate
    Sigma_1 = np.array([[1, -0.6], [-0.6, 0.5]])
    Sigma_2 = np.array([[1, 0], [0, 1]])
    Sigma_3 = np.array([[0.5, 0.4], [0.4, 1]])
    
    Sigmas = (Sigma_1, Sigma_2, Sigma_3)
    
    q = 2
    df = 15
    c_dq = 2 * np.exp((1 / q) * sum(digamma((df - np.arange(1, q + 1) + 1) / 2)))
    sample = sim_regression_matrices(Sigmas=Sigmas, t=np.linspace(start=0, stop=1, num=11), df=df)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    color_palette = get_cmap('hsv')
    
    for i in range(11):
        true_Sigma = Sigma_t(t_array=np.array([sample['t'][i]]), Sigma_array=Sigmas)[0]
        
        # Plot sample ellipses
        for k in range(30):
            plot_ellipse(wishart(df=df, scale=1/c_dq * true_Sigma).rvs(size=1), 
                        ax=ax, xy=(20*i/100, 0.5), scale_factor=1/5, 
                        edgecolor=color_palette(20*i), alpha=0.1)
        
        # Plot true ellipse
        plot_ellipse(true_Sigma, ax=ax, xy=(20*i/100, 0.5), scale_factor=1/5, edgecolor='black')
    
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-0.2, 2.15)
    ax.set_ylim(0.2, 0.8)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    ax.axhline(y=0.5, linestyle='dashed', color='black', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = ROOT_DIR / 'results_plots'
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / 'SPD_interpolation_plot.png'
    fig.savefig(filename, bbox_inches='tight', format='png', transparent=True)
    plt.show()

def create_sphere_prediction_balls():
    """Create sphere population prediction balls visualization."""
    if not PYFRECHET_AVAILABLE:
        print("Warning: pyfrechet not available. Cannot create sphere prediction balls.")
        return
        
    fig = plt.figure(figsize=(7, 7), constrained_layout=True)
    ax = plt.axes(projection='3d', computed_zorder=False)
    
    # Setup parameters
    mu = np.array([0, 1])
    alphas = [0.25, 0.1, 0.01]
    M = Sphere(dim=2)
    
    np.random.seed(1000)
    kappa = 200
    
    # Calculate radii for different alpha values
    R_alpha_values = []
    theta_centers = np.array([np.pi/2])
    for alpha in alphas:
        # Generate MC samples
        MC_samples = simulate_sphere_data(kappa, mu, np.array([theta_centers[0]]))[1][0]
        cartesian_c = np.array([np.cos(theta_centers[0]), 
                               np.sin(theta_centers[0]) * mu[0], 
                               np.sin(theta_centers[0]) * mu[1]])
        
        # Calculate distances
        sphere_distances = [M.d(S, cartesian_c) for S in MC_samples]
        R_alpha_values.append(np.quantile(sphere_distances, 1 - alpha))
    
    # Create points for the spherical caps
    theta_max = np.pi / 15
    u_cap = np.linspace(0, theta_max, 75)
    v_cap = np.linspace(0, 2 * np.pi, 250)
    u_cap, v_cap = np.meshgrid(u_cap, v_cap)
    
    # Convert to Cartesian coordinates
    x_cap = np.sin(u_cap) * np.cos(v_cap)
    y_cap = np.sin(u_cap) * np.sin(v_cap)
    z_cap = np.cos(u_cap)
    Y = np.stack((x_cap, y_cap, z_cap), axis=-1).reshape(-1, 3)
    
    # Plot the sphere
    u = np.linspace(0, np.pi, 100)
    v = np.linspace(0, 2 * np.pi, 100)
    u, v = np.meshgrid(u, v)
    x_sphere = np.sin(u) * np.cos(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(u)
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.3, zorder=0, linewidth=1)
    
    # Rotate to target point
    rotation_matrix = rotate_north_pole_to(np.array([-np.sqrt(0.1), np.sqrt(0.6), np.sqrt(.3)]))
    cartesian_c = np.array([-np.sqrt(0.1), np.sqrt(0.6), np.sqrt(0.3)])
    
    # Plot the rotated spherical caps with different colors
    colors = ['#FFCB42', '#A534BC', '#33C773']
    alphas_plot = [0.25, 0.1, 0.01]
    
    for i, (alpha, color) in enumerate(zip(alphas_plot, colors)):
        for point in Y:
            rotated_point = rotation_matrix @ point
            dist = M.d(point, np.array([0, 0, 1]))
            
            if i == 0 and dist < R_alpha_values[0]:  # α = 0.25
                ax.scatter(1.0015*rotated_point[0], 1.0015*rotated_point[1], 1.0015*rotated_point[2], 
                          color=color, s=0.15, alpha=1, zorder=1)
            elif i == 1 and R_alpha_values[0] <= dist < R_alpha_values[1]:  # α = 0.1
                ax.scatter(1.001*rotated_point[0], 1.001*rotated_point[1], 1.001*rotated_point[2], 
                          color=color, s=0.15, alpha=1, zorder=1)
            elif i == 2 and R_alpha_values[1] <= dist < R_alpha_values[2]:  # α = 0.01
                ax.scatter(rotated_point[0], rotated_point[1], rotated_point[2], 
                          color=color, s=0.15, alpha=1, zorder=1)
    
    # Plot the center point
    ax.scatter([-np.sqrt(0.1)], [np.sqrt(0.6)], [np.sqrt(0.3)], color='black', marker='.', s=100, zorder=1)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#33C773', markersize=8, label=r'$\alpha = 0.01$'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#A534BC', markersize=8, label=r'$\alpha = 0.05$'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFCB42', markersize=8, label=r'$\alpha = 0.1$')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=13)
    
    ax.view_init(elev=20, azim=90)
    ax.set_xlim(-0.3, .3)
    ax.set_ylim(0, 1)
    ax.set_zlim(-.75, 1.2)
    Axes3D.set_aspect(ax, 'equal')
    ax.grid(False)
    ax.set_axis_off()
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = ROOT_DIR / 'results_plots'
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / 'prediction_balls_sphere.png'
    fig.savefig(filename, bbox_inches='tight', format='png', dpi=75, transparent=True)
    plt.show()

def create_hyperboloid_prediction_balls():
    """Create hyperboloid population prediction balls visualization."""
    
    # Center point on the hyperboloid
    mu = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    alphas = np.array([0.25, 0.1, 0.01])
    
    center_t = -0.15
    center_x = np.cosh(center_t)
    center_y = np.sinh(center_t) * mu[0]
    center_z = np.sinh(center_t) * mu[1]
    center_point = np.array([center_x, center_y, center_z])
    
    # Try to load hyperboloid prediction balls data
    try:
        csv_path = ROOT_DIR / 'simulations_H2' / 'H2_dibujo_pball_repst.csv'
        if not csv_path.exists():
            print(f"Warning: Hyperboloid prediction balls data file not found at {csv_path}")
            return
            
        hyp_points = pd.read_csv(csv_path)
        if 'Unnamed: 0' in hyp_points.columns:
            hyp_points.drop(columns=['Unnamed: 0'], inplace=True)
        if 'V1' not in hyp_points.columns:
            hyp_points.columns = ['t', 'V1', 'V2', 'V3']
    except Exception as e:
        print(f"Warning: Could not load hyperboloid prediction balls data: {e}")
        return
    
    M = H2(dim=2)
    
    # Calculate distances and quantiles
    hyp_distances = [M.d(S, center_point) for S in hyp_points.values[:,1:4]]
    R_alpha = np.quantile(hyp_distances, 1 - alphas)
    
    # Create dense grid
    t_range = 1
    t_min = max(0, center_t - t_range)
    t_max = min(1, center_t + t_range)
    
    t_dense = np.linspace(t_min, t_max, 1000)
    phi_dense = np.linspace(0, 2*np.pi, 2000)
    t_grid, phi_grid = np.meshgrid(t_dense, phi_dense)
    
    x_points = np.cosh(t_grid)
    y_points = np.sinh(t_grid) * np.cos(phi_grid)
    z_points = np.sinh(t_grid) * np.sin(phi_grid)
    points = np.stack((x_points.flatten(), y_points.flatten(), z_points.flatten()), axis=1)
    
    distances = np.array([M.d(point, center_point) for point in points])
    distances = distances.reshape(x_points.shape)
    
    # Create hyperboloid wireframe
    u = np.linspace(0, .8, 1000)
    v = np.linspace(0, 2*np.pi, 1000)
    u, v = np.meshgrid(u, v)
    
    x_grid = np.cosh(np.abs(u))
    y_grid = np.sinh(np.abs(u)) * np.sign(u) * np.cos(v)
    z_grid = np.sinh(np.abs(u)) * np.sign(u) * np.sin(v)
    
    fig = plt.figure(constrained_layout=True, figsize=(7, 7))
    ax = plt.axes(projection='3d', computed_zorder=False)
    
    ax.plot_wireframe(x_grid, y_grid, z_grid, color='lightblue', alpha=0.3, zorder=0, linewidth=0.85)
    
    # Plot prediction balls
    colors = ['#FFCB42', '#A534BC', '#33C773']
    scale_factors = [1.0015, 1.001, 1.0]
    
    for i, (alpha, color, scale) in enumerate(zip([0.25, 0.1, 0.01], colors, scale_factors)):
        if i == 0:  # α = 0.25 (innermost)
            mask = (distances < R_alpha[0])
        elif i == 1:  # α = 0.1 (middle)
            mask = (distances >= R_alpha[0]) & (distances < R_alpha[1])
        else:  # α = 0.01 (outermost)
            mask = (distances >= R_alpha[1]) & (distances < R_alpha[2])
        
        if mask.any():
            x_sel = scale * x_points[mask]
            y_sel = scale * y_points[mask]
            z_sel = scale * z_points[mask]
            ax.scatter(x_sel, y_sel, z_sel, color=color, s=0.15, alpha=1, zorder=1)
    
    # Plot center point
    ax.scatter(center_x, center_y, center_z, color='black', marker='.', s=150, zorder=1)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#33C773', markersize=8, label=r'$\alpha = 0.01$'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#A534BC', markersize=8, label=r'$\alpha = 0.05$'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFCB42', markersize=8, label=r'$\alpha = 0.1$')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=13)
    
    ax.set_axis_off()
    ax.set_ylim(-1, 1)
    ax.view_init(elev=15, azim=-10, roll=0)
    ax.grid(False)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = ROOT_DIR / 'results_plots'
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / 'prediction_balls_hyperboloid.png'
    fig.savefig(filename, bbox_inches='tight', dpi=75, format='png', transparent=True)
    plt.show()

def load_sunspot_data():
    """
    Load and concatenate sunspot data from cycles 21, 22, and 23.
    
    Returns:
    --------
    tuple : (X_test, y_test, X_test_pred, y_pred) where
        - X_test, y_test are the true birth-death pairs
        - X_test_pred, y_pred are the birth-prediction pairs
    """
    import os
    
    # Load cycle data
    cycles = [21, 22, 23]
    X_test_list = []
    y_test_list = []
    y_pred_list = []
    
    for cycle in cycles:
        filename = os.path.join(os.getcwd(), "sunspots/results", f"hypothesis_results_cycle_{cycle}.npy")
        sample = np.load(filename, allow_pickle=True).item()
        X_test_list.append(sample['test_data']['X'])
        y_test_list.append(sample['test_data']['y'])
        y_pred_list.append(sample['predictions']['sphere'])
    
    # Concatenate all data (in reverse order: 23, 22, 21)
    X_test = np.concatenate((X_test_list[2], X_test_list[1]), axis=0)
    X_test = np.concatenate((X_test, X_test_list[0]), axis=0)
    
    y_test = np.concatenate((y_test_list[2], y_test_list[1]), axis=0)
    y_test = np.concatenate((y_test, y_test_list[0]), axis=0)
    
    y_pred = np.concatenate((y_pred_list[2], y_pred_list[1]), axis=0)
    y_pred = np.concatenate((y_pred, y_pred_list[0]), axis=0)
    
    return X_test, y_test, X_test, y_pred

def create_sunspot_trajectories_plot(use_predictions=False, save_plot=True):
    """
    Create visualization of sunspot trajectories on the sphere.
    
    Parameters:
    -----------
    use_predictions : bool
        If True, plot birth-to-prediction trajectories. 
        If False, plot birth-to-death trajectories (ground truth).
    save_plot : bool
        Whether to save the plot to file.
    """
    from scipy.spatial import geometric_slerp
    
    # Load data
    X_test, y_test, _, y_pred = load_sunspot_data()
    
    # Choose what to plot
    if use_predictions:
        y_to_plot = y_pred
        filename = 'sunspot_predicted_trajectories.png'
    else:
        y_to_plot = y_test
        filename = 'sunspot_trajectories.png'
    
    # Create figure
    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection='3d', computed_zorder=False)
    
    # Create sphere wireframe
    u = np.linspace(0, np.pi, 150)
    v = np.linspace(0, 2 * np.pi, 150)
    u, v = np.meshgrid(u, v)
    
    x_sphere = np.sin(u) * np.cos(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(u)
    
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.3, zorder=0)
    
    # Define colors
    positive_color = '#14A114'  # Green
    negative_color = '#CA1919'  # Red
    n_points = 150
    
    # Plot geodesics and points
    for birth, death in zip(X_test, y_to_plot):
        # Only plot if birth y-coordinate is positive
        if birth[1] > 0:
            # Compute theta difference
            theta_birth = np.arctan2(birth[1], birth[0])
            theta_death = np.arctan2(death[1], death[0])
            delta_theta = theta_death - theta_birth
            # Normalize to [-pi, pi]
            delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
            
            # Choose color based on direction
            line_color = positive_color if delta_theta > 0 else negative_color
            
            # Create geodesic path using SLERP
            t_vals = np.linspace(0, 1, n_points)
            geodesic_points = geometric_slerp(birth, death, t_vals)
            
            # Draw geodesic path and points
            ax.plot(geodesic_points[:, 0], geodesic_points[:, 1], geodesic_points[:, 2], 
                   color=line_color, linewidth=1, alpha=0.9)
            ax.scatter(birth[0], birth[1], birth[2], color=line_color, linewidth=0, s=3, zorder=2)
            ax.scatter(death[0], death[1], death[2], color=line_color, linewidth=0, s=3, zorder=2)
    
    # Create legend
    legend_elements = [
        Line2D([0], [0], color=positive_color, linestyle='solid', markersize=10, label=r'$\Delta_\theta > 0$'),
        Line2D([0], [0], color=negative_color, linestyle='solid', markersize=10, label=r'$\Delta_\theta < 0$')
    ]
    
    # Set view and styling
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(0, 1)
    ax.set_zlim(-0.75, 1.2)
    ax.view_init(elev=20, azim=90)
    Axes3D.set_aspect(ax, 'equal')
    ax.grid(False)
    ax.set_axis_off()
    fig.tight_layout()
    
    # Save if requested
    if save_plot:
        output_dir = Path("results_plots")
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / filename, format='png', dpi=75, bbox_inches='tight', transparent=True)
    
    plt.show()

def create_sunspot_plots():
    """Create both sunspot trajectory plots (true and predicted)."""
    print("=== Creating sunspot true trajectories plot ===")
    create_sunspot_trajectories_plot(use_predictions=False, save_plot=True)
    
    print("\n=== Creating sunspot predicted trajectories plot ===")
    create_sunspot_trajectories_plot(use_predictions=True, save_plot=True)

def create_all_paper_plots():
    """Create all additional plots from the paper."""
    print("=== Creating hyperboloid visualization ===")
    create_hyperboloid_visualization()
    
    print("\n=== Creating sphere visualization ===")
    create_sphere_visualization()
    
    print("\n=== Creating SPD interpolation plot ===")
    create_spd_interpolation_plot()
    
    print("\n=== Creating sphere prediction balls ===")
    create_sphere_prediction_balls()

    print("\n=== Creating hyperboloid prediction balls ===")
    create_hyperboloid_prediction_balls()
    
    print("\n=== Creating sunspot plots ===")
    create_sunspot_plots()

# ================================
# UNIFIED PUBLIC INTERFACE FUNCTIONS
# ================================

def load_coverage_results(metric_space):
    """
    Load coverage results for the specified metric space.
    
    Parameters:
    -----------
    metric_space : str
        The metric space ('euclidean', 'sphere', 'hyperboloid', 'spd')
    
    Returns:
    --------
    DataFrame : Coverage results for the specified metric space
    """
    if metric_space == 'euclidean':
        return all_coverage_results()
    elif metric_space == 'sphere':
        return sphere_H2_coverage_results(space='sphere')
    elif metric_space == 'hyperboloid':
        return sphere_H2_coverage_results(space='hyperboloid')
    elif metric_space == 'spd':
        return spd_coverage_results()
    else:
        raise ValueError(f"Unknown metric space: {metric_space}. Must be one of: 'euclidean', 'sphere', 'hyperboloid', 'spd'")

def create_type_i_tables(coverage_df, metric_space):
    """
    Create Type I coverage tables for the specified metric space.
    
    Parameters:
    -----------
    coverage_df : DataFrame
        Coverage results dataframe
    metric_space : str
        The metric space ('euclidean', 'sphere', 'hyperboloid', 'spd')
    
    Returns:
    --------
    DataFrame or tuple : Type I coverage tables
    """
    if metric_space == 'euclidean':
        return euclidean_type_i_analysis(coverage_df)
    elif metric_space in ['sphere', 'hyperboloid']:
        return sphere_H2_type_i_analysis(coverage_df)
    elif metric_space == 'spd':
        return spd_type_i_analysis(coverage_df)
    else:
        raise ValueError(f"Unknown metric space: {metric_space}. Must be one of: 'euclidean', 'sphere', 'hyperboloid', 'spd'")

def create_type_ii_plots(coverage_df, metric_space, save_individual=True):
    """
    Create Type II plots for the specified metric space.
    
    Parameters:
    -----------
    coverage_df : DataFrame
        Coverage results dataframe
    metric_space : str
        The metric space ('euclidean', 'sphere', 'hyperboloid', 'spd')
    save_individual : bool
        Whether to save individual plots in addition to the combined display
    """
    if metric_space == 'euclidean':
        euclidean_type_ii_analysis(coverage_df, save_individual=save_individual)
    elif metric_space in ['sphere', 'hyperboloid']:
        sphere_H2_type_ii_analysis(coverage_df, save_individual=save_individual)
    elif metric_space == 'spd':
        spd_type_ii_analysis(coverage_df, save_individual=save_individual)
    else:
        raise ValueError(f"Unknown metric space: {metric_space}. Must be one of: 'euclidean', 'sphere', 'hyperboloid', 'spd'")

def create_type_iii_tables(coverage_df, metric_space):
    """
    Create Type III coverage tables for the specified metric space.
    
    Parameters:
    -----------
    coverage_df : DataFrame
        Coverage results dataframe
    metric_space : str
        The metric space ('euclidean', 'sphere', 'hyperboloid', 'spd')
    
    Returns:
    --------
    DataFrame or tuple : Type III coverage tables
    """
    if metric_space == 'euclidean':
        return euclidean_type_iii_analysis(coverage_df)
    elif metric_space in ['sphere', 'hyperboloid']:
        return sphere_H2_type_iii_analysis(coverage_df)
    elif metric_space == 'spd':
        return create_spd_type_iii_tables(coverage_df)
    else:
        raise ValueError(f"Unknown metric space: {metric_space}. Must be one of: 'euclidean', 'sphere', 'hyperboloid', 'spd'")

def create_type_iv_plots(coverage_df, metric_space, save_individual=True):
    """
    Create Type IV plots for the specified metric space.
    
    Parameters:
    -----------
    coverage_df : DataFrame
        Coverage results dataframe
    metric_space : str
        The metric space ('euclidean', 'sphere', 'hyperboloid', 'spd')
    save_individual : bool
        Whether to save individual plots in addition to the combined display
    """
    if metric_space == 'euclidean':
        euclidean_type_iv_analysis(coverage_df, save_individual=save_individual)
    elif metric_space in ['sphere', 'hyperboloid']:
        sphere_H2_type_iv_analysis(coverage_df, save_individual=save_individual)
    elif metric_space == 'spd':
        spd_type_iv_analysis(coverage_df, save_individual=save_individual)
    else:
        raise ValueError(f"Unknown metric space: {metric_space}. Must be one of: 'euclidean', 'sphere', 'hyperboloid', 'spd'")

def calculate_mse_comparison(coverage_df, metric_space):
    """
    Calculate MSE comparison for the specified metric space.
    
    Parameters:
    -----------
    coverage_df : DataFrame
        Coverage results dataframe
    metric_space : str
        The metric space ('euclidean', 'sphere', 'hyperboloid', 'spd')
    
    Returns:
    --------
    DataFrame : MSE comparison results (only available for euclidean)
    """
    if metric_space == 'euclidean':
        return euclidean_mse_analysis(coverage_df)
    else:
        raise ValueError(f"MSE comparison only available for 'euclidean' metric space, got: {metric_space}")

def create_radius_plots(coverage_df, metric_space, save_individual=True):
    """
    Create radius boxplots for the specified metric space.
    
    Parameters:
    -----------
    coverage_df : DataFrame
        Coverage results dataframe
    metric_space : str
        The metric space ('sphere', 'hyperboloid', 'spd')
    save_individual : bool
        Whether to save individual plots in addition to the combined display
    """
    if metric_space in ['sphere', 'hyperboloid']:
        sphere_H2_radius_analysis(coverage_df, space=metric_space, save_individual=save_individual)
    elif metric_space == 'spd':
        spd_radius_analysis(coverage_df, save_individual=save_individual)
    else:
        raise ValueError(f"Radius plots only available for 'sphere', 'hyperboloid', and 'spd' metric spaces, got: {metric_space}")

# ================================
# SPD FRÉCHET MEAN ANALYSIS FUNCTIONS
# ================================

def interpolation_matrices(Sigma_1, Sigma_2, Sigma_3, Sigma_4):
    """Provides an array with the matrices given by a regression model."""  
    """The regression starts with Sigma_1 and then goes to Sigma_2 and Sigma_3 and ends in Sigma_4."""
        
    # Define time intervals for interpolation
    t_1 = np.linspace(-1, 0, 200)
    t_1 = t_1[:, None, None]
    t_2 = np.linspace(0, 1, 200)
    t_2 = t_2[:, None, None]
    t_3 = np.linspace(1, 2, 200)
    t_3 = t_3[:, None, None]

    # Interpolate between the matrices
    Sigma_t1 = - t_1 * Sigma_1 + (1 + t_1) * Sigma_2
    Sigma_t2 = (1-t_2) * Sigma_2 + t_2 * Sigma_3
    Sigma_t3 =  (2-t_3) * Sigma_3 + (t_3 - 1) * Sigma_4

    # Concatenate the interpolated matrices
    return np.concatenate([Sigma_t1, Sigma_t2, Sigma_t3])

def generate_random_spd_matrix(q_array, limits_unif=5, seed=1):
    """
    Generate a list of random symmetric positive definite (SPD) matrices.
    
    Parameters:
        q_array (list of int): A list of integers specifying the dimensions of the SPD matrices.
        limits_unif (float): Scaling factor for random matrix generation.
        seed (int): A seed for the random number generator to ensure reproducibility.
    
    Returns:
        list of np.ndarray: A list of SPD matrices of the specified dimensions.
    """
    np.random.RandomState(seed)
    np.random.seed(seed)
    
    q_array = np.array(q_array, dtype=int)
    # Ensure the matrices are symmetric positive definite
    mat = [(np.random.rand(q_array[i], q_array[i])-1/2)*limits_unif for i in range(len(q_array))]
    return [np.dot(mat[i], mat[i].T) for i in range(len(q_array))]

def is_spd(matrix):
    """Check if a matrix is symmetric positive definite."""
    if np.allclose(matrix, matrix.T):
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False
    return False

def log_cholesky_distance(S1, S2):
    """Computes the log-Cholesky distance between two SPD matrices."""
    from scipy.linalg import sqrtm
    
    # Compute Cholesky decompositions
    R1 = np.linalg.cholesky(S1)
    R2 = np.linalg.cholesky(S2)
    
    # Compute differences in lower triangular and diagonal parts
    lower_diff = np.tril(R1, -1) - np.tril(R2, -1)
    diag_diff = np.diag(np.log(np.diag(R1)) - np.log(np.diag(R2)))
    
    # Compute Frobenius norms
    lower_norm = np.linalg.norm(lower_diff, ord='fro')**2
    diag_norm = np.linalg.norm(diag_diff, ord='fro')**2
    
    # Return the log-Cholesky distance
    return np.sqrt(lower_norm + diag_norm)

def affine_invariant_distance(S1, S2):
    """
    Computes the affine-invariant Riemannian metric (AIRM) between two SPD matrices S1 and S2.

    Parameters:
        S1 (ndarray): Symmetric positive definite matrix of shape (n, n).
        S2 (ndarray): Symmetric positive definite matrix of shape (n, n).

    Returns:
        float: The affine-invariant Riemannian metric distance between S1 and S2.
    """
    from scipy.linalg import sqrtm, eigvals
    
    # Compute the matrix S1^{-1/2}
    inv_sqrt_S1 = np.linalg.inv(sqrtm(S1)).T

    # Compute the matrix S1^{-1/2} S2 S1^{-1/2}
    inv_S1_S2 = inv_sqrt_S1 @ S2 @ inv_sqrt_S1
    # Compute the eigenvalues of the matrix
    eigenvalues = eigvals(inv_S1_S2)
    
    # Compute the log of eigenvalues and sum of their squares
    log_eigenvalues = np.log(eigenvalues.real)  # Ensure real part is taken
    distance = np.sqrt(np.sum(log_eigenvalues**2))
    
    return distance

def extrinsic_mean(Sigma, d):
    """Calculate the extrinsic mean."""
    return d * Sigma 

def frechet_mean_ai(Sigma, d):
    """Calculates the Fréchet mean from a Wishart distribution for the affine-invariant metric."""
    from scipy.special import digamma
    
    # Compute the digamma term
    q = Sigma.shape[0]
    digamma_term = (1 / q) * sum([digamma((d - i + 1) / 2) for i in range(1, q + 1)])
    # Return the Fréchet mean
    return 2 * np.exp(digamma_term) * Sigma

def frechet_mean_lc(Sigma, d):
    """Calculates the Fréchet mean from a Wishart distribution for the log-Cholesky metric."""
    from scipy.special import digamma, gamma
    
    # Cholesky factor of Sigma
    L = np.linalg.cholesky(Sigma)

    # Cholesky factor of the Fréchet mean (T)
    q = L.shape[0]

    # Create index arrays for upper and lower triangular parts
    i, j = np.tril_indices(q, -1)

    # Compute diagonal elements
    diag_indices = np.arange(q)
    T_diag = L[diag_indices, diag_indices] * np.sqrt(2) * np.exp(0.5 * digamma((d - diag_indices) / 2))

    # Compute lower triangular elements (excluding diagonal)
    T_lower = L[i, j] * np.sqrt(2) * gamma((d - j + 1) / 2) / gamma((d - j) / 2)

    # Assign results to T
    T = np.zeros_like(L)
    T[diag_indices, diag_indices] = T_diag
    T[i, j] = T_lower
    # Return the Fréchet mean
    return T @ T.T

def simulate_frechet_loss(d_array, Sigma_array, matrices, n_samples=10000, plot_title_suffix=""):
    """ 
    Simulate the Fréchet loss function for multiple (d, q) combinations.
    
    Args:
        d_array (list or np.array): Array of degrees of freedom for the Wishart distribution.
        Sigma_array (list or np.array): Array of SPD matrices.
        matrices (list): List of interpolation matrices for each (d, Sigma) combination.
        n_samples (int): Number of samples from the Wishart distribution.
        plot_title_suffix (str): Additional text for plot identification.
    
    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    sns.set_style("whitegrid")
    fig = plt.figure(facecolor="white", figsize=(7, 7))
    ax = fig.add_subplot(111)
    d_array = np.array(d_array, dtype=int)
    
    k = 0 
    for Sigma in Sigma_array:
        for d in d_array:
            
            q = Sigma.shape[0]
            np.random.seed(1)

            if d < q: 
                continue
            else:
                # Control
                samples = wishart.rvs(df=d, scale=Sigma, size=n_samples)

                # Compute distances log-Cholesky and affine-invariant
                distances_LC = np.array([
                    [log_cholesky_distance(matrix, Y) for Y in samples]
                    for matrix in matrices[k]
                ])
                
                distances_AI = np.array([
                    [affine_invariant_distance(matrix, Y) for Y in samples]
                    for matrix in matrices[k]
                ])

                fm_lc = frechet_mean_lc(Sigma, d)
                functional_fm_lc = np.mean(
                    np.array([
                        log_cholesky_distance(fm_lc, Y)**2 for Y in samples
                    ])
                )

                fm_ai = frechet_mean_ai(Sigma, d)
                functional_fm_ai = np.mean(
                    np.array([
                        affine_invariant_distance(fm_ai, Y)**2 for Y in samples
                    ])
                )

                # Compute Fréchet loss
                normalized_loss_LC = (np.mean(distances_LC**2, axis=1) - functional_fm_lc) / functional_fm_lc 
                normalized_loss_AI = (np.mean(distances_AI**2, axis=1) - functional_fm_ai) / functional_fm_ai

                t_array = np.linspace(-1, 2, 600)

                # Plot the normalized losses
                if len(Sigma_array) == 1:
                    label1 = f"LC, d={d}"
                    label2 = f"AI, d={d}"
                elif len(d_array) == 1:
                    label1 = f"LC, $\Sigma_{k+1}$"
                    label2 = f"AI, $\Sigma_{k+1}$"
                else:
                    label1 = f"LC, d={d}, q={q}, $\Sigma_{k+1}$"
                    label2 = f"AI, d={d}, q={q}, $\Sigma_{k+1}$"
                
                linestyles = ['-', '--', '-.', ':']
                ax.plot(t_array, normalized_loss_LC, label=label1, color='red', linestyle=linestyles[k % len(linestyles)])
                ax.plot(t_array, normalized_loss_AI, label=label2, color='blue', linestyle=linestyles[k % len(linestyles)])
                k += 1        

    ax.set_xticks(ticks=[-1, 0, 1, 2], labels=[r"$\boldsymbol{M}_{\mathrm{Ext}}$", r"$\boldsymbol{M}_{\mathrm{AI}}$", r"$\boldsymbol{M}_{\mathrm{LC}}$", r"$\boldsymbol{M}_{\mathrm{Ext}}$"])
    ax.tick_params(labelsize=17)
    ax.set_xlabel(r"$\boldsymbol{M}(t)$", fontsize=17)
    ax.set_ylabel("Normalized Fréchet Loss", fontsize=17)
    ax.legend(loc='upper center', ncol=2, fontsize=14)
    ax.grid(True)
    fig.tight_layout()
    
    # Save to results_plots directory
    output_dir = ROOT_DIR / "results_plots"
    output_dir.mkdir(exist_ok=True)
    
    if plot_title_suffix:
        filename = f"frechet_mean_spd_{plot_title_suffix}.png"
    else:
        filename = "frechet_mean_spd_analysis.png"
    
    fig.savefig(output_dir / filename, bbox_inches='tight', dpi=300)
    plt.show()
    return fig

def create_spd_frechet_plots():
    """
    Create all three SPD Fréchet mean analysis plots.
    
    Returns:
        tuple: Three matplotlib figures for the different analysis scenarios.
    """
    
    # Setup data
    d_array = [5, 15, 25]  
    d1_array = [5]
    d2_array = [15]
    d3_array = [25]
    
    # Generate SPD matrices
    list_sigmas = generate_random_spd_matrix(q_array=np.array([2, 2, 6, 6, 10, 10]), seed=0)
    
    # Create interpolation matrices for all combinations
    list_interp = []
    for i in range(len(list_sigmas)):
        for j in range(len(d_array)):
            if d_array[j] >= list_sigmas[i].shape[0]:
                list_interp += [interpolation_matrices(
                    extrinsic_mean(np.array(list_sigmas[i]), d=d_array[j]), 
                    frechet_mean_ai(np.array(list_sigmas[i]), d=d_array[j]), 
                    frechet_mean_lc(np.array(list_sigmas[i]), d=d_array[j]), 
                    extrinsic_mean(np.array(list_sigmas[i]), d=d_array[j])
                )]
    
    # Extract specific combinations for plots
    sigmas_q_2 = [list_sigmas[i] for i in [0, 1]]
    sigmas_q_6 = [list_sigmas[i] for i in [2, 3]]
    
    interp_d_15_q_2 = [list_interp[i] for i in [1, 4]]
    interp_d_15_q_6 = [list_interp[i] for i in [6, 8]]
    interp_sigma_2_q_6 = [list_interp[i] for i in [8, 9]]
    
    # Create the three main plots
    print("\n--- Plot 1: d=15, q=2, different sigmas ---")
    fig1 = simulate_frechet_loss(
        d_array=d2_array, 
        Sigma_array=sigmas_q_2, 
        matrices=interp_d_15_q_2, 
        n_samples=25000,
        plot_title_suffix="d15_q2_different_sigmas"
    )
    
    print("\n--- Plot 2: d=15, q=6, different sigmas ---")
    fig2 = simulate_frechet_loss(
        d_array=d2_array, 
        Sigma_array=sigmas_q_6, 
        matrices=interp_d_15_q_6, 
        n_samples=25000,
        plot_title_suffix="d15_q6_different_sigmas"
    )
    
    print("\n--- Plot 3: q=6, one sigma, different d values ---")
    fig3 = simulate_frechet_loss(
        d_array=d_array, 
        Sigma_array=np.array([sigmas_q_6[1]]), 
        matrices=interp_sigma_2_q_6, 
        n_samples=25000,
        plot_title_suffix="q6_one_sigma_different_d"
    )
    
    return fig1, fig2, fig3