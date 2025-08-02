"""
Utilities for reproducible simulation results across different metric spaces.
This module provides common functions for loading data, generating plots, and creating tables.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Get the root directory of the project (where this file is located)
ROOT_DIR = Path(__file__).parent.absolute()

def setup_paths():
    """Setup paths for importing pyfrechet modules."""
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

# Set up paths immediately when module is imported
setup_paths()

def get_simulation_dir(metric_space: str) -> Path:
    """Get the simulation directory for a given metric space."""
    valid_spaces = ['euc', 'sphere', 'H2', 'SPD']
    if metric_space not in valid_spaces:
        raise ValueError(f"metric_space must be one of {valid_spaces}")
    
    return ROOT_DIR / f"simulations_{metric_space}"

def get_results_dir(metric_space: str) -> Path:
    """Get the results directory for a given metric space."""
    sim_dir = get_simulation_dir(metric_space)
    return sim_dir / "results"

def load_coverage_results_euclidean() -> pd.DataFrame:
    """Load coverage results for Euclidean space."""
    results_dir = get_results_dir('euc')
    
    coverage_df = pd.DataFrame(columns=[
        'sample_index', 'train_size', 'sigma', 'OOB_quantile', 
        'pb_i_cov', 'pb_ii_cov', 'pb_iii_cov', 'pb_iv_cov',
        'conf_i_cov', 'conf_ii_cov', 'conf_iii_cov', 'conf_iv_cov',
        'pb_mse', 'conf_mse', 'quantile', 'pb_time', 'conf_time'
    ])
    
    for file in results_dir.glob('*.npy'):
        try:
            result = np.load(file, allow_pickle=True).item()
            file_parts = file.stem.split('_')
            
            row_data = {
                'sample_index': int(file_parts[1][4:]),
                'train_size': int(file_parts[2][1:]),
                'sigma': file_parts[3][5:],
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
            }
            
            coverage_df = pd.concat([
                coverage_df, 
                pd.DataFrame(row_data, index=pd.RangeIndex(0, 1))
            ], ignore_index=True)
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    coverage_df['train_size'] = coverage_df['train_size'].astype('category')
    coverage_df['sigma'] = coverage_df['sigma'].astype('category')
    return coverage_df

def load_coverage_results_sphere() -> pd.DataFrame:
    """Load coverage results for Sphere."""
    results_dir = get_results_dir('sphere')
    
    coverage_df = pd.DataFrame(columns=[
        'sample_index', 'train_size', 'kappa', 'OOB_quantile', 
        'i_cov', 'ii_cov', 'iii_cov', 'iv_cov'
    ])
    
    for file in results_dir.glob('*.npy'):
        try:
            result = np.load(file, allow_pickle=True).item()
            file_parts = file.stem.split('_')
            
            row_data = {
                'sample_index': int(file_parts[1][4:]),
                'train_size': int(file_parts[2][1:]),
                'kappa': file_parts[3][5:],
                'i_cov': [result['i_cov']],
                'ii_cov': [result['ii_cov']],
                'iii_cov': [result['iii_cov']],
                'iv_cov': [result['iv_cov']],
                'OOB_quantile': [result['OOB_quantile']],
            }
            
            coverage_df = pd.concat([
                coverage_df, 
                pd.DataFrame(row_data, index=pd.RangeIndex(0, 1))
            ], ignore_index=True)
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    coverage_df['train_size'] = coverage_df['train_size'].astype('category')
    coverage_df['kappa'] = coverage_df['kappa'].astype('category')
    return coverage_df

def load_coverage_results_h2() -> pd.DataFrame:
    """Load coverage results for Hyperboloid (H2)."""
    results_dir = get_results_dir('H2')
    
    coverage_df = pd.DataFrame(columns=[
        'sample_index', 'train_size', 'kappa', 'OOB_quantile', 
        'i_cov', 'ii_cov', 'iii_cov', 'iv_cov'
    ])
    
    for file in results_dir.glob('*.npy'):
        try:
            result = np.load(file, allow_pickle=True).item()
            file_parts = file.stem.split('_')
            
            row_data = {
                'sample_index': int(file_parts[1][4:]),
                'train_size': int(file_parts[2][1:]),
                'kappa': file_parts[3][5:],
                'i_cov': [result['i_cov']],
                'ii_cov': [result['ii_cov']],
                'iii_cov': [result['iii_cov']],
                'iv_cov': [result['iv_cov']],
                'OOB_quantile': [result['OOB_quantile']],
            }
            
            coverage_df = pd.concat([
                coverage_df, 
                pd.DataFrame(row_data, index=pd.RangeIndex(0, 1))
            ], ignore_index=True)
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    coverage_df['train_size'] = coverage_df['train_size'].astype('category')
    coverage_df['kappa'] = coverage_df['kappa'].astype('category')
    return coverage_df

def load_coverage_results_spd() -> pd.DataFrame:
    """Load coverage results for SPD manifolds."""
    results_dir = get_results_dir('SPD')
    
    coverage_df = pd.DataFrame(columns=[
        'sample_index', 'train_size', 'df', 
        'ai_i_cov', 'ai_ii_cov', 'ai_iii_cov', 'ai_iv_cov',
        'lc_i_cov', 'lc_ii_cov', 'lc_iii_cov', 'lc_iv_cov', 
        'le_i_cov', 'le_ii_cov', 'le_iii_cov', 'le_iv_cov',
        'ai_OOB_quantile', 'lc_OOB_quantile', 'le_OOB_quantile'
    ])
    
    for file in results_dir.glob('*.npy'):
        try:
            result = np.load(file, allow_pickle=True).item()
            file_parts = file.stem.split('_')
            
            row_data = {
                'sample_index': int(file_parts[1][4:]),
                'train_size': int(file_parts[2][1:]),
                'df': file_parts[3][2:],
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
            }
            
            coverage_df = pd.concat([
                coverage_df, 
                pd.DataFrame(row_data, index=pd.RangeIndex(0, 1))
            ], ignore_index=True)
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    coverage_df['train_size'] = coverage_df['train_size'].astype('category')
    coverage_df['df'] = coverage_df['df'].astype('category')
    return coverage_df

def create_coverage_table(df: pd.DataFrame, metric_space: str, coverage_type: str) -> pd.DataFrame:
    """Create a coverage probability table for a specific metric space and coverage type."""
    
    if metric_space == 'euclidean':
        # For Euclidean, we have both PB and conformal methods
        if 'pb' in coverage_type.lower():
            cov_col = f'pb_{coverage_type.split("_")[-1]}_cov'
        else:
            cov_col = f'conf_{coverage_type.split("_")[-1]}_cov'
        group_cols = ['train_size', 'sigma']
    elif metric_space in ['sphere', 'h2']:
        cov_col = f'{coverage_type}_cov'
        group_cols = ['train_size', 'kappa']
    elif metric_space == 'spd':
        cov_col = f'{coverage_type}_cov'
        group_cols = ['train_size', 'df']
    
    # Calculate mean coverage probabilities for each confidence level
    results = []
    confidence_levels = ['99%', '95%', '90%']
    
    for group_vals, group_data in df.groupby(group_cols):
        row_result = {}
        
        # Add grouping columns
        for i, col in enumerate(group_cols):
            row_result[col] = group_vals[i] if len(group_cols) > 1 else group_vals
        
        # Calculate coverage for each confidence level
        all_coverage_values = []
        for _, row in group_data.iterrows():
            coverage_values = row[cov_col]
            if hasattr(coverage_values, '__len__') and not isinstance(coverage_values, str):
                all_coverage_values.append(coverage_values)
        
        if all_coverage_values:
            # Convert to numpy array for easier processing
            coverage_array = np.array(all_coverage_values)
            
            # Handle different array shapes
            if coverage_array.ndim == 1:
                # Single confidence level
                row_result['Coverage'] = np.mean(coverage_array)
            else:
                # Multiple confidence levels
                for i, conf_level in enumerate(confidence_levels):
                    if i < coverage_array.shape[1]:
                        row_result[f'Coverage_{conf_level}'] = np.mean(coverage_array[:, i])
        
        results.append(row_result)
    
    result_df = pd.DataFrame(results)
    return result_df

def create_type_ii_plot(df: pd.DataFrame, metric_space: str, method: str = 'pb') -> plt.Figure:
    """Create Type II coverage plot (coverage vs sample size)."""
    
    if metric_space == 'euclidean':
        cov_col = f'{method}_ii_cov'
        param_col = 'sigma'
        param_name = 'σ'
    elif metric_space in ['sphere', 'h2']:
        cov_col = 'ii_cov'
        param_col = 'kappa'
        param_name = 'κ'
    elif metric_space == 'spd':
        cov_col = f'{method}_ii_cov'
        param_col = 'df'
        param_name = 'df'
    
    # Prepare data for plotting
    plot_data = []
    confidence_levels = [0.99, 0.95, 0.9]
    
    for _, row in df.iterrows():
        coverage_values = row[cov_col]
        # Handle both array and single value cases
        if hasattr(coverage_values, '__len__') and len(coverage_values) > 1:
            # Array case - coverage_values is 1D array where each element is for a confidence level
            for i, conf_level in enumerate(confidence_levels):
                if i < len(coverage_values):
                    plot_data.append({
                        'train_size': row['train_size'],
                        param_col: row[param_col],
                        'confidence_level': f'{int(conf_level*100)}%',
                        'coverage': coverage_values[i]  # Direct access to element
                    })
        else:
            # Single value case - assume it's for one confidence level
            plot_data.append({
                'train_size': row['train_size'],
                param_col: row[param_col],
                'confidence_level': '95%',  # Default
                'coverage': coverage_values if np.isscalar(coverage_values) else coverage_values[0]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create plot
    fig, axes = plt.subplots(1, len(plot_df[param_col].unique()), 
                            figsize=(5*len(plot_df[param_col].unique()), 4))
    if len(plot_df[param_col].unique()) == 1:
        axes = [axes]
    
    for i, param_val in enumerate(sorted(plot_df[param_col].unique())):
        subset = plot_df[plot_df[param_col] == param_val]
        
        for conf_level in subset['confidence_level'].unique():
            conf_subset = subset[subset['confidence_level'] == conf_level]
            axes[i].plot(conf_subset['train_size'], conf_subset['coverage'], 
                        marker='o', label=conf_level)
        
        # Add nominal coverage lines
        for coverage in [0.99, 0.95, 0.9]:
            axes[i].axhline(y=coverage, color='gray', linestyle='--', alpha=0.5)
        
        axes[i].set_xlabel('Sample Size')
        axes[i].set_ylabel('Coverage Probability')
        axes[i].set_title(f'{param_name} = {param_val}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0.8, 1.0)  # Set reasonable y-axis limits
    
    plt.tight_layout()
    return fig

def create_type_iv_plot(df: pd.DataFrame, metric_space: str, method: str = 'pb') -> plt.Figure:
    """Create Type IV coverage plot (coverage vs sample size)."""
    
    if metric_space == 'euclidean':
        cov_col = f'{method}_iv_cov'
        param_col = 'sigma'
        param_name = 'σ'
    elif metric_space in ['sphere', 'h2']:
        cov_col = 'iv_cov'
        param_col = 'kappa'
        param_name = 'κ'
    elif metric_space == 'spd':
        cov_col = f'{method}_iv_cov'
        param_col = 'df'
        param_name = 'df'
    
    # Prepare data for plotting
    plot_data = []
    confidence_levels = [0.99, 0.95, 0.9]
    
    for _, row in df.iterrows():
        coverage_values = row[cov_col]
        # Handle both array and single value cases
        if hasattr(coverage_values, '__len__') and len(coverage_values) > 1:
            # Array case - coverage_values is 1D array where each element is for a confidence level
            for i, conf_level in enumerate(confidence_levels):
                if i < len(coverage_values):
                    plot_data.append({
                        'train_size': row['train_size'],
                        param_col: row[param_col],
                        'confidence_level': f'{int(conf_level*100)}%',
                        'coverage': coverage_values[i]  # Direct access to element
                    })
        else:
            # Single value case - assume it's for one confidence level
            plot_data.append({
                'train_size': row['train_size'],
                param_col: row[param_col],
                'confidence_level': '95%',  # Default
                'coverage': coverage_values if np.isscalar(coverage_values) else coverage_values[0]
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create plot
    fig, axes = plt.subplots(1, len(plot_df[param_col].unique()), 
                            figsize=(5*len(plot_df[param_col].unique()), 4))
    if len(plot_df[param_col].unique()) == 1:
        axes = [axes]
    
    for i, param_val in enumerate(sorted(plot_df[param_col].unique())):
        subset = plot_df[plot_df[param_col] == param_val]
        
        for conf_level in subset['confidence_level'].unique():
            conf_subset = subset[subset['confidence_level'] == conf_level]
            axes[i].plot(conf_subset['train_size'], conf_subset['coverage'], 
                        marker='o', label=conf_level)
        
        # Add nominal coverage lines
        for coverage in [0.99, 0.95, 0.9]:
            axes[i].axhline(y=coverage, color='gray', linestyle='--', alpha=0.5)
        
        axes[i].set_xlabel('Sample Size')
        axes[i].set_ylabel('Coverage Probability')
        axes[i].set_title(f'{param_name} = {param_val}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0.8, 1.0)  # Set reasonable y-axis limits
    
    plt.tight_layout()
    return fig

def create_radius_plot(df: pd.DataFrame, metric_space: str, method: str = 'pb') -> plt.Figure:
    """Create radius plots for non-Euclidean spaces."""
    
    if metric_space == 'euclidean':
        raise ValueError("Radius plots are not applicable for Euclidean space")
    
    if metric_space in ['sphere', 'h2']:
        quantile_col = 'OOB_quantile'
        param_col = 'kappa'
        param_name = 'κ'
    elif metric_space == 'spd':
        quantile_col = f'{method}_OOB_quantile'
        param_col = 'df'
        param_name = 'df'
    
    # Prepare data for plotting
    plot_data = []
    confidence_levels = ['99%', '95%', '90%']
    
    for _, row in df.iterrows():
        quantile_values = row[quantile_col]
        # Handle both array and single value cases
        if hasattr(quantile_values, '__len__') and len(quantile_values) > 1:
            for i, level in enumerate(confidence_levels):
                if i < len(quantile_values):
                    radius_val = quantile_values[i] if hasattr(quantile_values[i], '__len__') == False else np.mean(quantile_values[i])
                    plot_data.append({
                        'train_size': row['train_size'],
                        param_col: row[param_col],
                        'confidence_level': level,
                        'radius': radius_val
                    })
        else:
            # Single value case
            plot_data.append({
                'train_size': row['train_size'],
                param_col: row[param_col],
                'confidence_level': '95%',  # Default
                'radius': np.mean(quantile_values) if hasattr(quantile_values, '__len__') else quantile_values
            })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create plot
    fig, axes = plt.subplots(1, len(plot_df[param_col].unique()), 
                            figsize=(5*len(plot_df[param_col].unique()), 4))
    if len(plot_df[param_col].unique()) == 1:
        axes = [axes]
    
    for i, param_val in enumerate(sorted(plot_df[param_col].unique())):
        subset = plot_df[plot_df[param_col] == param_val]
        
        for conf_level in subset['confidence_level'].unique():
            conf_subset = subset[subset['confidence_level'] == conf_level]
            axes[i].plot(conf_subset['train_size'], conf_subset['radius'], 
                        marker='o', label=conf_level)
        
        axes[i].set_xlabel('Sample Size')
        axes[i].set_ylabel('Radius')
        axes[i].set_title(f'{param_name} = {param_val}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
    
    for i, param_val in enumerate(sorted(plot_df[param_col].unique())):
        subset = plot_df[plot_df[param_col] == param_val]
        
        for conf_level in subset['confidence_level'].unique():
            conf_subset = subset[subset['confidence_level'] == conf_level]
            axes[i].plot(conf_subset['train_size'], conf_subset['radius'], 
                        marker='o', label=conf_level)
        
        axes[i].set_xlabel('Sample Size')
        axes[i].set_ylabel('Radius')
        axes[i].set_title(f'{param_name} = {param_val}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def calculate_mse_comparison(df_euc: pd.DataFrame) -> pd.DataFrame:
    """Calculate MSE comparison for Euclidean space."""
    
    mse_data = []
    for _, row in df_euc.iterrows():
        mse_data.append({
            'train_size': row['train_size'],
            'sigma': row['sigma'],
            'pb_mse': np.mean(row['pb_mse']),
            'conf_mse': np.mean(row['conf_mse'])
        })
    
    mse_df = pd.DataFrame(mse_data)
    
    # Calculate relative improvement
    mse_df['relative_improvement'] = (mse_df['conf_mse'] - mse_df['pb_mse']) / mse_df['conf_mse'] * 100
    
    return mse_df.groupby(['train_size', 'sigma']).agg({
        'pb_mse': 'mean',
        'conf_mse': 'mean',
        'relative_improvement': 'mean'
    }).round(4)

def save_plots_and_tables(output_dir: Path):
    """Save all plots and tables to the specified directory."""
    output_dir.mkdir(exist_ok=True)
    
    # Set up matplotlib parameters for publication-quality plots
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
    
    print("Utility functions are ready. Use them in the main notebook to generate results.")
