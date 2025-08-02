#!/usr/bin/env python3
"""
Setup script to verify the repository is ready for reproducing results.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is adequate."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 
        'scikit-learn', 'scipy', 'joblib', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements_diego.txt")
        return False
    
    return True

def check_data_directories():
    """Check if simulation result directories exist."""
    root_dir = Path(__file__).parent
    
    required_dirs = [
        'simulations_euc/results',
        'simulations_sphere/results', 
        'simulations_H2/results',
        'simulations_SPD/results'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = root_dir / dir_path
        if full_path.exists() and any(full_path.glob('*.npy')):
            print(f"✅ {dir_path} (contains {len(list(full_path.glob('*.npy')))} result files)")
        else:
            print(f"❌ {dir_path} (missing or empty)")
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\n❌ Missing result directories: {', '.join(missing_dirs)}")
        print("You may need to run the simulation scripts first.")
        return False
        
    return True

def check_pyfrechet():
    """Check if pyfrechet library can be imported."""
    root_dir = Path(__file__).parent
    sys.path.insert(0, str(root_dir))
    
    try:
        from pyfrechet.metric_spaces import MetricSpace, Euclidean, Sphere, H2
        from pyfrechet.regression.bagged_regressor import BaggedRegressor
        print("✅ pyfrechet library")
        return True
    except ImportError as e:
        print(f"❌ pyfrechet library: {e}")
        return False

def main():
    """Run all checks."""
    print("🔍 Checking repository setup for reproducibility...\n")
    
    checks = [
        ("Python version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Pyfrechet library", check_pyfrechet),
        ("Simulation results", check_data_directories),
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 Repository is ready! Run 'jupyter notebook main_results.ipynb' to generate results.")
    else:
        print("⚠️  Some checks failed. Please fix the issues above before proceeding.")
        
    print("="*50)

if __name__ == "__main__":
    main()
