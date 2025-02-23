import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm
import time
import os

from pde_solver import PoissonSolver
from data_generator import DataGenerator
from experiments import Experiment

def calculate_error_metrics(solution: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    """
    Calculate various error metrics between solution and reference.
    
    Args:
        solution: Computed solution
        reference: Reference solution
        
    Returns:
        Dictionary containing different error metrics
    """
    error = np.abs(solution - reference)
    return {
        'l1': np.mean(error),           # Mean absolute error
        'l2': np.sqrt(np.mean(error**2)), # Root mean square error
        'linf': np.max(error),          # Maximum absolute error
        'rel_l2': np.sqrt(np.sum(error**2)) / np.sqrt(np.sum(reference**2))  # Relative L2 error
    }

def run_grid_comparison(n_problems: int = 10):
    """
    Compare solution times between different grid sizes:
    1. 20x20 grid (full solution)
    2. 40x40 grid (full solution)
    
    Args:
        n_problems (int): Number of test problems to solve
    """
    # Initialize experiments
    exp_20 = Experiment(nx=20, ny=20, n_subdomains=(1, 1))  # Single domain 20x20
    exp_40 = Experiment(nx=40, ny=40, n_subdomains=(1, 1))  # Single domain 40x40
    
    # Initialize timing arrays
    times_20x20 = []
    times_40x40 = []
    
    # Initialize error arrays
    errors_20x20 = []  # List of error metric dictionaries for 20x20
    error_fields_20x20 = []  # Full error fields for visualization
    
    print(f"\nRunning grid size comparison over {n_problems} problems...")
    
    for i in tqdm(range(n_problems), desc="Solving problems"):
        # Generate random problem for 20x20
        theta_20, f_20 = exp_20.data_generator.generate_random_problem()
        bc_dict_20 = exp_20.data_generator.generate_boundary_conditions()
        
        # Generate random problem for 40x40
        theta_40, f_40 = exp_40.data_generator.generate_random_problem()
        bc_dict_40 = exp_40.data_generator.generate_boundary_conditions()
        
        # Get reference solution (40x40)
        u_ref, _ = exp_40.full_solver.solve_full_domain(
            theta_40, f_40,
            bc_dict_40['left'], bc_dict_40['right'],
            bc_dict_40['bottom'], bc_dict_40['top']
        )
        
        # Time and solve 20x20 solution
        start = time.time()
        u_20, _ = exp_20.full_solver.solve_full_domain(
            theta_20, f_20,
            bc_dict_20['left'], bc_dict_20['right'],
            bc_dict_20['bottom'], bc_dict_20['top']
        )
        times_20x20.append(time.time() - start)
        
        # Interpolate 20x20 solution to 40x40 grid for comparison
        u_20_interp = np.zeros_like(u_ref)
        for ii in range(20):
            for jj in range(20):
                u_20_interp[2*ii:2*(ii+1), 2*jj:2*(jj+1)] = u_20[ii, jj]
        
        # Calculate comprehensive error metrics for 20x20
        errors_20x20.append(calculate_error_metrics(u_20_interp, u_ref))
        error_fields_20x20.append(np.abs(u_20_interp - u_ref))
        
        # Time 40x40 solution
        start = time.time()
        exp_40.full_solver.solve_full_domain(
            theta_40, f_40,
            bc_dict_40['left'], bc_dict_40['right'],
            bc_dict_40['bottom'], bc_dict_40['top']
        )
        times_40x40.append(time.time() - start)
    
    # Calculate statistics including error metrics
    stats = {
        '20x20': {
            'mean': np.mean(times_20x20),
            'std': np.std(times_20x20),
            'min': np.min(times_20x20),
            'max': np.max(times_20x20),
            'errors': {
                'l1_mean': np.mean([e['l1'] for e in errors_20x20]),
                'l1_std': np.std([e['l1'] for e in errors_20x20]),
                'l2_mean': np.mean([e['l2'] for e in errors_20x20]),
                'l2_std': np.std([e['l2'] for e in errors_20x20]),
                'linf_mean': np.mean([e['linf'] for e in errors_20x20]),
                'linf_std': np.std([e['linf'] for e in errors_20x20]),
                'rel_l2_mean': np.mean([e['rel_l2'] for e in errors_20x20]),
                'rel_l2_std': np.std([e['rel_l2'] for e in errors_20x20])
            }
        },
        '40x40': {
            'mean': np.mean(times_40x40),
            'std': np.std(times_40x40),
            'min': np.min(times_40x40),
            'max': np.max(times_40x40)
        }
    }
    
    # Print results with detailed error metrics
    print("\nGrid Size Comparison Results:")
    for grid, data in stats.items():
        print(f"\n{grid}:")
        print(f"  Mean time: {data['mean']:.6f} seconds")
        print(f"  Std dev:   {data['std']:.6f} seconds")
        print(f"  Min time:  {data['min']:.6f} seconds")
        print(f"  Max time:  {data['max']:.6f} seconds")
        if 'errors' in data:
            print(f"  Error Metrics:")
            print(f"    L1 (mean abs):     {data['errors']['l1_mean']:.6f} ± {data['errors']['l1_std']:.6f}")
            print(f"    L2 (root mean sq): {data['errors']['l2_mean']:.6f} ± {data['errors']['l2_std']:.6f}")
            print(f"    L∞ (max abs):      {data['errors']['linf_mean']:.6f} ± {data['errors']['linf_std']:.6f}")
            print(f"    Relative L2:       {data['errors']['rel_l2_mean']:.6f} ± {data['errors']['rel_l2_std']:.6f}")
    
    return stats

if __name__ == "__main__":
    # Run grid comparison
    stats = run_grid_comparison(n_problems=10) 