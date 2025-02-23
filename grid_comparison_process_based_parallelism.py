import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm
import time
import os
from mpi4py import MPI

from pde_solver import PoissonSolver
from data_generator import DataGenerator
from ml_model import InterfacePredictor
from experiments import Experiment
from pde_solver_ml_mpi import solve_pde_ml_parallel, create_boundary_array

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

def solve_subdomain(solver: PoissonSolver, theta: np.ndarray, f: np.ndarray, bc_dict: Dict) -> Tuple[np.ndarray, int]:
    """Helper function to solve a single subdomain for parallel execution."""
    return solver.solve_subdomain(theta, f, bc_dict)

def combine_solutions(solutions: List[Tuple[np.ndarray, int]], positions: List[Tuple[int, int]], shape: Tuple[int, int]) -> np.ndarray:
    """
    Combine subdomain solutions into a full domain solution.
    
    Args:
        solutions: List of (solution, iterations) tuples for each subdomain
        positions: List of (y_start, x_start) positions for each subdomain
        shape: Shape of the full domain solution (ny, nx)
    
    Returns:
        Combined solution array
    """
    full_solution = np.zeros(shape)
    for (sol, _), (y_start, x_start) in zip(solutions, positions):
        ny, nx = sol.shape
        full_solution[y_start:y_start+ny, x_start:x_start+nx] = sol
    return full_solution

def run_grid_comparison(n_problems: int = 10):
    """
    Compare solution times between different grid sizes with detailed timing breakdown:
    1. 20x20 grid (full solution)
    2. 40x40 grid (full solution)
    3. 40x40 grid with ML-predicted interfaces using MPI parallelization
    
    Args:
        n_problems (int): Number of test problems to solve
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < 5:
        if rank == 0:
            print(f"Error: Need at least 5 MPI processes (1 master + 4 workers), but got {size}")
        comm.Abort(1)
        return None
    
    # Initialize experiments
    if rank == 0:
        exp_20 = Experiment(nx=20, ny=20, n_subdomains=(1, 1))  # Single domain 20x20
        exp_40 = Experiment(nx=40, ny=40, n_subdomains=(1, 1))  # Single domain 40x40
        exp_ml = Experiment(nx=40, ny=40, n_subdomains=(2, 2))  # 40x40 with ML interfaces
        
        # Initialize timing arrays
        times_20x20 = []
        times_40x40 = []
        times_ml_inference = []  # Time for ML to predict interface conditions
        times_ml_subdomains = []  # Time to solve subdomains in parallel
        times_ml_total = []  # Total time for ML solution
        
        # Initialize error arrays
        errors_20x20 = []  # List of error metric dictionaries for 20x20
        errors_ml = []     # List of error metric dictionaries for ML solution
        error_fields_20x20 = []  # Full error fields for visualization
        error_fields_ml = []     # Full error fields for visualization
        
        print(f"\nRunning grid size comparison over {n_problems} problems...")
    else:
        exp_ml = None
    
    # Broadcast exp_ml to all processes
    exp_ml = comm.bcast(exp_ml, root=0)
    
    for i in range(n_problems):
        if rank == 0:
            # Generate random problem for 20x20
            theta_20, f_20 = exp_20.data_generator.generate_random_problem()
            bc_dict_20 = exp_20.data_generator.generate_boundary_conditions()
            
            # Generate random problem for 40x40
            theta_40, f_40 = exp_40.data_generator.generate_random_problem()
            bc_dict_40 = exp_40.data_generator.generate_boundary_conditions()
            
            # Convert boundary conditions to arrays for MPI
            bc_arrays_40 = {
                'left': create_boundary_array(bc_dict_40['left'], 40),
                'right': create_boundary_array(bc_dict_40['right'], 40),
                'bottom': create_boundary_array(bc_dict_40['bottom'], 40),
                'top': create_boundary_array(bc_dict_40['top'], 40)
            }
            
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
            
            print(f"Problem {i+1}/{n_problems}")
        else:
            theta_40 = None
            f_40 = None
            bc_arrays_40 = None
            u_ref = None
        
        # Broadcast problem data to all processes
        theta_40 = comm.bcast(theta_40, root=0)
        f_40 = comm.bcast(f_40, root=0)
        bc_arrays_40 = comm.bcast(bc_arrays_40, root=0)
        
        # Solve using ML and MPI parallel computation
        solution, ml_time, solve_time = solve_pde_ml_parallel(theta_40, f_40, bc_arrays_40, exp_ml)
        
        if rank == 0:
            if solution is not None:
                # Record timing information
                times_ml_inference.append(ml_time)
                times_ml_subdomains.append(solve_time)
                times_ml_total.append(ml_time + solve_time)
                
                # Calculate error metrics for ML solution
                errors_ml.append(calculate_error_metrics(solution, u_ref))
                error_fields_ml.append(np.abs(solution - u_ref))
    
    # Only rank 0 processes results and creates plots
    if rank == 0:
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
            },
            'ML Inference': {
                'mean': np.mean(times_ml_inference),
                'std': np.std(times_ml_inference),
                'min': np.min(times_ml_inference),
                'max': np.max(times_ml_inference)
            },
            'ML Subdomains': {
                'mean': np.mean(times_ml_subdomains),
                'std': np.std(times_ml_subdomains),
                'min': np.min(times_ml_subdomains),
                'max': np.max(times_ml_subdomains)
            },
            'ML Total': {
                'mean': np.mean(times_ml_total),
                'std': np.std(times_ml_total),
                'min': np.min(times_ml_total),
                'max': np.max(times_ml_total),
                'errors': {
                    'l1_mean': np.mean([e['l1'] for e in errors_ml]),
                    'l1_std': np.std([e['l1'] for e in errors_ml]),
                    'l2_mean': np.mean([e['l2'] for e in errors_ml]),
                    'l2_std': np.std([e['l2'] for e in errors_ml]),
                    'linf_mean': np.mean([e['linf'] for e in errors_ml]),
                    'linf_std': np.std([e['linf'] for e in errors_ml]),
                    'rel_l2_mean': np.mean([e['rel_l2'] for e in errors_ml]),
                    'rel_l2_std': np.std([e['rel_l2'] for e in errors_ml])
                }
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
        
        # Calculate speedups
        speedup_vs_20 = stats['20x20']['mean'] / stats['ML Total']['mean']
        speedup_vs_40 = stats['40x40']['mean'] / stats['ML Total']['mean']
        print(f"\nSpeedup Factors:")
        print(f"ML vs 20x20: {speedup_vs_20:.2f}x")
        print(f"ML vs 40x40: {speedup_vs_40:.2f}x")
        
        # Create comparison plots
        plt.figure(figsize=(15, 15))
        
        # Plot 1: Time distributions
        plt.subplot(3, 2, 1)
        plt.hist(times_20x20, bins=10, alpha=0.5, label='20x20')
        plt.hist(times_40x40, bins=10, alpha=0.5, label='40x40')
        plt.hist(times_ml_total, bins=10, alpha=0.5, label='ML Total')
        plt.xlabel('Solution Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Solution Times')
        plt.legend()
        
        # Plot 2: Box plot comparison
        plt.subplot(3, 2, 2)
        plt.boxplot([times_20x20, times_40x40, times_ml_inference, times_ml_subdomains, times_ml_total],
                    labels=['20x20', '40x40', 'ML Inference', 'ML Subdomains', 'ML Total'])
        plt.xticks(rotation=45)
        plt.ylabel('Time (seconds)')
        plt.title('Solution Time Comparison')
        
        # Plot 3: Bar plot of mean times with error bars
        plt.subplot(3, 2, 3)
        means = [stats[k]['mean'] for k in ['20x20', '40x40', 'ML Inference', 'ML Subdomains', 'ML Total']]
        stds = [stats[k]['std'] for k in ['20x20', '40x40', 'ML Inference', 'ML Subdomains', 'ML Total']]
        plt.bar(['20x20', '40x40', 'ML Inference', 'ML Subdomains', 'ML Total'], 
                means, yerr=stds, capsize=5)
        plt.xticks(rotation=45)
        plt.ylabel('Mean Time (seconds)')
        plt.title('Average Solution Times')
        
        # Plot 4: ML time breakdown
        plt.subplot(3, 2, 4)
        ml_means = [stats['ML Inference']['mean'], stats['ML Subdomains']['mean']]
        ml_stds = [stats['ML Inference']['std'], stats['ML Subdomains']['std']]
        plt.bar(['ML Inference', 'ML Subdomains'], ml_means, yerr=ml_stds, capsize=5)
        plt.ylabel('Time (seconds)')
        plt.title('ML Solution Time Breakdown')
        
        # Plot 5: Error metrics comparison
        plt.subplot(3, 2, 5)
        error_metrics = ['L1', 'L2', 'L∞', 'Rel L2']
        ml_errors = [stats['ML Total']['errors'][f'{m}_mean'] for m in ['l1', 'l2', 'linf', 'rel_l2']]
        ml_stds = [stats['ML Total']['errors'][f'{m}_std'] for m in ['l1', 'l2', 'linf', 'rel_l2']]
        x20_errors = [stats['20x20']['errors'][f'{m}_mean'] for m in ['l1', 'l2', 'linf', 'rel_l2']]
        x20_stds = [stats['20x20']['errors'][f'{m}_std'] for m in ['l1', 'l2', 'linf', 'rel_l2']]
        
        x = np.arange(len(error_metrics))
        width = 0.35
        plt.bar(x - width/2, x20_errors, width, yerr=x20_stds, label='20x20', capsize=5)
        plt.bar(x + width/2, ml_errors, width, yerr=ml_stds, label='ML Solution', capsize=5)
        plt.xticks(x, error_metrics)
        plt.ylabel('Error')
        plt.title('Error Metrics Comparison')
        plt.legend()
        
        # Plot 6: Example error fields (from last problem)
        plt.subplot(3, 2, 6)
        plt.imshow(np.hstack([error_fields_20x20[-1], error_fields_ml[-1]]))
        plt.colorbar(label='Absolute Error')
        plt.axvline(x=40, color='white', linestyle='--')
        plt.xticks([20, 60], ['20x20', 'ML Solution'])
        plt.title('Example Error Fields (Last Problem)')
        
        plt.tight_layout()
        plt.savefig('plots/grid_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return stats
    
    return None

if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        # Make sure the model is trained first
        exp = Experiment()
        exp.train_model(n_samples=1000, n_epochs=100, force_retrain=False)
    
    # Run grid comparison
    stats = run_grid_comparison(n_problems=10)
    
    # Clean exit
    comm.Barrier()
    if rank == 0:
        print("\nDone!") 