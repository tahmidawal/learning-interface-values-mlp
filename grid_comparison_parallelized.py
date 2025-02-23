import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm
import time
import os
import jax
import jax.numpy as jnp
import random

from pde_solver_jax import PoissonSolverJax, solve_multiple_subdomains
from data_generator import DataGenerator
from ml_model import InterfacePredictor
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

def combine_solutions(solutions: jnp.ndarray, positions: List[Tuple[int, int]], shape: Tuple[int, int]) -> np.ndarray:
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
    for idx, (y_start, x_start) in enumerate(positions):
        ny, nx = solutions[idx].shape
        full_solution[y_start:y_start+ny, x_start:x_start+nx] = np.array(solutions[idx])
    return full_solution

def plot_solution_comparison(u_20: np.ndarray, u_40: np.ndarray, u_ml: np.ndarray, 
                           sample_idx: int, save_dir: str = 'plots'):
    """Plot comparison of solutions from different methods."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f'Solution Comparison - Sample {sample_idx}', fontsize=16)
    
    # Plot 20x20 solution
    im1 = axes[0, 0].imshow(u_20)
    axes[0, 0].set_title('20x20 Solution')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 40x40
    im2 = axes[0, 1].imshow(u_40)
    axes[0, 1].set_title('40x40 Solution')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot ML solution
    im3 = axes[1, 0].imshow(u_ml)
    axes[1, 0].set_title('ML Solution (4 Subdomains)')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Plot error comparison
    error_ml = np.abs(u_ml - u_40)
    im4_1 = axes[1, 1].imshow(error_ml)
    axes[1, 1].set_title('ML Solution Error vs 40x40')
    plt.colorbar(im4_1, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sample_comparison_{sample_idx}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def run_grid_comparison(n_problems: int = 10):
    """
    Compare solution times between different grid sizes with JAX-based parallel solving.
    
    Args:
        n_problems (int): Number of test problems to solve
    """
    # Initialize experiments with JAX solvers
    exp_20 = Experiment(nx=20, ny=20, n_subdomains=(1, 1))
    exp_40 = Experiment(nx=40, ny=40, n_subdomains=(1, 1))
    exp_ml = Experiment(nx=40, ny=40, n_subdomains=(2, 2))
    
    # Replace solvers with JAX versions
    solver_20 = PoissonSolverJax(20, 20)
    solver_40 = PoissonSolverJax(40, 40)
    solver_ml = PoissonSolverJax(20, 20)  # For subdomains
    
    # Initialize timing and error arrays
    times_20x20 = []
    times_40x40 = []
    times_ml_inference = []
    times_ml_subdomains = []
    times_ml_total = []
    errors_ml = []
    error_fields_ml = []
    
    # Store solutions for visualization
    solutions_20x20 = []
    solutions_40x40 = []
    solutions_ml = []
    
    print(f"\nRunning grid size comparison over {n_problems} problems...")
    
    for i in tqdm(range(n_problems), desc="Solving problems"):
        # Generate random problems for each grid size independently
        theta_20, f_20 = exp_20.data_generator.generate_random_problem()
        bc_dict_20 = exp_20.data_generator.generate_boundary_conditions()
        
        theta_40, f_40 = exp_40.data_generator.generate_random_problem()
        bc_dict_40 = exp_40.data_generator.generate_boundary_conditions()
        
        # Convert inputs to JAX arrays
        theta_20_jax = jnp.array(theta_20)
        f_20_jax = jnp.array(f_20)
        theta_40_jax = jnp.array(theta_40)
        f_40_jax = jnp.array(f_40)
        
        # Prepare boundary conditions as arrays
        bc_left_20 = jnp.array([bc_dict_20['left'](y) for y in range(20)])
        bc_right_20 = jnp.array([bc_dict_20['right'](y) for y in range(20)])
        bc_bottom_20 = jnp.array([bc_dict_20['bottom'](x) for x in range(20)])
        bc_top_20 = jnp.array([bc_dict_20['top'](x) for x in range(20)])
        
        bc_left_40 = jnp.array([bc_dict_40['left'](y) for y in range(40)])
        bc_right_40 = jnp.array([bc_dict_40['right'](y) for y in range(40)])
        bc_bottom_40 = jnp.array([bc_dict_40['bottom'](x) for x in range(40)])
        bc_top_40 = jnp.array([bc_dict_40['top'](x) for x in range(40)])
        
        # Solve 20x20 grid
        start = time.time()
        u_20, _ = solver_20.solve_full_domain(
            theta_20_jax, f_20_jax,
            bc_left_20, bc_right_20,
            bc_bottom_20, bc_top_20
        )
        times_20x20.append(time.time() - start)
        solutions_20x20.append(np.array(u_20))
        
        # Solve 40x40 grid
        start = time.time()
        u_40, _ = solver_40.solve_full_domain(
            theta_40_jax, f_40_jax,
            bc_left_40, bc_right_40,
            bc_bottom_40, bc_top_40
        )
        times_40x40.append(time.time() - start)
        u_40 = np.array(u_40)
        solutions_40x40.append(u_40)
        
        # Time ML solution with breakdown
        start_ml = time.time()
        interface_data = exp_ml.data_generator.extract_interface_data(np.zeros((40, 40)))
        interface_predictions = {}
        for data in interface_data:
            pred_values = exp_ml.predictor.predict(data)
            key = (data['position'], data['type'])
            interface_predictions[key] = pred_values
        ml_inference_time = time.time() - start_ml
        times_ml_inference.append(ml_inference_time)
        
        # Prepare subdomain problems for parallel solving
        start_subdomains = time.time()
        thetas = []
        fs = []
        positions = []
        bc_arrays = {
            'left': [], 'right': [], 'bottom': [], 'top': []
        }
        
        for i in range(2):
            for j in range(2):
                x_start = j * 20
                x_end = (j + 1) * 20
                y_start = i * 20
                y_end = (i + 1) * 20
                
                thetas.append(theta_40[y_start:y_end, x_start:x_end])
                fs.append(f_40[y_start:y_end, x_start:x_end])
                positions.append((y_start, x_start))
                
                # Prepare boundary conditions
                bc_left = []
                if j == 0:
                    bc_left = [bc_dict_40['left'](y + y_start) for y in range(20)]
                else:
                    key = ((x_start, y_start), 'vertical')
                    values = interface_predictions[key]
                    bc_left = [float(values[int(y)]) for y in range(20)]
                bc_arrays['left'].append(bc_left)
                
                bc_right = []
                if j == 1:
                    bc_right = [bc_dict_40['right'](y + y_start) for y in range(20)]
                else:
                    key = ((x_end, y_start), 'vertical')
                    values = interface_predictions[key]
                    bc_right = [float(values[int(y)]) for y in range(20)]
                bc_arrays['right'].append(bc_right)
                
                bc_bottom = []
                if i == 0:
                    bc_bottom = [bc_dict_40['bottom'](x + x_start) for x in range(20)]
                else:
                    key = ((x_start, y_start), 'horizontal')
                    values = interface_predictions[key]
                    bc_bottom = [float(values[int(x)]) for x in range(20)]
                bc_arrays['bottom'].append(bc_bottom)
                
                bc_top = []
                if i == 1:
                    bc_top = [bc_dict_40['top'](x + x_start) for x in range(20)]
                else:
                    key = ((x_start, y_end), 'horizontal')
                    values = interface_predictions[key]
                    bc_top = [float(values[int(x)]) for x in range(20)]
                bc_arrays['top'].append(bc_top)
        
        # Convert to JAX arrays
        thetas_jax = jnp.stack([jnp.array(t) for t in thetas])
        fs_jax = jnp.stack([jnp.array(f) for f in fs])
        bc_dicts_jax = {
            k: jnp.stack([jnp.array(bc) for bc in v])
            for k, v in bc_arrays.items()
        }
        
        # Solve all subdomains in parallel
        solutions, _ = solve_multiple_subdomains(
            solver_ml, thetas_jax, fs_jax, bc_dicts_jax
        )
        
        # Combine solutions
        u_ml = combine_solutions(solutions, positions, (40, 40))
        solutions_ml.append(u_ml)
        subdomain_time = time.time() - start_subdomains
        times_ml_subdomains.append(subdomain_time)
        
        total_ml_time = ml_inference_time + subdomain_time
        times_ml_total.append(total_ml_time)
        
        # Only compute errors for ML solution vs 40x40
        errors_ml.append(calculate_error_metrics(u_ml, u_40))
        error_fields_ml.append(np.abs(u_ml - u_40))
    
    # Plot 5 random samples
    print("\nPlotting 5 random solution comparisons...")
    os.makedirs('plots', exist_ok=True)
    sample_indices = random.sample(range(n_problems), min(5, n_problems))
    for idx in sample_indices:
        plot_solution_comparison(
            solutions_20x20[idx],
            solutions_40x40[idx],
            solutions_ml[idx],
            idx
        )
    
    # Calculate statistics including error metrics
    stats = {
        '20x20': {
            'mean': np.mean(times_20x20),
            'std': np.std(times_20x20),
            'min': np.min(times_20x20),
            'max': np.max(times_20x20)
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
    
    plt.bar(error_metrics, ml_errors, yerr=ml_stds, capsize=5)
    plt.ylabel('Error')
    plt.title('ML Solution Error Metrics')
    
    # Plot 6: Example error field (from last problem)
    plt.subplot(3, 2, 6)
    plt.imshow(error_fields_ml[-1])
    plt.colorbar(label='Absolute Error')
    plt.title('ML Solution Error Field (Last Problem)')
    
    plt.tight_layout()
    plt.savefig('plots/grid_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return stats

if __name__ == "__main__":
    # Enable JAX 64-bit mode for better precision
    jax.config.update("jax_enable_x64", True)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Make sure the model is trained first
    exp = Experiment()
    exp.train_model(n_samples=1000, n_epochs=100, force_retrain=False)
    
    # Run grid comparison
    stats = run_grid_comparison(n_problems=10) 