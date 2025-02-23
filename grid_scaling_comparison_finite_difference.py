import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from tqdm import tqdm
import time
import os
import jax
import jax.numpy as jnp
import random

from pde_solver_jax import PoissonSolverJax
from data_generator import DataGenerator
from experiments import Experiment

def plot_scaling_comparison(times_dict: Dict[str, List[float]], save_dir: str = 'plots'):
    """Plot comparison of solution times for different grid sizes."""
    plt.figure(figsize=(12, 8))
    
    # Box plot
    plt.subplot(2, 1, 1)
    grid_sizes = ['20x20', '40x40', '80x80', '160x160', '320x320']
    times_list = [times_dict[size] for size in grid_sizes]
    plt.boxplot(times_list, labels=grid_sizes)
    plt.ylabel('Time (seconds)')
    plt.title('Solution Time Distribution by Grid Size')
    plt.yscale('log')  # Use log scale for better visualization
    
    # Bar plot with error bars
    plt.subplot(2, 1, 2)
    means = [np.mean(times_dict[size]) for size in grid_sizes]
    stds = [np.std(times_dict[size]) for size in grid_sizes]
    
    # Calculate theoretical scaling (n^2 complexity)
    base_time = means[0]  # 20x20 time
    base_size = 20
    sizes = [20, 40, 80, 160, 320]
    theoretical = [base_time * (size/base_size)**2 for size in sizes]
    
    x = np.arange(len(grid_sizes))
    width = 0.35
    plt.bar(x - width/2, means, width, yerr=stds, label='Measured', capsize=5)
    plt.bar(x + width/2, theoretical, width, alpha=0.5, label='Theoretical (n²)', capsize=5)
    plt.xticks(x, grid_sizes)
    plt.ylabel('Mean Time (seconds)')
    plt.title('Average Solution Time by Grid Size vs Theoretical Scaling')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'grid_scaling_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

def run_scaling_comparison(n_problems: int = 10):
    """
    Compare solution times between different grid sizes.
    
    Args:
        n_problems (int): Number of test problems to solve for each grid size
    """
    # Initialize solvers for different grid sizes
    solver_20 = PoissonSolverJax(20, 20)
    solver_40 = PoissonSolverJax(40, 40)
    solver_80 = PoissonSolverJax(80, 80)
    solver_160 = PoissonSolverJax(160, 160)
    solver_320 = PoissonSolverJax(320, 320)
    
    # Initialize experiments for different grid sizes
    exp_20 = Experiment(nx=20, ny=20, n_subdomains=(1, 1))
    exp_40 = Experiment(nx=40, ny=40, n_subdomains=(1, 1))
    exp_80 = Experiment(nx=80, ny=80, n_subdomains=(1, 1))
    exp_160 = Experiment(nx=160, ny=160, n_subdomains=(1, 1))
    exp_320 = Experiment(nx=320, ny=320, n_subdomains=(1, 1))
    
    # Initialize timing arrays
    times = {
        '20x20': [],
        '40x40': [],
        '80x80': [],
        '160x160': [],
        '320x320': []
    }
    
    print(f"\nRunning grid scaling comparison over {n_problems} problems...")
    
    for i in tqdm(range(n_problems), desc="Solving problems"):
        # 20x20 grid
        theta_20, f_20 = exp_20.data_generator.generate_random_problem()
        bc_dict_20 = exp_20.data_generator.generate_boundary_conditions()
        bc_left_20 = jnp.array([bc_dict_20['left'](y) for y in range(20)])
        bc_right_20 = jnp.array([bc_dict_20['right'](y) for y in range(20)])
        bc_bottom_20 = jnp.array([bc_dict_20['bottom'](x) for x in range(20)])
        bc_top_20 = jnp.array([bc_dict_20['top'](x) for x in range(20)])
        
        start = time.time()
        result_20 = solver_20.solve_full_domain(
            jnp.array(theta_20), jnp.array(f_20),
            bc_left_20, bc_right_20,
            bc_bottom_20, bc_top_20
        )
        for arr in result_20:
            if hasattr(arr, 'block_until_ready'):
                arr.block_until_ready()
        times['20x20'].append(time.time() - start)
        
        # 40x40 grid
        theta_40, f_40 = exp_40.data_generator.generate_random_problem()
        bc_dict_40 = exp_40.data_generator.generate_boundary_conditions()
        bc_left_40 = jnp.array([bc_dict_40['left'](y) for y in range(40)])
        bc_right_40 = jnp.array([bc_dict_40['right'](y) for y in range(40)])
        bc_bottom_40 = jnp.array([bc_dict_40['bottom'](x) for x in range(40)])
        bc_top_40 = jnp.array([bc_dict_40['top'](x) for x in range(40)])
        
        start = time.time()
        result_40 = solver_40.solve_full_domain(
            jnp.array(theta_40), jnp.array(f_40),
            bc_left_40, bc_right_40,
            bc_bottom_40, bc_top_40
        )
        for arr in result_40:
            if hasattr(arr, 'block_until_ready'):
                arr.block_until_ready()
        times['40x40'].append(time.time() - start)
        
        # 80x80 grid
        theta_80, f_80 = exp_80.data_generator.generate_random_problem()
        bc_dict_80 = exp_80.data_generator.generate_boundary_conditions()
        bc_left_80 = jnp.array([bc_dict_80['left'](y) for y in range(80)])
        bc_right_80 = jnp.array([bc_dict_80['right'](y) for y in range(80)])
        bc_bottom_80 = jnp.array([bc_dict_80['bottom'](x) for x in range(80)])
        bc_top_80 = jnp.array([bc_dict_80['top'](x) for x in range(80)])
        
        start = time.time()
        result_80 = solver_80.solve_full_domain(
            jnp.array(theta_80), jnp.array(f_80),
            bc_left_80, bc_right_80,
            bc_bottom_80, bc_top_80
        )
        for arr in result_80:
            if hasattr(arr, 'block_until_ready'):
                arr.block_until_ready()
        times['80x80'].append(time.time() - start)
        
        # 160x160 grid
        theta_160, f_160 = exp_160.data_generator.generate_random_problem()
        bc_dict_160 = exp_160.data_generator.generate_boundary_conditions()
        bc_left_160 = jnp.array([bc_dict_160['left'](y) for y in range(160)])
        bc_right_160 = jnp.array([bc_dict_160['right'](y) for y in range(160)])
        bc_bottom_160 = jnp.array([bc_dict_160['bottom'](x) for x in range(160)])
        bc_top_160 = jnp.array([bc_dict_160['top'](x) for x in range(160)])
        
        start = time.time()
        result_160 = solver_160.solve_full_domain(
            jnp.array(theta_160), jnp.array(f_160),
            bc_left_160, bc_right_160,
            bc_bottom_160, bc_top_160
        )
        for arr in result_160:
            if hasattr(arr, 'block_until_ready'):
                arr.block_until_ready()
        times['160x160'].append(time.time() - start)
        
        # 320x320 grid
        theta_320, f_320 = exp_320.data_generator.generate_random_problem()
        bc_dict_320 = exp_320.data_generator.generate_boundary_conditions()
        bc_left_320 = jnp.array([bc_dict_320['left'](y) for y in range(320)])
        bc_right_320 = jnp.array([bc_dict_320['right'](y) for y in range(320)])
        bc_bottom_320 = jnp.array([bc_dict_320['bottom'](x) for x in range(320)])
        bc_top_320 = jnp.array([bc_dict_320['top'](x) for x in range(320)])
        
        start = time.time()
        result_320 = solver_320.solve_full_domain(
            jnp.array(theta_320), jnp.array(f_320),
            bc_left_320, bc_right_320,
            bc_bottom_320, bc_top_320
        )
        for arr in result_320:
            if hasattr(arr, 'block_until_ready'):
                arr.block_until_ready()
        times['320x320'].append(time.time() - start)
    
    # Calculate and print statistics
    print("\nGrid Scaling Comparison Results:")
    for size, size_times in times.items():
        print(f"\n{size}:")
        print(f"  Mean time: {np.mean(size_times):.6f} seconds")
        print(f"  Std dev:   {np.std(size_times):.6f} seconds")
        print(f"  Min time:  {np.min(size_times):.6f} seconds")
        print(f"  Max time:  {np.max(size_times):.6f} seconds")
    
    # Calculate scaling factors
    base_time = np.mean(times['20x20'])
    for size in ['40x40', '80x80', '160x160', '320x320']:
        mean_time = np.mean(times[size])
        scaling = mean_time / base_time
        theoretical = (int(size.split('x')[0]) / 20) ** 2
        print(f"\nScaling factor for {size} vs 20x20:")
        print(f"  Measured:    {scaling:.2f}x")
        print(f"  Theoretical: {theoretical:.2f}x (n²)")
        print(f"  Efficiency:  {theoretical/scaling*100:.1f}%")
    
    # Plot results
    os.makedirs('plots', exist_ok=True)
    plot_scaling_comparison(times)
    
    return times

if __name__ == "__main__":
    # Enable JAX 64-bit mode for better precision
    jax.config.update("jax_enable_x64", True)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run scaling comparison
    times = run_scaling_comparison(n_problems=10) 