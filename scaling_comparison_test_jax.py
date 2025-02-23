"""
This script is used to test the scaling of the JAX implementation of the Poisson solver for different grid sizes.
"""
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
from experiments import Experiment

def generate_theta_f(nx: int, ny: int, k1: float, k2: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate theta and f for Poisson equation with f = sin(k1*2πx)*sin(k2*2πy)."""
    x = jnp.linspace(0, 1, nx)
    y = jnp.linspace(0, 1, ny)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    
    f = jnp.sin(k1 * 2 * jnp.pi * X) * jnp.sin(k2 * 2 * jnp.pi * Y)
    theta = jnp.zeros_like(f)  # Initial guess for solution (can be zero)
    
    return theta, f

def generate_boundary_conditions(nx: int, ny: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate zero Dirichlet boundary conditions for the domain."""
    bc_left = jnp.zeros(ny)
    bc_right = jnp.zeros(ny)
    bc_bottom = jnp.zeros(nx)
    bc_top = jnp.zeros(nx)
    return bc_left, bc_right, bc_bottom, bc_top

def plot_scaling_comparison(times_dict: Dict[str, List[float]], save_dir: str = 'plots'):
    """Plot comparison of solution times for different grid sizes."""
    plt.figure(figsize=(12, 8))
    
    grid_sizes = list(times_dict.keys())
    times_list = [times_dict[size] for size in grid_sizes]

    # Box plot of times
    plt.subplot(2, 1, 1)
    plt.boxplot(times_list, labels=grid_sizes)
    plt.ylabel('Time (seconds)')
    plt.title('Solution Time Distribution by Grid Size')
    plt.yscale('log')  # Use log scale for clarity
    
    # Bar plot with measured vs. theoretical times
    plt.subplot(2, 1, 2)
    means = [np.mean(times_dict[size]) for size in grid_sizes]
    stds = [np.std(times_dict[size]) for size in grid_sizes]

    base_time = means[0]  # Use 20x20 time as baseline
    base_size = int(grid_sizes[0].split('x')[0])
    sizes = [int(size.split('x')[0]) for size in grid_sizes]
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
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'grid_scaling_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

def run_scaling_comparison(n_problems: int = 10, k1: float = 2, k2: float = 2):
    """
    Compare solution times between different grid sizes using the Poisson equation:
    f(x, y) = sin(k1*2πx)*sin(k2*2πy).
    
    Args:
        n_problems (int): Number of test problems to solve for each grid size
        k1, k2 (float): Wave numbers for the sinusoidal forcing term
    """
    grid_sizes = [20, 40, 80, 160, 320]
    
    # Initialize solvers and times
    solvers = {f"{size}x{size}": PoissonSolverJax(size, size) for size in grid_sizes}
    times = {f"{size}x{size}": [] for size in grid_sizes}

    print(f"\nRunning Poisson equation scaling comparison with f=sin({k1}*2πx)*sin({k2}*2πy) over {n_problems} problems...")

    for i in tqdm(range(n_problems), desc="Solving problems"):
        for size in grid_sizes:
            theta, f = generate_theta_f(size, size, k1, k2)
            bc_left, bc_right, bc_bottom, bc_top = generate_boundary_conditions(size, size)

            start = time.time()
            solvers[f"{size}x{size}"].solve_full_domain(
                theta, f,
                bc_left, bc_right,
                bc_bottom, bc_top
            )
            times[f"{size}x{size}"].append(time.time() - start)
    
    # Print results
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
        theoretical = (int(size.split('x')[0]) / 20) ** 2
        scaling = mean_time / base_time
        efficiency = (theoretical / scaling) * 100
        print(f"\nScaling factor for {size} vs 20x20:")
        print(f"  Measured:    {scaling:.2f}x")
        print(f"  Theoretical: {theoretical:.2f}x (n²)")
        print(f"  Efficiency:  {efficiency:.1f}%")
    
    # Plot results
    plot_scaling_comparison(times)
    
    return times

if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    random.seed(42)
    np.random.seed(42)
    
    times = run_scaling_comparison(n_problems=10, k1=2, k2=2)
