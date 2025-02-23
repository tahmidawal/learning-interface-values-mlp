import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from tqdm import tqdm
import time
import os
import jax
import jax.numpy as jnp
import random
import fenics as fe

from pde_solver_jax import PoissonSolverJax
from data_generator import DataGenerator
from experiments import Experiment

def solve_poisson_fem(mesh, f, theta, bc_dict):
    """
    Solve Poisson equation using FEM with FEniCS.
    
    Args:
        mesh: FEniCS mesh
        f: Source term function
        theta: Diffusion coefficient function
        bc_dict: Dictionary containing boundary condition functions
    """
    # Define function space
    V = fe.FunctionSpace(mesh, 'P', 1)
    
    # Define boundary conditions
    def boundary_left(x, on_boundary):
        return on_boundary and fe.near(x[0], 0)
    
    def boundary_right(x, on_boundary):
        return on_boundary and fe.near(x[0], 1)
    
    def boundary_bottom(x, on_boundary):
        return on_boundary and fe.near(x[1], 0)
    
    def boundary_top(x, on_boundary):
        return on_boundary and fe.near(x[1], 1)
    
    # Create FEniCS expressions for boundary conditions
    class BCExpression(fe.Expression):
        def __init__(self, bc_func, **kwargs):
            super().__init__(**kwargs)
            self.bc_func = bc_func
        
        def eval(self, values, x):
            if fe.near(x[0], 0) or fe.near(x[0], 1):
                values[0] = float(self.bc_func(x[1]))
            else:
                values[0] = float(self.bc_func(x[0]))
        
        def value_shape(self):
            return ()
    
    # Apply Dirichlet boundary conditions with degree matching the function space
    bc_left = fe.DirichletBC(V, BCExpression(bc_dict['left'], degree=1), boundary_left)
    bc_right = fe.DirichletBC(V, BCExpression(bc_dict['right'], degree=1), boundary_right)
    bc_bottom = fe.DirichletBC(V, BCExpression(bc_dict['bottom'], degree=1), boundary_bottom)
    bc_top = fe.DirichletBC(V, BCExpression(bc_dict['top'], degree=1), boundary_top)
    bcs = [bc_left, bc_right, bc_bottom, bc_top]
    
    # Define variational problem
    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)
    
    # Convert theta to FEniCS Function
    theta_fe = fe.Function(V)
    theta_fe.vector()[:] = theta.flatten()
    
    # Convert f to FEniCS Function
    f_fe = fe.Function(V)
    f_fe.vector()[:] = f.flatten()
    
    a = theta_fe * fe.dot(fe.grad(u), fe.grad(v)) * fe.dx
    L = f_fe * v * fe.dx
    
    # Solve the problem
    u = fe.Function(V)
    fe.solve(a == L, u, bcs)
    
    return u

def plot_scaling_comparison(times_dict: Dict[str, List[float]], save_dir: str = 'plots'):
    """Plot comparison of solution times for different grid sizes."""
    plt.figure(figsize=(12, 8))
    
    # Box plot
    plt.subplot(2, 1, 1)
    grid_sizes = ['20x20', '40x40', '80x80', '160x160', '320x320']
    times_list = [times_dict[size] for size in grid_sizes]
    plt.boxplot(times_list, labels=grid_sizes)
    plt.ylabel('Time (seconds)')
    plt.title('FEM Solution Time Distribution by Grid Size')
    plt.yscale('log')
    
    # Bar plot with error bars
    plt.subplot(2, 1, 2)
    means = [np.mean(times_dict[size]) for size in grid_sizes]
    stds = [np.std(times_dict[size]) for size in grid_sizes]
    
    # Calculate theoretical scaling (n^2 complexity for 2D FEM)
    base_time = means[0]
    base_size = 20
    sizes = [20, 40, 80, 160, 320]
    theoretical = [base_time * (size/base_size)**2 for size in sizes]
    
    x = np.arange(len(grid_sizes))
    width = 0.35
    plt.bar(x - width/2, means, width, yerr=stds, label='Measured', capsize=5)
    plt.bar(x + width/2, theoretical, width, alpha=0.5, label='Theoretical (n²)', capsize=5)
    plt.xticks(x, grid_sizes)
    plt.ylabel('Mean Time (seconds)')
    plt.title('Average FEM Solution Time by Grid Size vs Theoretical Scaling')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'grid_scaling_comparison_fem.png'), dpi=150, bbox_inches='tight')
    plt.close()

def run_scaling_comparison(n_problems: int = 10):
    """
    Compare solution times between different grid sizes using FEM.
    
    Args:
        n_problems (int): Number of test problems to solve for each grid size
    """
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
    
    # Create meshes for different grid sizes
    mesh_20 = fe.RectangleMesh(fe.Point(0, 0), fe.Point(1, 1), 19, 19)
    mesh_40 = fe.RectangleMesh(fe.Point(0, 0), fe.Point(1, 1), 39, 39)
    mesh_80 = fe.RectangleMesh(fe.Point(0, 0), fe.Point(1, 1), 79, 79)
    mesh_160 = fe.RectangleMesh(fe.Point(0, 0), fe.Point(1, 1), 159, 159)
    mesh_320 = fe.RectangleMesh(fe.Point(0, 0), fe.Point(1, 1), 319, 319)
    
    print(f"\nRunning FEM grid scaling comparison over {n_problems} problems...")
    
    for i in tqdm(range(n_problems), desc="Solving problems"):
        # 20x20 grid
        theta_20, f_20 = exp_20.data_generator.generate_random_problem()
        bc_dict_20 = exp_20.data_generator.generate_boundary_conditions()
        
        start = time.time()
        _ = solve_poisson_fem(mesh_20, f_20, theta_20, bc_dict_20)
        times['20x20'].append(time.time() - start)
        
        # 40x40 grid
        theta_40, f_40 = exp_40.data_generator.generate_random_problem()
        bc_dict_40 = exp_40.data_generator.generate_boundary_conditions()
        
        start = time.time()
        _ = solve_poisson_fem(mesh_40, f_40, theta_40, bc_dict_40)
        times['40x40'].append(time.time() - start)
        
        # 80x80 grid
        theta_80, f_80 = exp_80.data_generator.generate_random_problem()
        bc_dict_80 = exp_80.data_generator.generate_boundary_conditions()
        
        start = time.time()
        _ = solve_poisson_fem(mesh_80, f_80, theta_80, bc_dict_80)
        times['80x80'].append(time.time() - start)
        
        # 160x160 grid
        theta_160, f_160 = exp_160.data_generator.generate_random_problem()
        bc_dict_160 = exp_160.data_generator.generate_boundary_conditions()
        
        start = time.time()
        _ = solve_poisson_fem(mesh_160, f_160, theta_160, bc_dict_160)
        times['160x160'].append(time.time() - start)
        
        # 320x320 grid
        theta_320, f_320 = exp_320.data_generator.generate_random_problem()
        bc_dict_320 = exp_320.data_generator.generate_boundary_conditions()
        
        start = time.time()
        _ = solve_poisson_fem(mesh_320, f_320, theta_320, bc_dict_320)
        times['320x320'].append(time.time() - start)
    
    # Calculate and print statistics
    print("\nFEM Grid Scaling Comparison Results:")
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
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Run scaling comparison
    times = run_scaling_comparison(n_problems=10)



