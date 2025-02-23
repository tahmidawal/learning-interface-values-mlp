import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import time
from mpi4py import MPI
from tqdm import tqdm

from pde_solver import PoissonSolver
from experiments import Experiment
from pde_solver_ml_mpi import solve_pde_ml_parallel, create_boundary_array, BoundaryCondition

def calculate_error_metrics(solution: np.ndarray, reference: np.ndarray) -> Dict[str, float]:
    """Calculate various error metrics between solution and reference."""
    error = np.abs(solution - reference)
    return {
        'l1': np.mean(error),           # Mean absolute error
        'l2': np.sqrt(np.mean(error**2)), # Root mean square error
        'linf': np.max(error),          # Maximum absolute error
        'rel_l2': np.sqrt(np.sum(error**2)) / np.sqrt(np.sum(reference**2))  # Relative L2 error
    }

def solve_subdomain(solver: PoissonSolver, theta: np.ndarray, f: np.ndarray, bc_dict: Dict) -> Tuple[np.ndarray, int]:
    """Helper function to solve a single subdomain."""
    return solver.solve_subdomain(theta, f, bc_dict)

def run_comparison(n_problems: int = 10):
    """
    Compare solution times and errors between methods, running each method separately:
    1. First solve all 20x20 problems
    2. Then solve all 40x40 problems
    3. Finally solve all 40x40 ML-MPI problems
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < 5:
        if rank == 0:
            print(f"Error: Need at least 5 MPI processes (1 master + 4 workers), but got {size}")
        return None
    
    if rank == 0:
        print(f"Initializing with {size} processes")
        
        # Initialize results storage
        results = {
            '20x20': {'times': [], 'errors': [], 'error_fields': []},
            '40x40': {'times': [], 'errors': [], 'error_fields': []},
            'ML-MPI': {'times': [], 'inference_times': [], 'solve_times': [], 'errors': [], 'error_fields': []}
        }
        
        # Generate all problems upfront
        print("\nGenerating test problems...")
        problems = []
        for _ in range(n_problems):
            exp = Experiment(nx=40, ny=40, n_subdomains=(1, 1))
            theta, f = exp.data_generator.generate_random_problem()
            bc_dict = exp.data_generator.generate_boundary_conditions()
            problems.append((theta, f, bc_dict))
        
        # Get reference solutions
        print("\nComputing reference solutions...")
        reference_solutions = []
        exp_40 = Experiment(nx=40, ny=40, n_subdomains=(1, 1))
        for theta, f, bc_dict in problems:
            u_ref, _ = exp_40.full_solver.solve_full_domain(
                theta, f,
                bc_dict['left'], bc_dict['right'],
                bc_dict['bottom'], bc_dict['top']
            )
            reference_solutions.append(u_ref)
        del exp_40  # Clear memory
        
        # 1. Solve 20x20 problems
        print("\nSolving 20x20 problems...")
        exp_20 = Experiment(nx=20, ny=20, n_subdomains=(1, 1))
        for i in tqdm(range(n_problems), desc="20x20 grid"):
            theta_40, f_40, bc_dict_40 = problems[i]
            u_ref = reference_solutions[i]
            
            # Downsample to 20x20
            theta_20 = theta_40[::2, ::2]
            f_20 = f_40[::2, ::2]
            bc_dict_20 = {
                'left': lambda y: bc_dict_40['left'](2*y),
                'right': lambda y: bc_dict_40['right'](2*y),
                'bottom': lambda x: bc_dict_40['bottom'](2*x),
                'top': lambda x: bc_dict_40['top'](2*x)
            }
            
            start = time.time()
            u_20, _ = exp_20.full_solver.solve_full_domain(
                theta_20, f_20,
                bc_dict_20['left'], bc_dict_20['right'],
                bc_dict_20['bottom'], bc_dict_20['top']
            )
            solve_time = time.time() - start
            results['20x20']['times'].append(solve_time)
            
            # Interpolate to 40x40 for comparison
            u_20_interp = np.zeros_like(u_ref)
            for ii in range(20):
                for jj in range(20):
                    u_20_interp[2*ii:2*(ii+1), 2*jj:2*(jj+1)] = u_20[ii, jj]
            
            results['20x20']['errors'].append(calculate_error_metrics(u_20_interp, u_ref))
            results['20x20']['error_fields'].append(np.abs(u_20_interp - u_ref))
            
            print(f"\n20x20 Problem {i+1} results:")
            print(f"Solution time: {solve_time:.6f} seconds")
            print(f"Mean absolute error: {results['20x20']['errors'][-1]['l1']:.6e}")
            print(f"Relative L2 error: {results['20x20']['errors'][-1]['rel_l2']:.6e}")
        
        del exp_20  # Clear memory
        
        # 2. Solve 40x40 problems
        print("\nSolving 40x40 problems...")
        exp_40 = Experiment(nx=40, ny=40, n_subdomains=(1, 1))
        for i in tqdm(range(n_problems), desc="40x40 grid"):
            theta, f, bc_dict = problems[i]
            u_ref = reference_solutions[i]
            
            start = time.time()
            u_40, _ = exp_40.full_solver.solve_full_domain(
                theta, f,
                bc_dict['left'], bc_dict['right'],
                bc_dict['bottom'], bc_dict['top']
            )
            solve_time = time.time() - start
            results['40x40']['times'].append(solve_time)
            
            results['40x40']['errors'].append(calculate_error_metrics(u_40, u_ref))
            results['40x40']['error_fields'].append(np.abs(u_40 - u_ref))
            
            print(f"\n40x40 Problem {i+1} results:")
            print(f"Solution time: {solve_time:.6f} seconds")
            print(f"Mean absolute error: {results['40x40']['errors'][-1]['l1']:.6e}")
            print(f"Relative L2 error: {results['40x40']['errors'][-1]['rel_l2']:.6e}")
        
        del exp_40  # Clear memory
        
        # 3. Solve ML-MPI problems
        print("\nSolving ML-MPI problems...")
        exp_ml = Experiment(nx=40, ny=40, n_subdomains=(2, 2))
        for i in tqdm(range(n_problems), desc="ML-MPI"):
            theta, f, bc_dict = problems[i]
            u_ref = reference_solutions[i]
            
            # Convert boundary conditions to arrays
            bc_arrays = {
                'left': create_boundary_array(bc_dict['left'], 40),
                'right': create_boundary_array(bc_dict['right'], 40),
                'bottom': create_boundary_array(bc_dict['bottom'], 40),
                'top': create_boundary_array(bc_dict['top'], 40)
            }
            
            # Solve using ML-MPI method
            u_ml, ml_time, solve_time = solve_pde_ml_parallel(theta, f, bc_arrays, exp_ml)
            
            results['ML-MPI']['inference_times'].append(ml_time)
            results['ML-MPI']['solve_times'].append(solve_time)
            results['ML-MPI']['times'].append(ml_time + solve_time)
            results['ML-MPI']['errors'].append(calculate_error_metrics(u_ml, u_ref))
            results['ML-MPI']['error_fields'].append(np.abs(u_ml - u_ref))
            
            print(f"\nML-MPI Problem {i+1} results:")
            print(f"ML prediction time: {ml_time:.6f} seconds")
            print(f"Parallel solve time: {solve_time:.6f} seconds")
            print(f"Total time: {ml_time + solve_time:.6f} seconds")
            print(f"Mean absolute error: {results['ML-MPI']['errors'][-1]['l1']:.6e}")
            print(f"Relative L2 error: {results['ML-MPI']['errors'][-1]['rel_l2']:.6e}")
        
        # Calculate summary statistics
        stats = {}
        for method in ['20x20', '40x40', 'ML-MPI']:
            stats[method] = {
                'mean_time': np.mean(results[method]['times']),
                'std_time': np.std(results[method]['times']),
                'min_time': np.min(results[method]['times']),
                'max_time': np.max(results[method]['times']),
                'errors': {
                    'l1_mean': np.mean([e['l1'] for e in results[method]['errors']]),
                    'l1_std': np.std([e['l1'] for e in results[method]['errors']]),
                    'l2_mean': np.mean([e['l2'] for e in results[method]['errors']]),
                    'l2_std': np.std([e['l2'] for e in results[method]['errors']]),
                    'linf_mean': np.mean([e['linf'] for e in results[method]['errors']]),
                    'linf_std': np.std([e['linf'] for e in results[method]['errors']]),
                    'rel_l2_mean': np.mean([e['rel_l2'] for e in results[method]['errors']]),
                    'rel_l2_std': np.std([e['rel_l2'] for e in results[method]['errors']])
                }
            }
        
        if 'ML-MPI' in stats:
            stats['ML-MPI'].update({
                'mean_inference_time': np.mean(results['ML-MPI']['inference_times']),
                'std_inference_time': np.std(results['ML-MPI']['inference_times']),
                'mean_solve_time': np.mean(results['ML-MPI']['solve_times']),
                'std_solve_time': np.std(results['ML-MPI']['solve_times'])
            })
        
        # Print summary statistics
        print("\nSummary Statistics:")
        for method in ['20x20', '40x40', 'ML-MPI']:
            print(f"\n{method}:")
            print(f"  Mean time: {stats[method]['mean_time']:.6f} ± {stats[method]['std_time']:.6f} seconds")
            print(f"  Min time: {stats[method]['min_time']:.6f} seconds")
            print(f"  Max time: {stats[method]['max_time']:.6f} seconds")
            print(f"  Error Metrics:")
            print(f"    L1 (mean abs):     {stats[method]['errors']['l1_mean']:.6f} ± {stats[method]['errors']['l1_std']:.6f}")
            print(f"    L2 (root mean sq): {stats[method]['errors']['l2_mean']:.6f} ± {stats[method]['errors']['l2_std']:.6f}")
            print(f"    L∞ (max abs):      {stats[method]['errors']['linf_mean']:.6f} ± {stats[method]['errors']['linf_std']:.6f}")
            print(f"    Relative L2:       {stats[method]['errors']['rel_l2_mean']:.6f} ± {stats[method]['errors']['rel_l2_std']:.6f}")
        
        if 'ML-MPI' in stats:
            print(f"\nML-MPI Breakdown:")
            print(f"  Mean inference time: {stats['ML-MPI']['mean_inference_time']:.6f} ± {stats['ML-MPI']['std_inference_time']:.6f} seconds")
            print(f"  Mean solve time: {stats['ML-MPI']['mean_solve_time']:.6f} ± {stats['ML-MPI']['std_solve_time']:.6f} seconds")
        
        # Calculate speedups
        speedup_vs_20 = stats['20x20']['mean_time'] / stats['ML-MPI']['mean_time']
        speedup_vs_40 = stats['40x40']['mean_time'] / stats['ML-MPI']['mean_time']
        print(f"\nSpeedup Factors:")
        print(f"ML-MPI vs 20x20: {speedup_vs_20:.2f}x")
        print(f"ML-MPI vs 40x40: {speedup_vs_40:.2f}x")
        
        # Create visualization plots
        plt.figure(figsize=(15, 15))
        
        # Plot 1: Time distributions
        plt.subplot(3, 2, 1)
        plt.hist(results['20x20']['times'], bins=10, alpha=0.5, label='20x20')
        plt.hist(results['40x40']['times'], bins=10, alpha=0.5, label='40x40')
        plt.hist(results['ML-MPI']['times'], bins=10, alpha=0.5, label='ML-MPI')
        plt.xlabel('Solution Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Solution Times')
        plt.legend()
        
        # Plot 2: Box plot comparison
        plt.subplot(3, 2, 2)
        plt.boxplot([
            results['20x20']['times'],
            results['40x40']['times'],
            results['ML-MPI']['inference_times'],
            results['ML-MPI']['solve_times'],
            results['ML-MPI']['times']
        ], labels=['20x20', '40x40', 'ML Inference', 'ML Solve', 'ML Total'])
        plt.xticks(rotation=45)
        plt.ylabel('Time (seconds)')
        plt.title('Solution Time Comparison')
        
        # Plot 3: Bar plot of mean times with error bars
        plt.subplot(3, 2, 3)
        methods = ['20x20', '40x40', 'ML-MPI']
        means = [stats[m]['mean_time'] for m in methods]
        stds = [stats[m]['std_time'] for m in methods]
        plt.bar(methods, means, yerr=stds, capsize=5)
        plt.xticks(rotation=45)
        plt.ylabel('Mean Time (seconds)')
        plt.title('Average Solution Times')
        
        # Plot 4: ML time breakdown
        plt.subplot(3, 2, 4)
        ml_means = [stats['ML-MPI']['mean_inference_time'], stats['ML-MPI']['mean_solve_time']]
        ml_stds = [stats['ML-MPI']['std_inference_time'], stats['ML-MPI']['std_solve_time']]
        plt.bar(['ML Inference', 'ML Solve'], ml_means, yerr=ml_stds, capsize=5)
        plt.ylabel('Time (seconds)')
        plt.title('ML-MPI Solution Time Breakdown')
        
        # Plot 5: Error metrics comparison
        plt.subplot(3, 2, 5)
        error_metrics = ['L1', 'L2', 'L∞', 'Rel L2']
        ml_errors = [stats['ML-MPI']['errors'][f'{m}_mean'] for m in ['l1', 'l2', 'linf', 'rel_l2']]
        ml_stds = [stats['ML-MPI']['errors'][f'{m}_std'] for m in ['l1', 'l2', 'linf', 'rel_l2']]
        trad_errors = [stats['40x40']['errors'][f'{m}_mean'] for m in ['l1', 'l2', 'linf', 'rel_l2']]
        trad_stds = [stats['40x40']['errors'][f'{m}_std'] for m in ['l1', 'l2', 'linf', 'rel_l2']]
        
        x = np.arange(len(error_metrics))
        width = 0.35
        plt.bar(x - width/2, trad_errors, width, yerr=trad_stds, label='Traditional', capsize=5)
        plt.bar(x + width/2, ml_errors, width, yerr=ml_stds, label='ML-MPI', capsize=5)
        plt.xticks(x, error_metrics)
        plt.ylabel('Error')
        plt.title('Error Metrics Comparison')
        plt.legend()
        
        # Plot 6: Example error fields
        plt.subplot(3, 2, 6)
        plt.imshow(np.hstack([results['20x20']['error_fields'][-1], results['ML-MPI']['error_fields'][-1]]))
        plt.colorbar(label='Absolute Error')
        plt.axvline(x=40, color='white', linestyle='--')
        plt.xticks([20, 60], ['Traditional', 'ML-MPI'])
        plt.title('Example Error Fields (Last Problem)')
        
        plt.tight_layout()
        plt.savefig('plots/traditional_vs_ml_mpi_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return stats
    
    else:
        # Worker processes for ML-MPI method
        exp = Experiment(nx=40, ny=40, n_subdomains=(2, 2))  # Initialize experiment for workers
        while True:
            status = MPI.Status()
            if comm.Iprobe(source=0, tag=MPI.ANY_TAG, status=status):
                tag = status.Get_tag()
                if tag == 999:  # Stop signal
                    break
                task, position = comm.recv(source=0, tag=tag)
                result = solve_subdomain(*task)
                comm.send(result, dest=0, tag=tag)
            else:
                time.sleep(0.1)  # Prevent busy waiting

if __name__ == "__main__":
    # Make sure the model is trained first
    exp = Experiment()
    exp.train_model(n_samples=1000, n_epochs=100, force_retrain=False)
    
    # Run comparison
    stats = run_comparison(n_problems=10) 