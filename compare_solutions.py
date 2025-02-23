import numpy as np
import matplotlib.pyplot as plt
from data_generator_v2 import DataGeneratorV2
from ml_model_v2 import InterfacePredictorV2
from pde_solver import PoissonSolver
import torch
from typing import Dict, List, Tuple, Callable, Optional
import seaborn as sns

def extract_subdomain_bcs(solution: np.ndarray, x_start: int, x_end: int,
                         y_start: int, y_end: int) -> Dict[str, Callable]:
    """Extract boundary conditions for a subdomain from a solution."""
    def bc_left(y):
        y_idx = int(y * (y_end - y_start)) + y_start
        return float(solution[y_idx, x_start])
    
    def bc_right(y):
        y_idx = int(y * (y_end - y_start)) + y_start
        return float(solution[y_idx, x_end-1])
    
    def bc_bottom(x):
        x_idx = int(x * (x_end - x_start)) + x_start
        return float(solution[y_start, x_idx])
    
    def bc_top(x):
        x_idx = int(x * (x_end - x_start)) + x_start
        return float(solution[y_end-1, x_idx])
    
    return {
        'left': bc_left,
        'right': bc_right,
        'bottom': bc_bottom,
        'top': bc_top
    }

def solve_with_ml_interfaces(predictor: InterfacePredictorV2, theta: np.ndarray,
                           f: np.ndarray, global_bc_dict: Dict[str, Callable],
                           nx: int = 40, ny: int = 40) -> np.ndarray:
    """
    Solve the PDE using ML-predicted interface conditions.
    
    Args:
        predictor: Trained ML model for interface prediction
        theta: Diffusion coefficient field
        f: Source term
        global_bc_dict: Global boundary conditions
        nx, ny: Grid dimensions
    
    Returns:
        np.ndarray: Combined solution
    """
    # Initialize solution array and solvers
    solution = np.zeros((ny, nx))
    subdomain_nx = nx // 2
    subdomain_ny = ny // 2
    solvers = [[PoissonSolver(subdomain_nx, subdomain_ny) for _ in range(2)] for _ in range(2)]
    
    # Get ML predictions for interface values
    interface_values = np.zeros((ny, nx))
    mask = np.zeros((ny, nx), dtype=bool)
    
    # Helper function to extract padded patch
    def extract_patch(field, center_x, center_y, patch_size=5):
        half_size = patch_size // 2
        padded_field = np.pad(field, ((half_size, half_size), (half_size, half_size)), mode='edge')
        center_y_pad = center_y + half_size
        center_x_pad = center_x + half_size
        return padded_field[center_y_pad-half_size:center_y_pad+half_size+1,
                          center_x_pad-half_size:center_x_pad+half_size+1]
    
    # Get global boundary values for normalization
    global_bc_values = []
    y_coords = np.linspace(0, 1, ny)
    x_coords = np.linspace(0, 1, nx)
    for y in y_coords:
        global_bc_values.append(global_bc_dict['left'](y))
        global_bc_values.append(global_bc_dict['right'](y))
    for x in x_coords:
        global_bc_values.append(global_bc_dict['bottom'](x))
        global_bc_values.append(global_bc_dict['top'](x))
    global_bc_values = np.array(global_bc_values)
    bc_mean = np.mean(global_bc_values)
    bc_std = np.std(global_bc_values) + 1e-8
    
    # Vertical interface (x = nx//2)
    for y in range(ny):
        interface_point = {
            'type': 'vertical',
            'position': (nx//2, y),
            'theta_patch': extract_patch(theta, nx//2, y),
            'f_patch': extract_patch(f, nx//2, y),
            'boundary_values': (np.array([
                global_bc_dict['left'](y/ny),
                global_bc_dict['right'](y/ny),
                global_bc_dict['bottom'](0.5),
                global_bc_dict['top'](0.5)
            ]) - bc_mean) / bc_std
        }
        pred = predictor.predict(interface_point)
        interface_values[y, nx//2] = pred * bc_std + bc_mean
        mask[y, nx//2] = True
    
    # Horizontal interface (y = ny//2)
    for x in range(nx):
        interface_point = {
            'type': 'horizontal',
            'position': (x, ny//2),
            'theta_patch': extract_patch(theta, x, ny//2),
            'f_patch': extract_patch(f, x, ny//2),
            'boundary_values': (np.array([
                global_bc_dict['left'](0.5),
                global_bc_dict['right'](0.5),
                global_bc_dict['bottom'](x/nx),
                global_bc_dict['top'](x/nx)
            ]) - bc_mean) / bc_std
        }
        pred = predictor.predict(interface_point)
        interface_values[ny//2, x] = pred * bc_std + bc_mean
        mask[ny//2, x] = True
    
    # Solve each subdomain
    for i in range(2):
        for j in range(2):
            y_start = i * subdomain_ny
            y_end = (i + 1) * subdomain_ny
            x_start = j * subdomain_nx
            x_end = (j + 1) * subdomain_nx
            
            # Extract subdomain data
            theta_sub = theta[y_start:y_end, x_start:x_end]
            f_sub = f[y_start:y_end, x_start:x_end]
            
            # Create boundary conditions for subdomain
            bc_dict = {}
            
            # Left boundary
            if j == 0:
                bc_dict['left'] = global_bc_dict['left']
            else:
                def bc_left(y):
                    y_idx = min(int(y * subdomain_ny) + y_start, ny - 1)
                    return float(interface_values[y_idx, x_start])
                bc_dict['left'] = bc_left
            
            # Right boundary
            if j == 1:
                bc_dict['right'] = global_bc_dict['right']
            else:
                def bc_right(y):
                    y_idx = min(int(y * subdomain_ny) + y_start, ny - 1)
                    return float(interface_values[y_idx, x_end])
                bc_dict['right'] = bc_right
            
            # Bottom boundary
            if i == 0:
                bc_dict['bottom'] = global_bc_dict['bottom']
            else:
                def bc_bottom(x):
                    x_idx = min(int(x * subdomain_nx) + x_start, nx - 1)
                    return float(interface_values[y_start, x_idx])
                bc_dict['bottom'] = bc_bottom
            
            # Top boundary
            if i == 1:
                bc_dict['top'] = global_bc_dict['top']
            else:
                def bc_top(x):
                    x_idx = min(int(x * subdomain_nx) + x_start, nx - 1)
                    return float(interface_values[y_end, x_idx])
                bc_dict['top'] = bc_top
            
            # Solve subdomain
            sol_sub, _ = solvers[i][j].solve_subdomain(theta_sub, f_sub, bc_dict)
            
            # Store solution
            solution[y_start:y_end, x_start:x_end] = sol_sub
    
    return solution, interface_values, mask

def plot_solution_comparison(analytical: np.ndarray, numerical: np.ndarray,
                           ml_subdomain: np.ndarray, interface_values: np.ndarray,
                           interface_mask: np.ndarray, save_path: Optional[str] = None):
    """Plot and compare different solutions with subdomain boundaries and interface values."""
    fig = plt.figure(figsize=(20, 15))
    
    # Plot analytical solution
    ax1 = plt.subplot(231)
    im1 = ax1.imshow(analytical, cmap='viridis')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title('Analytical Solution')
    
    # Plot numerical solution
    ax2 = plt.subplot(232)
    im2 = ax2.imshow(numerical, cmap='viridis')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('Full Numerical Solution')
    
    # Plot ML-based subdomain solution with boundaries
    ax3 = plt.subplot(233)
    im3 = ax3.imshow(ml_subdomain, cmap='viridis')
    plt.colorbar(im3, ax=ax3)
    
    # Add subdomain boundaries
    nx, ny = ml_subdomain.shape
    ax3.axvline(x=nx//2, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=ny//2, color='r', linestyle='--', alpha=0.5)
    
    # Plot interface values
    interface_x = np.where(interface_mask)[1]
    interface_y = np.where(interface_mask)[0]
    ax3.scatter(interface_x, interface_y, c='red', marker='x', s=50, label='Interface Points')
    ax3.legend()
    ax3.set_title('ML-Based Subdomain Solution\nwith Interface Points')
    
    # Plot difference between analytical and ML solution
    ax4 = plt.subplot(234)
    diff1 = np.abs(analytical - ml_subdomain)
    im4 = ax4.imshow(diff1, cmap='viridis')
    plt.colorbar(im4, ax=ax4)
    ax4.axvline(x=nx//2, color='r', linestyle='--', alpha=0.5)
    ax4.axhline(y=ny//2, color='r', linestyle='--', alpha=0.5)
    ax4.set_title('|Analytical - ML Solution|')
    
    # Plot difference between numerical and ML solution
    ax5 = plt.subplot(235)
    diff2 = np.abs(numerical - ml_subdomain)
    im5 = ax5.imshow(diff2, cmap='viridis')
    plt.colorbar(im5, ax=ax5)
    ax5.axvline(x=nx//2, color='r', linestyle='--', alpha=0.5)
    ax5.axhline(y=ny//2, color='r', linestyle='--', alpha=0.5)
    ax5.set_title('|Numerical - ML Solution|')
    
    # Plot subdomains separately
    ax6 = plt.subplot(236)
    subdomain_view = np.ma.masked_array(ml_subdomain.copy(), np.zeros_like(ml_subdomain, dtype=bool))
    
    # Add grid lines for subdomains
    ax6.axvline(x=nx//2, color='r', linestyle='--', alpha=0.5)
    ax6.axhline(y=ny//2, color='r', linestyle='--', alpha=0.5)
    
    # Plot interface values
    masked_interfaces = np.ma.masked_array(interface_values, ~interface_mask)
    im6 = ax6.imshow(subdomain_view, cmap='viridis')
    im6_interface = ax6.imshow(masked_interfaces, cmap='Reds', alpha=0.7)
    plt.colorbar(im6, ax=ax6, label='Subdomain Solution')
    plt.colorbar(im6_interface, ax=ax6, label='Interface Values')
    ax6.set_title('Subdomains with Interface Values')
    
    # Add text labels for subdomains
    ax6.text(nx//4, ny//4, 'Subdomain 1', ha='center', va='center', color='white')
    ax6.text(3*nx//4, ny//4, 'Subdomain 2', ha='center', va='center', color='white')
    ax6.text(nx//4, 3*ny//4, 'Subdomain 3', ha='center', va='center', color='white')
    ax6.text(3*nx//4, 3*ny//4, 'Subdomain 4', ha='center', va='center', color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate error metrics
    metrics = {
        'ml_vs_analytical': {
            'mae': np.mean(diff1),
            'rmse': np.sqrt(np.mean(diff1**2)),
            'max_error': np.max(diff1)
        },
        'ml_vs_numerical': {
            'mae': np.mean(diff2),
            'rmse': np.sqrt(np.mean(diff2**2)),
            'max_error': np.max(diff2)
        },
        'numerical_vs_analytical': {
            'mae': np.mean(np.abs(numerical - analytical)),
            'rmse': np.sqrt(np.mean((numerical - analytical)**2)),
            'max_error': np.max(np.abs(numerical - analytical))
        }
    }
    
    return metrics

def run_multiple_comparisons(n_samples: int = 15):
    """Run multiple comparisons and collect statistics."""
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs('comparison_results', exist_ok=True)
    
    # Create data generator
    data_gen = DataGeneratorV2(nx=40, ny=40, n_subdomains=(2, 2), patch_size=5)
    
    # Load trained model
    print("Loading trained model...")
    predictor = InterfacePredictorV2(patch_size=5)
    predictor.load_model('models/interface_predictor_v2.pth')
    
    # Initialize lists to store metrics
    all_metrics = []
    
    print(f"\nRunning {n_samples} comparisons...")
    for i in range(n_samples):
        print(f"\nComparison {i+1}/{n_samples}")
        
        # Generate a test problem
        print("Generating test problem...")
        theta, f, analytical_solution = data_gen.generate_random_problem()
        bc_dict = data_gen.generate_boundary_conditions(analytical_solution)
        
        # Solve full domain numerically
        print("Solving full domain numerically...")
        full_solver = PoissonSolver(40, 40)
        numerical_solution, _ = full_solver.solve_full_domain(
            theta, f,
            bc_dict['left'], bc_dict['right'],
            bc_dict['bottom'], bc_dict['top']
        )
        
        # Solve using ML-predicted interface conditions
        print("Solving with ML-predicted interface conditions...")
        ml_solution, interface_values, interface_mask = solve_with_ml_interfaces(
            predictor, theta, f, bc_dict
        )
        
        # Generate and save comparison plot
        print("Generating comparison plot...")
        metrics = plot_solution_comparison(
            analytical_solution,
            numerical_solution,
            ml_solution,
            interface_values,
            interface_mask,
            save_path=f'comparison_results/comparison_{i+1}.png'
        )
        
        all_metrics.append(metrics)
    
    # Calculate and plot statistical summary
    print("\nCalculating statistical summary...")
    
    # Extract metrics
    ml_analytical_mae = [m['ml_vs_analytical']['mae'] for m in all_metrics]
    ml_analytical_rmse = [m['ml_vs_analytical']['rmse'] for m in all_metrics]
    ml_numerical_mae = [m['ml_vs_numerical']['mae'] for m in all_metrics]
    ml_numerical_rmse = [m['ml_vs_numerical']['rmse'] for m in all_metrics]
    numerical_analytical_mae = [m['numerical_vs_analytical']['mae'] for m in all_metrics]
    numerical_analytical_rmse = [m['numerical_vs_analytical']['rmse'] for m in all_metrics]
    
    # Create error comparison plots
    plt.figure(figsize=(15, 10))
    
    # MAE comparison
    plt.subplot(211)
    plt.boxplot([ml_analytical_mae, ml_numerical_mae, numerical_analytical_mae],
                labels=['ML vs Analytical', 'ML vs Numerical', 'Numerical vs Analytical'])
    plt.title('Mean Absolute Error Comparison')
    plt.ylabel('Error')
    plt.grid(True)
    
    # RMSE comparison
    plt.subplot(212)
    plt.boxplot([ml_analytical_rmse, ml_numerical_rmse, numerical_analytical_rmse],
                labels=['ML vs Analytical', 'ML vs Numerical', 'Numerical vs Analytical'])
    plt.title('Root Mean Square Error Comparison')
    plt.ylabel('Error')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('comparison_results/error_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistical summary
    print("\nStatistical Summary:")
    print("\nMean Absolute Error:")
    print(f"ML vs Analytical: {np.mean(ml_analytical_mae):.6f} ± {np.std(ml_analytical_mae):.6f}")
    print(f"ML vs Numerical: {np.mean(ml_numerical_mae):.6f} ± {np.std(ml_numerical_mae):.6f}")
    print(f"Numerical vs Analytical: {np.mean(numerical_analytical_mae):.6f} ± {np.std(numerical_analytical_mae):.6f}")
    
    print("\nRoot Mean Square Error:")
    print(f"ML vs Analytical: {np.mean(ml_analytical_rmse):.6f} ± {np.std(ml_analytical_rmse):.6f}")
    print(f"ML vs Numerical: {np.mean(ml_numerical_rmse):.6f} ± {np.std(ml_numerical_rmse):.6f}")
    print(f"Numerical vs Analytical: {np.mean(numerical_analytical_rmse):.6f} ± {np.std(numerical_analytical_rmse):.6f}")

if __name__ == "__main__":
    run_multiple_comparisons(n_samples=15) 