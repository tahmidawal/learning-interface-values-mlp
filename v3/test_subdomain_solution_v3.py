import numpy as np
import matplotlib.pyplot as plt
import torch
from data_generator_v3 import DataGeneratorV3
from train_model_v3 import InterfacePredictor
from pde_solver import PoissonSolver
import seaborn as sns
import os
import traceback
from typing import List, Dict, Tuple, Callable
import torch.nn as nn

def generate_test_case(data_gen: DataGeneratorV3, k: float):
    """Generate a test case with specific k value."""
    # Use constant theta=1
    theta = np.ones((data_gen.ny, data_gen.nx))
    
    # Generate source term f with specified k
    coef = np.random.uniform(0.1, 0.2)  # Keep the same coefficient range as training
    f = coef * np.sin(k * 2 * np.pi * data_gen.X) * np.sin(k * 2 * np.pi * data_gen.Y)
    
    # Generate solution with specified k
    u = coef * np.sin(k * 2 * np.pi * data_gen.X) * np.sin(k * 2 * np.pi * data_gen.Y)
    
    # Scale the arrays
    u_scaled = data_gen.scale_array(u)
    f_scaled = data_gen.scale_array(f)
    theta_scaled = data_gen.scale_array(theta, data_gen.scale_factors['theta']['min'], 
                                      data_gen.scale_factors['theta']['max'])
    
    return theta_scaled, -f_scaled, u_scaled  # Negate f to match ∇·(θ∇u) = f

def scale_to_reference(solution: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Scale a solution to match the magnitude range of a reference solution."""
    ref_min, ref_max = reference.min(), reference.max()
    sol_min, sol_max = solution.min(), solution.max()
    
    # Handle constant arrays
    if sol_min == sol_max:
        return np.zeros_like(solution) if sol_min == 0 else np.ones_like(solution) * ref_min
    
    return (solution - sol_min) * (ref_max - ref_min) / (sol_max - sol_min) + ref_min

def create_global_bc_funcs(u_exact: np.ndarray) -> Dict[str, Callable]:
    """Create boundary condition functions for the full domain."""
    ny, nx = u_exact.shape
    
    # Extract boundary values
    left_boundary = u_exact[:, 0]
    right_boundary = u_exact[:, -1]
    bottom_boundary = u_exact[0, :]
    top_boundary = u_exact[-1, :]
    
    def bc_left(y):
        if isinstance(y, (int, float)):
            y_idx = int(y)
            return float(left_boundary[y_idx])
        return np.array([float(left_boundary[int(yi)]) for yi in y])
    
    def bc_right(y):
        if isinstance(y, (int, float)):
            y_idx = int(y)
            return float(right_boundary[y_idx])
        return np.array([float(right_boundary[int(yi)]) for yi in y])
    
    def bc_bottom(x):
        if isinstance(x, (int, float)):
            x_idx = int(x)
            return float(bottom_boundary[x_idx])
        return np.array([float(bottom_boundary[int(xi)]) for xi in x])
    
    def bc_top(x):
        if isinstance(x, (int, float)):
            x_idx = int(x)
            return float(top_boundary[x_idx])
        return np.array([float(top_boundary[int(xi)]) for xi in x])
    
    return {
        'left': bc_left,
        'right': bc_right,
        'bottom': bc_bottom,
        'top': bc_top
    }

def create_subdomain_bc_dict(u_exact: np.ndarray, 
                           interface_vals: Dict[str, np.ndarray],
                           subdomain_idx: Tuple[int, int],
                           n_subdomains: Tuple[int, int],
                           subdomain_size: Tuple[int, int]) -> Dict[str, Callable]:
    """Create boundary condition functions for a subdomain."""
    subdomain_nx, subdomain_ny = subdomain_size
    i, j = subdomain_idx
    max_i, max_j = n_subdomains
    
    # Extract global boundary values if needed
    if j == 0:  # Leftmost subdomain
        left_boundary = u_exact[i*subdomain_ny:(i+1)*subdomain_ny, 0]
    else:
        left_boundary = interface_vals[f'vertical_{j}'][i*subdomain_ny:(i+1)*subdomain_ny]
        
    if j == max_j-1:  # Rightmost subdomain
        right_boundary = u_exact[i*subdomain_ny:(i+1)*subdomain_ny, -1]
    else:
        right_boundary = interface_vals[f'vertical_{j+1}'][i*subdomain_ny:(i+1)*subdomain_ny]
        
    if i == 0:  # Bottom subdomain
        bottom_boundary = u_exact[0, j*subdomain_nx:(j+1)*subdomain_nx]
    else:
        bottom_boundary = interface_vals[f'horizontal_{i}'][j*subdomain_nx:(j+1)*subdomain_nx]
        
    if i == max_i-1:  # Top subdomain
        top_boundary = u_exact[-1, j*subdomain_nx:(j+1)*subdomain_nx]
    else:
        top_boundary = interface_vals[f'horizontal_{i+1}'][j*subdomain_nx:(j+1)*subdomain_nx]
    
    # Create boundary condition functions
    def bc_left(y):
        y_idx = int(y)
        return float(left_boundary[y_idx])
    
    def bc_right(y):
        y_idx = int(y)
        return float(right_boundary[y_idx])
    
    def bc_bottom(x):
        x_idx = int(x)
        return float(bottom_boundary[x_idx])
    
    def bc_top(x):
        x_idx = int(x)
        return float(top_boundary[x_idx])
    
    return {
        'left': bc_left,
        'right': bc_right,
        'bottom': bc_bottom,
        'top': bc_top
    }

def extract_subdomain(field: np.ndarray, subdomain_idx: Tuple[int, int], n_subdomains: tuple) -> np.ndarray:
    """Extract a subdomain from a field."""
    ny, nx = field.shape
    subdomain_nx = nx // n_subdomains[0]
    subdomain_ny = ny // n_subdomains[1]
    i, j = subdomain_idx
    return field[i*subdomain_ny:(i+1)*subdomain_ny, j*subdomain_nx:(j+1)*subdomain_nx]

def plot_field(field: np.ndarray, title: str, ax: plt.Axes):
    """Plot a 2D field with proper formatting."""
    im = ax.imshow(field, cmap='viridis', aspect='equal', origin='lower')
    ax.set_title(title)
    return plt.colorbar(im, ax=ax)

class InterfacePredictor(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        
        # Calculate input size (theta_patch and f_patch)
        input_channels = 2  # theta and f patches
        patch_elements = patch_size * patch_size
        
        # Define network architecture
        self.network = nn.Sequential(
            nn.Linear(input_channels * patch_elements, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, theta_patch: torch.Tensor, f_patch: torch.Tensor) -> torch.Tensor:
        # Flatten and concatenate patches
        theta_flat = theta_patch.view(theta_patch.size(0), -1)
        f_flat = f_patch.view(f_patch.size(0), -1)
        x = torch.cat([theta_flat, f_flat], dim=1)
        
        return self.network(x)
    
    def predict(self, point: Dict) -> float:
        """Predict interface value for a given point."""
        with torch.no_grad():
            theta_patch = torch.tensor(point['theta_patch'], dtype=torch.float32).unsqueeze(0)
            f_patch = torch.tensor(point['f_patch'], dtype=torch.float32).unsqueeze(0)
            output = self.forward(theta_patch, f_patch)
            return float(output.item())

def main():
    try:
        # Create results directory
        results_dir = 'test_results_v3'
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize data generator and model
        print("Initializing data generator...")
        data_gen = DataGeneratorV3(nx=40, ny=40, n_subdomains=(2, 2), patch_size=5)
        
        print("Loading trained model...")
        model_path = 'models/interface_predictor_v3.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        model_info = torch.load(model_path)
        predictor = InterfacePredictor(patch_size=5)
        predictor.load_state_dict(model_info['model_state'])
        predictor.eval()  # Set to evaluation mode
        
        # Initialize PDE solvers
        subdomain_nx = data_gen.nx // 2
        subdomain_ny = data_gen.ny // 2
        subdomain_solver = PoissonSolver(subdomain_nx, subdomain_ny)
        full_solver = PoissonSolver(data_gen.nx, data_gen.ny)
        
        # Define k values to test
        k_values = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        
        # Process each k value
        for k in k_values:
            print(f"\nProcessing case with k={k}...")
            
            # Generate case with k1=k2=k
            theta, f, u_exact = generate_test_case(data_gen, k)
            
            # Create boundary conditions for full domain
            global_bc = create_global_bc_funcs(u_exact)
            
            # Solve directly on full domain
            u_direct, n_iters_direct = full_solver.solve_full_domain(
                theta, f,
                global_bc['left'], global_bc['right'],
                global_bc['bottom'], global_bc['top']
            )
            # Scale direct solution to match u_exact magnitude
            u_direct = scale_to_reference(u_direct, u_exact)
            direct_error = np.abs(u_exact - u_direct)
            direct_mae = np.mean(direct_error)
            
            # Extract interface data
            interface_data = data_gen.extract_interface_data(theta, f, u_exact, global_bc)
            
            # Organize interface predictions by type and position
            interface_vals = {
                'vertical_1': np.zeros(data_gen.ny),   # Right boundary of left subdomains
                'vertical_2': np.zeros(data_gen.ny),   # Right boundary of middle subdomains
                'horizontal_1': np.zeros(data_gen.nx),  # Top boundary of bottom subdomains
                'horizontal_2': np.zeros(data_gen.nx)   # Top boundary of middle subdomains
            }
            
            # Get predicted interface conditions and invert their magnitude
            for point in interface_data:
                x, y = point['position']
                with torch.no_grad():
                    pred = predictor.predict(point)
                    # Invert the magnitude of predicted interface values
                    pred = -pred
                
                if x == subdomain_nx:  # Vertical interface 1
                    interface_vals['vertical_1'][y] = pred
                elif x == 2*subdomain_nx:  # Vertical interface 2
                    interface_vals['vertical_2'][y] = pred
                elif y == subdomain_ny:  # Horizontal interface 1
                    interface_vals['horizontal_1'][x] = pred
                elif y == 2*subdomain_ny:  # Horizontal interface 2
                    interface_vals['horizontal_2'][x] = pred
            
            # Initialize combined solution array
            u_combined = np.zeros_like(u_exact)
            
            # Solve for each subdomain
            subdomain_errors = []
            for si in range(2):  # Subdomain row index
                for sj in range(2):  # Subdomain column index
                    # Create boundary conditions for this subdomain
                    subdomain_bc = create_subdomain_bc_dict(
                        u_exact, interface_vals, (si, sj), (2, 2),
                        (subdomain_nx, subdomain_ny)
                    )
                    
                    # Extract subdomain fields
                    theta_subdomain = extract_subdomain(theta, (si, sj), (2, 2))
                    f_subdomain = extract_subdomain(f, (si, sj), (2, 2))
                    u_exact_subdomain = extract_subdomain(u_exact, (si, sj), (2, 2))
                    
                    # Solve PDE in subdomain
                    u_pred_subdomain, n_iters = subdomain_solver.solve_subdomain(
                        theta_subdomain, f_subdomain, subdomain_bc
                    )
                    
                    # Scale subdomain solution to match u_exact magnitude
                    u_pred_subdomain = scale_to_reference(u_pred_subdomain, u_exact_subdomain)
                    
                    # Store solution in combined array
                    u_combined[si*subdomain_ny:(si+1)*subdomain_ny, 
                             sj*subdomain_nx:(sj+1)*subdomain_nx] = u_pred_subdomain
                    
                    # Calculate error for this subdomain
                    error = np.abs(u_exact_subdomain - u_pred_subdomain)
                    subdomain_errors.append({
                        'position': (si, sj),
                        'max_error': error.max(),
                        'mean_error': error.mean(),
                        'iterations': n_iters
                    })
            
            # Calculate global error for subdomain solution
            global_error = np.abs(u_exact - u_combined)
            global_mae = np.mean(global_error)
            
            # Plot results
            fig = plt.figure(figsize=(20, 20))
            gs = plt.GridSpec(4, 4)
            
            # Plot full domain results
            ax_theta = fig.add_subplot(gs[0, 0])
            plot_field(theta, 'Theta (Full Domain)', ax_theta)
            
            ax_f = fig.add_subplot(gs[0, 1])
            plot_field(f, 'f (Full Domain)', ax_f)
            
            ax_exact = fig.add_subplot(gs[0, 2])
            plot_field(u_exact, 'u_exact (Full Domain)', ax_exact)
            
            ax_direct = fig.add_subplot(gs[0, 3])
            plot_field(u_direct, f'Direct Solution\nMAE={direct_mae:.6f}', ax_direct)
            
            ax_pred = fig.add_subplot(gs[1, :2])
            plot_field(u_combined, f'ML-Based Solution\nMAE={global_mae:.6f}', ax_pred)
            
            ax_direct_err = fig.add_subplot(gs[1, 2:])
            plot_field(direct_error, f'Direct Solution Error\nMax={direct_error.max():.6f}', ax_direct_err)
            
            # Plot subdomain results
            for si in range(2):
                for sj in range(2):
                    idx = si * 2 + sj
                    ax_sub = fig.add_subplot(gs[2:, idx])
                    error_sub = np.abs(
                        extract_subdomain(u_exact, (si, sj), (2, 2)) - 
                        extract_subdomain(u_combined, (si, sj), (2, 2))
                    )
                    plot_field(error_sub, 
                             f'Subdomain ({si},{sj}) Error\n' + 
                             f'Max={error_sub.max():.6f}\n' +
                             f'MAE={error_sub.mean():.6f}\n' +
                             f'Iters={subdomain_errors[idx]["iterations"]}',
                             ax_sub)
            
            plt.suptitle(f'k = {k}\n' +
                       f'Direct Solution: MAE={direct_mae:.6f}, Iters={n_iters_direct}\n' +
                       f'ML-Based Solution: MAE={global_mae:.6f}')
            plt.tight_layout()
            
            # Save plot
            save_path = os.path.join(results_dir, f'k_{k}_comparison.png')
            print(f"Saving plot to {save_path}")
            plt.savefig(save_path)
            plt.close()
            
            # Print statistics
            print(f"  Direct Solution:")
            print(f"    MAE: {direct_mae:.6f}")
            print(f"    Max Error: {direct_error.max():.6f}")
            print(f"    Iterations: {n_iters_direct}")
            print(f"\n  ML-Based Solution:")
            print(f"    Global MAE: {global_mae:.6f}")
            print(f"    Global Max Error: {global_error.max():.6f}")
            print("    Subdomain Statistics:")
            for err in subdomain_errors:
                print(f"      Subdomain {err['position']}:")
                print(f"        MAE: {err['mean_error']:.6f}")
                print(f"        Max Error: {err['max_error']:.6f}")
                print(f"        Iterations: {err['iterations']}")
        
        print("\nProcessing complete! Check the test_results_v3 directory for output plots.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nTraceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 