import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from data_generator_v4 import DataGeneratorV4
from test_subdomain_solution_v4 import InterfacePredictor, generate_test_case, extract_subdomain
import seaborn as sns
import os
from typing import List, Dict, Tuple, Callable

def extract_patch_safe(data_gen, field, center_x, center_y):
    """Extract a patch safely, handling boundary cases."""
    half_size = data_gen.patch_size // 2
    
    # Calculate patch boundaries with padding if needed
    y_start = max(0, center_y - half_size)
    y_end = min(field.shape[0], center_y + half_size + 1)
    x_start = max(0, center_x - half_size)
    x_end = min(field.shape[1], center_x + half_size + 1)
    
    # Extract the patch
    patch = field[y_start:y_end, x_start:x_end]
    
    # Pad if necessary to ensure correct size
    if patch.shape[0] < data_gen.patch_size or patch.shape[1] < data_gen.patch_size:
        padded_patch = np.zeros((data_gen.patch_size, data_gen.patch_size))
        
        # Calculate where to place the extracted patch in the padded patch
        y_offset = max(0, half_size - center_y)
        x_offset = max(0, half_size - center_x)
        
        # Place the extracted patch in the padded patch
        padded_patch[y_offset:y_offset + patch.shape[0], 
                    x_offset:x_offset + patch.shape[1]] = patch
        
        return padded_patch
    
    return patch

def plot_interface_values(data_gen, predictor, theta, f, u_exact, k, case_num, results_dir):
    """
    Plot the values along vertical and horizontal interfaces, comparing:
    1. Numerical solution
    2. ML predicted values
    3. ML-based solution values
    """
    # Initialize PDE solvers
    from pde_solver import PoissonSolver
    subdomain_nx = data_gen.nx // 2
    subdomain_ny = data_gen.ny // 2
    subdomain_solver = PoissonSolver(subdomain_nx, subdomain_ny)
    
    # Extract interface data
    interface_data = []
    
    # Extract vertical interface points
    x = data_gen.nx // 2  # Middle of the domain (vertical interface)
    for y in range(data_gen.ny):
        theta_patch = extract_patch_safe(data_gen, theta, x, y)
        f_patch = extract_patch_safe(data_gen, f, x, y)
        true_value = u_exact[y, x]
        
        interface_data.append({
            'position': (x, y),
            'type': 'vertical',
            'theta_patch': theta_patch,
            'f_patch': f_patch,
            'target': true_value
        })
    
    # Extract horizontal interface points
    y = data_gen.ny // 2  # Middle of the domain (horizontal interface)
    for x in range(data_gen.nx):
        theta_patch = extract_patch_safe(data_gen, theta, x, y)
        f_patch = extract_patch_safe(data_gen, f, x, y)
        true_value = u_exact[y, x]
        
        interface_data.append({
            'position': (x, y),
            'type': 'horizontal',
            'theta_patch': theta_patch,
            'f_patch': f_patch,
            'target': true_value
        })
    
    # Organize interface predictions by type and position
    interface_vals = {
        'vertical': np.zeros(data_gen.ny),   # Vertical interface at x=nx/2
        'horizontal': np.zeros(data_gen.nx)  # Horizontal interface at y=ny/2
    }
    
    # Get predicted interface conditions
    for point in interface_data:
        x, y = point['position']
        with torch.no_grad():
            pred = predictor.predict(point)
        
        if point['type'] == 'vertical':
            interface_vals['vertical'][y] = pred
        else:  # horizontal
            interface_vals['horizontal'][x] = pred
    
    # Create boundary conditions for subdomains
    def create_subdomain_bc_dict(u_exact, interface_vals, subdomain_idx, n_subdomains, subdomain_size):
        si, sj = subdomain_idx
        nx, ny = subdomain_size
        
        # Create boundary condition functions
        def bc_left(y_normalized):
            y_global = si*ny + int(y_normalized*ny)
            if y_global >= data_gen.ny:
                y_global = data_gen.ny - 1
                
            if sj == 0:  # Left edge of domain
                return u_exact[y_global, 0]
            else:  # Interior interface
                return interface_vals['vertical'][y_global]
                
        def bc_right(y_normalized):
            y_global = si*ny + int(y_normalized*ny)
            if y_global >= data_gen.ny:
                y_global = data_gen.ny - 1
                
            if sj == n_subdomains[1]-1:  # Right edge of domain
                return u_exact[y_global, -1]
            else:  # Interior interface
                return interface_vals['vertical'][y_global]
                
        def bc_bottom(x_normalized):
            x_global = sj*nx + int(x_normalized*nx)
            if x_global >= data_gen.nx:
                x_global = data_gen.nx - 1
                
            if si == 0:  # Bottom edge of domain
                return u_exact[0, x_global]
            else:  # Interior interface
                return interface_vals['horizontal'][x_global]
                
        def bc_top(x_normalized):
            x_global = sj*nx + int(x_normalized*nx)
            if x_global >= data_gen.nx:
                x_global = data_gen.nx - 1
                
            if si == n_subdomains[0]-1:  # Top edge of domain
                return u_exact[-1, x_global]
            else:  # Interior interface
                return interface_vals['horizontal'][x_global]
                
        return {
            'left': bc_left,
            'right': bc_right,
            'bottom': bc_bottom,
            'top': bc_top
        }
    
    # Solve for each subdomain to get ML-based solution
    u_combined = np.zeros_like(u_exact)
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
            
            # Solve PDE in subdomain
            u_pred_subdomain, n_iters = subdomain_solver.solve_subdomain(
                theta_subdomain, f_subdomain, subdomain_bc
            )
            
            # Insert subdomain solution into combined solution
            i_start, i_end = si * subdomain_ny, (si + 1) * subdomain_ny
            j_start, j_end = sj * subdomain_nx, (sj + 1) * subdomain_nx
            u_combined[i_start:i_end, j_start:j_end] = u_pred_subdomain
    
    # Plot vertical interface values
    plt.figure(figsize=(10, 6))
    y_coords = np.arange(data_gen.ny)
    
    # Get values along vertical interface from numerical solution
    vertical_interface_x = data_gen.nx // 2
    numerical_values = u_exact[:, vertical_interface_x]
    
    # Get values along vertical interface from ML-based solution
    ml_solution_values = u_combined[:, vertical_interface_x]
    
    # Plot the values
    plt.plot(y_coords, numerical_values, 'b-', linewidth=2, label='Numerical')
    plt.plot(y_coords, interface_vals['vertical'], 'r--', linewidth=2, label='ML Predicted')
    plt.plot(y_coords, ml_solution_values, 'g:', linewidth=2, label='ML Solution')
    
    plt.grid(True)
    plt.xlabel('Y coordinate')
    plt.ylabel('Solution value')
    plt.title(f'Values along Vertical Interface (x={vertical_interface_x}, k={k:.1f})')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{results_dir}/k_{k:.1f}_case_{case_num}_vertical_interface.png')
    
    # Plot horizontal interface values
    plt.figure(figsize=(10, 6))
    x_coords = np.arange(data_gen.nx)
    
    # Get values along horizontal interface from numerical solution
    horizontal_interface_y = data_gen.ny // 2
    numerical_values = u_exact[horizontal_interface_y, :]
    
    # Get values along horizontal interface from ML-based solution
    ml_solution_values = u_combined[horizontal_interface_y, :]
    
    # Plot the values
    plt.plot(x_coords, numerical_values, 'b-', linewidth=2, label='Numerical')
    plt.plot(x_coords, interface_vals['horizontal'], 'r--', linewidth=2, label='ML Predicted')
    plt.plot(x_coords, ml_solution_values, 'g:', linewidth=2, label='ML Solution')
    
    plt.grid(True)
    plt.xlabel('X coordinate')
    plt.ylabel('Solution value')
    plt.title(f'Values along Horizontal Interface (y={horizontal_interface_y}, k={k:.1f})')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{results_dir}/k_{k:.1f}_case_{case_num}_horizontal_interface.png')

def main():
    # Create results directory
    results_dir = 'interface_values_results_v4'
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize data generator and model
    print("Initializing data generator...")
    data_gen = DataGeneratorV4(nx=40, ny=40, n_subdomains=(2, 2), patch_size=5)
    
    print("Loading trained model...")
    model_path = 'models/interface_predictor_v4.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    model_info = torch.load(model_path)
    predictor = InterfacePredictor(patch_size=5)
    predictor.load_state_dict(model_info['model_state'])
    predictor.eval()  # Set to evaluation mode
    
    # Generate test cases with different k values
    k_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    n_cases_per_k = 2
    
    for k in k_values:
        print(f"\nGenerating cases for k={k:.1f}...")
        
        for i in range(n_cases_per_k):
            print(f"Processing case {i+1} for k={k:.1f}...")
            
            # Generate case with specific k value
            theta, f, u_exact = generate_test_case(data_gen, k)
            
            # Plot interface values
            plot_interface_values(data_gen, predictor, theta, f, u_exact, k, i+1, results_dir)
    
    print(f"\nProcessing complete! Check the {results_dir} directory for output plots.")

if __name__ == "__main__":
    main() 