import numpy as np
import matplotlib.pyplot as plt
import torch
from data_generator_v2 import DataGeneratorV2
from ml_model_v2 import InterfacePredictorV2
from test_subdomain_solution import generate_in_distribution_case, generate_out_of_distribution_case, extract_subdomain
import seaborn as sns
import os
from typing import List, Dict, Tuple, Callable

def plot_interface_values(data_gen, predictor, theta, f, u_exact, bc_dict, case_type, case_num, results_dir):
    """
    Plot the values along vertical and horizontal interfaces, comparing:
    1. Analytical solution
    2. ML predicted values
    3. ML-based solution values
    """
    # Initialize PDE solvers
    from pde_solver import PoissonSolver
    subdomain_nx = data_gen.nx // 2
    subdomain_ny = data_gen.ny // 2
    subdomain_solver = PoissonSolver(subdomain_nx, subdomain_ny)
    
    # Extract interface data
    interface_data = data_gen.extract_interface_data(theta, f, u_exact, bc_dict)
    
    # Organize interface predictions by type and position
    interface_vals = {
        'vertical_1': np.zeros(data_gen.ny),   # Right boundary of left subdomains
        'vertical_2': np.zeros(data_gen.ny),   # Right boundary of middle subdomains
        'horizontal_1': np.zeros(data_gen.nx),  # Top boundary of bottom subdomains
        'horizontal_2': np.zeros(data_gen.nx)   # Top boundary of middle subdomains
    }
    
    # Get predicted interface conditions
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
                return interface_vals['vertical_1'][y_global]
                
        def bc_right(y_normalized):
            y_global = si*ny + int(y_normalized*ny)
            if y_global >= data_gen.ny:
                y_global = data_gen.ny - 1
                
            if sj == n_subdomains[1]-1:  # Right edge of domain
                return u_exact[y_global, -1]
            else:  # Interior interface
                return interface_vals['vertical_2'][y_global]
                
        def bc_bottom(x_normalized):
            x_global = sj*nx + int(x_normalized*nx)
            if x_global >= data_gen.nx:
                x_global = data_gen.nx - 1
                
            if si == 0:  # Bottom edge of domain
                return u_exact[0, x_global]
            else:  # Interior interface
                return interface_vals['horizontal_1'][x_global]
                
        def bc_top(x_normalized):
            x_global = sj*nx + int(x_normalized*nx)
            if x_global >= data_gen.nx:
                x_global = data_gen.nx - 1
                
            if si == n_subdomains[0]-1:  # Top edge of domain
                return u_exact[-1, x_global]
            else:  # Interior interface
                return interface_vals['horizontal_2'][x_global]
                
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
    
    # Get values along vertical interface from analytical solution
    vertical_interface_x = data_gen.nx // 2
    analytical_values = u_exact[:, vertical_interface_x]
    
    # Get values along vertical interface from ML-based solution
    ml_solution_values = u_combined[:, vertical_interface_x]
    
    # Plot the values
    plt.plot(y_coords, analytical_values, 'b-', linewidth=2, label='Analytical')
    plt.plot(y_coords, interface_vals['vertical_1'], 'r--', linewidth=2, label='ML Predicted')
    plt.plot(y_coords, ml_solution_values, 'g:', linewidth=2, label='ML Solution')
    
    plt.grid(True)
    plt.xlabel('Y coordinate')
    plt.ylabel('Solution value')
    plt.title(f'Values along Vertical Interface (x={vertical_interface_x})')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{results_dir}/{case_type}_case_{case_num}_vertical_interface.png')
    
    # Plot horizontal interface values
    plt.figure(figsize=(10, 6))
    x_coords = np.arange(data_gen.nx)
    
    # Get values along horizontal interface from analytical solution
    horizontal_interface_y = data_gen.ny // 2
    analytical_values = u_exact[horizontal_interface_y, :]
    
    # Get values along horizontal interface from ML-based solution
    ml_solution_values = u_combined[horizontal_interface_y, :]
    
    # Plot the values
    plt.plot(x_coords, analytical_values, 'b-', linewidth=2, label='Analytical')
    plt.plot(x_coords, interface_vals['horizontal_1'], 'r--', linewidth=2, label='ML Predicted')
    plt.plot(x_coords, ml_solution_values, 'g:', linewidth=2, label='ML Solution')
    
    plt.grid(True)
    plt.xlabel('X coordinate')
    plt.ylabel('Solution value')
    plt.title(f'Values along Horizontal Interface (y={horizontal_interface_y})')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'{results_dir}/{case_type}_case_{case_num}_horizontal_interface.png')

def main():
    # Create results directory
    results_dir = 'interface_values_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize data generator and model
    print("Initializing data generator...")
    data_gen = DataGeneratorV2(nx=40, ny=40, n_subdomains=(2, 2), patch_size=5)
    
    print("Loading trained model...")
    model_path = 'models/interface_predictor_v2.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    model_info = torch.load(model_path)
    predictor = InterfacePredictorV2(patch_size=5)
    predictor.model.load_state_dict(model_info['model_state'])
    predictor.model.eval()  # Set to evaluation mode
    
    # Generate test cases
    n_cases = 3
    for case_type in ['in_distribution', 'out_of_distribution']:
        print(f"\nGenerating {case_type} cases...")
        
        for i in range(n_cases):
            print(f"Processing {case_type} case {i+1}...")
            
            # Generate case
            if case_type == 'in_distribution':
                theta, f, u_exact, bc_dict = generate_in_distribution_case(data_gen)
            else:
                theta, f, u_exact, bc_dict = generate_out_of_distribution_case(data_gen)
            
            # Plot interface values
            plot_interface_values(data_gen, predictor, theta, f, u_exact, bc_dict, case_type, i+1, results_dir)
    
    print(f"\nProcessing complete! Check the {results_dir} directory for output plots.")

if __name__ == "__main__":
    main() 