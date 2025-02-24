import numpy as np
import matplotlib.pyplot as plt
import torch
from data_generator_v2 import DataGeneratorV2
from ml_model_v2 import InterfacePredictorV2
from pde_solver import PoissonSolver
import os

def unscale_value(value: float, scale_min: float, scale_max: float,
                  target_min: float, target_max: float) -> float:
    """Unscale a value from the scaled range back to the original range."""
    return (value - scale_min) * (target_max - target_min) / (scale_max - scale_min) + target_min

def unscale_array(arr: np.ndarray, scale_min: float, scale_max: float,
                 target_min: float, target_max: float) -> np.ndarray:
    """Unscale an array from the scaled range back to the original range."""
    return (arr - scale_min) * (target_max - target_min) / (scale_max - scale_min) + target_min

def plot_comparison(solutions: dict, interface_predictions: dict = None, title: str = "Solution Comparison"):
    """
    Plot multiple solutions side by side with their differences.
    
    Args:
        solutions: Dictionary of solution arrays
        interface_predictions: Dictionary containing vertical and horizontal predictions
        title: Plot title
    """
    n_solutions = len(solutions)
    fig = plt.figure(figsize=(5*n_solutions, 10))
    
    # Plot solutions
    for i, (name, sol) in enumerate(solutions.items()):
        ax = plt.subplot(2, n_solutions, i+1)
        
        # Create a copy of the solution to modify with interface values
        plot_sol = sol.copy()
        
        # Add interface lines and values only for ML Interface solution
        if interface_predictions is not None and name == 'ML Interface':
            # Plot vertical interfaces
            if 'vertical' in interface_predictions:
                for x in interface_predictions['vertical'].keys():
                    # Add vertical interface line
                    ax.axvline(x=x, color='red', linestyle='--', linewidth=2, alpha=0.8)
                    # Set interface values
                    for y in range(40):
                        plot_sol[y, x] = interface_predictions['vertical'][x][y]
            
            # Plot horizontal interfaces
            if 'horizontal' in interface_predictions:
                for y in interface_predictions['horizontal'].keys():
                    # Add horizontal interface line
                    ax.axhline(y=y, color='red', linestyle='--', linewidth=2, alpha=0.8)
                    # Set interface values
                    for x in range(40):
                        plot_sol[y, x] = interface_predictions['horizontal'][y][x]
        
        im = ax.imshow(plot_sol, cmap='viridis', aspect='equal')
        ax.set_title(name)
        plt.colorbar(im, ax=ax)
    
    # Plot differences with analytical
    analytical = solutions['Analytical']
    for i, (name, sol) in enumerate(solutions.items()):
        if name != 'Analytical':
            ax = plt.subplot(2, n_solutions, n_solutions + i + 1)
            diff = np.abs(sol - analytical)
            im = ax.imshow(diff, cmap='viridis', aspect='equal')
            ax.set_title(f'Error: {name} vs Analytical')
            plt.colorbar(im, ax=ax)
            
            # Add interface lines to difference plots only for ML Interface solution
            if interface_predictions is not None and name == 'ML Interface':
                # Plot vertical interfaces
                if 'vertical' in interface_predictions:
                    for x in interface_predictions['vertical'].keys():
                        ax.axvline(x=x, color='red', linestyle='--', linewidth=2, alpha=0.8)
                
                # Plot horizontal interfaces
                if 'horizontal' in interface_predictions:
                    for y in interface_predictions['horizontal'].keys():
                        ax.axhline(y=y, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def create_bc_functions(values: np.ndarray, nx: int, ny: int):
    """Create boundary condition functions from values array."""
    def bc_left(y):
        if isinstance(y, np.ndarray):
            indices = np.clip((y * (ny - 1)).astype(int), 0, ny - 1)
            return values[indices, 0]
        return float(values[min(int(y * (ny - 1)), ny - 1), 0])
    
    def bc_right(y):
        if isinstance(y, np.ndarray):
            indices = np.clip((y * (ny - 1)).astype(int), 0, ny - 1)
            return values[indices, -1]
        return float(values[min(int(y * (ny - 1)), ny - 1), -1])
    
    def bc_bottom(x):
        if isinstance(x, np.ndarray):
            indices = np.clip((x * (nx - 1)).astype(int), 0, nx - 1)
            return values[0, indices]
        return float(values[0, min(int(x * (nx - 1)), nx - 1)])
    
    def bc_top(x):
        if isinstance(x, np.ndarray):
            indices = np.clip((x * (nx - 1)).astype(int), 0, nx - 1)
            return values[-1, indices]
        return float(values[-1, min(int(x * (nx - 1)), nx - 1)])
    
    return {
        'left': bc_left,
        'right': bc_right,
        'bottom': bc_bottom,
        'top': bc_top
    }

def create_bc_functions_unscaled(values: np.ndarray, nx: int, ny: int, scale_factors: dict):
    """Create boundary condition functions that unscale values for the PDE solver."""
    def bc_left(y):
        if isinstance(y, np.ndarray):
            indices = np.clip((y * (ny - 1)).astype(int), 0, ny - 1)
            scaled_vals = values[indices, 0]
            return unscale_array(scaled_vals, -1, 1, scale_factors['solution']['min'], scale_factors['solution']['max'])
        scaled_val = float(values[min(int(y * (ny - 1)), ny - 1), 0])
        return unscale_value(scaled_val, -1, 1, scale_factors['solution']['min'], scale_factors['solution']['max'])
    
    def bc_right(y):
        if isinstance(y, np.ndarray):
            indices = np.clip((y * (ny - 1)).astype(int), 0, ny - 1)
            scaled_vals = values[indices, -1]
            return unscale_array(scaled_vals, -1, 1, scale_factors['solution']['min'], scale_factors['solution']['max'])
        scaled_val = float(values[min(int(y * (ny - 1)), ny - 1), -1])
        return unscale_value(scaled_val, -1, 1, scale_factors['solution']['min'], scale_factors['solution']['max'])
    
    def bc_bottom(x):
        if isinstance(x, np.ndarray):
            indices = np.clip((x * (nx - 1)).astype(int), 0, nx - 1)
            scaled_vals = values[0, indices]
            return unscale_array(scaled_vals, -1, 1, scale_factors['solution']['min'], scale_factors['solution']['max'])
        scaled_val = float(values[0, min(int(x * (nx - 1)), nx - 1)])
        return unscale_value(scaled_val, -1, 1, scale_factors['solution']['min'], scale_factors['solution']['max'])
    
    def bc_top(x):
        if isinstance(x, np.ndarray):
            indices = np.clip((x * (nx - 1)).astype(int), 0, nx - 1)
            scaled_vals = values[-1, indices]
            return unscale_array(scaled_vals, -1, 1, scale_factors['solution']['min'], scale_factors['solution']['max'])
        scaled_val = float(values[-1, min(int(x * (nx - 1)), nx - 1)])
        return unscale_value(scaled_val, -1, 1, scale_factors['solution']['min'], scale_factors['solution']['max'])
    
    return {
        'left': bc_left,
        'right': bc_right,
        'bottom': bc_bottom,
        'top': bc_top
    }

def compare_solutions():
    """Compare ML-predicted interface solution with analytical and full domain solutions."""
    # Create results directory
    results_dir = 'comparison_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize data generator and solvers
    print("Initializing...")
    data_gen = DataGeneratorV2(nx=40, ny=40, n_subdomains=(2, 2), patch_size=5)
    full_solver = PoissonSolver(nx=40, ny=40)
    subdomain_solver = PoissonSolver(nx=20, ny=20)  # For 20x20 subdomains
    
    # Load trained model
    print("Loading ML model...")
    model_info = torch.load('models/interface_predictor_v2.pth')
    predictor = InterfacePredictorV2(patch_size=5)
    predictor.model.load_state_dict(model_info['model_state'])
    scale_factors = model_info['scale_factors']
    
    # Generate a test problem
    print("Generating test problem...")
    theta_scaled, f_scaled, u_analytical_scaled = data_gen.generate_random_problem()
    
    # Unscale the values for PDE solving
    theta = unscale_array(theta_scaled, -1, 1, 
                         scale_factors['theta']['min'], 
                         scale_factors['theta']['max'])
    f = unscale_array(f_scaled, -1, 1,
                      scale_factors['f']['min'],
                      scale_factors['f']['max'])
    
    # Create boundary condition functions for unscaled PDE solving
    bc_dict = create_bc_functions_unscaled(u_analytical_scaled, 40, 40, scale_factors)
    
    # Solve full domain with unscaled boundary conditions
    print("Solving full domain...")
    u_full, _ = full_solver.solve_full_domain(
        theta, f,
        bc_dict['left'], bc_dict['right'],
        bc_dict['bottom'], bc_dict['top']
    )
    
    # Get ML predictions for interfaces (keep them scaled between -1 and 1)
    print("Getting ML predictions for interfaces...")
    interface_data = data_gen.extract_interface_data(theta_scaled, f_scaled, u_analytical_scaled, bc_dict)
    
    # Store both scaled and unscaled predictions
    vertical_predictions = {}  # (y_idx) -> scaled_value
    horizontal_predictions = {}  # (x_idx) -> scaled_value
    vertical_predictions_unscaled = {}  # (y_idx) -> unscaled_value
    horizontal_predictions_unscaled = {}  # (x_idx) -> unscaled_value
    
    for point in interface_data:
        x, y = point['position']
        pred_scaled = predictor.predict(point)
        pred_unscaled = unscale_value(pred_scaled, -1, 1, 
                                     scale_factors['solution']['min'],
                                     scale_factors['solution']['max'])
        
        if point['type'] == 'vertical':
            vertical_predictions[y] = pred_scaled
            vertical_predictions_unscaled[y] = pred_unscaled
        else:
            horizontal_predictions[x] = pred_scaled
            horizontal_predictions_unscaled[x] = pred_unscaled
    
    # Solve subdomains using unscaled predicted interface values
    print("Solving subdomains...")
    u_ml_interface = np.zeros_like(u_full)
    
    # Helper function to create interface BC function with array indices
    def create_interface_bc(predictions, subdomain_range, interface_type='vertical'):
        y_start, y_end, x_start, x_end = subdomain_range
        def bc_func(x):
            if isinstance(x, np.ndarray):
                # For array input, x is already array indices
                if interface_type == 'horizontal':
                    global_indices = x + x_start
                else:  # vertical
                    global_indices = x + y_start
                return np.array([predictions.get(int(i), 0) for i in global_indices])
            else:
                # For single input, x is array index
                if interface_type == 'horizontal':
                    global_idx = int(x) + x_start
                else:  # vertical
                    global_idx = int(x) + y_start
                return float(predictions.get(global_idx, 0))
        return bc_func
    
    # Solve bottom-left subdomain
    print("Solving bottom-left subdomain...")
    bc_right = create_interface_bc(
        {y: vertical_predictions_unscaled[y] for y in range(40)},
        (0, 20, 0, 20),
        'vertical'
    )
    bc_top = create_interface_bc(
        {x: horizontal_predictions_unscaled[x] for x in range(40)},
        (0, 20, 0, 20),
        'horizontal'
    )
    u_bl, _ = subdomain_solver.solve_subdomain(
        theta[:20, :20], f[:20, :20],
        {
            'left': lambda y: bc_dict['left'](y/20),
            'right': bc_right,
            'bottom': lambda x: bc_dict['bottom'](x/20),
            'top': bc_top
        }
    )
    u_ml_interface[:20, :20] = u_bl
    
    # Solve bottom-right subdomain
    print("Solving bottom-right subdomain...")
    bc_left = create_interface_bc(
        {y: vertical_predictions_unscaled[y] for y in range(40)},
        (0, 20, 20, 40),
        'vertical'
    )
    bc_top = create_interface_bc(
        {x: horizontal_predictions_unscaled[x] for x in range(40)},
        (0, 20, 20, 40),
        'horizontal'
    )
    u_br, _ = subdomain_solver.solve_subdomain(
        theta[:20, 20:], f[:20, 20:],
        {
            'left': bc_left,
            'right': lambda y: bc_dict['right'](y/20),
            'bottom': lambda x: bc_dict['bottom']((x+20)/40),
            'top': bc_top
        }
    )
    u_ml_interface[:20, 20:] = u_br
    
    # Solve top-left subdomain
    print("Solving top-left subdomain...")
    bc_right = create_interface_bc(
        {y: vertical_predictions_unscaled[y] for y in range(40)},
        (20, 40, 0, 20),
        'vertical'
    )
    bc_bottom = create_interface_bc(
        {x: horizontal_predictions_unscaled[x] for x in range(40)},
        (20, 40, 0, 20),
        'horizontal'
    )
    u_tl, _ = subdomain_solver.solve_subdomain(
        theta[20:, :20], f[20:, :20],
        {
            'left': lambda y: bc_dict['left']((y+20)/40),
            'right': bc_right,
            'bottom': bc_bottom,
            'top': lambda x: bc_dict['top'](x/20)
        }
    )
    u_ml_interface[20:, :20] = u_tl
    
    # Solve top-right subdomain
    print("Solving top-right subdomain...")
    bc_left = create_interface_bc(
        {y: vertical_predictions_unscaled[y] for y in range(40)},
        (20, 40, 20, 40),
        'vertical'
    )
    bc_bottom = create_interface_bc(
        {x: horizontal_predictions_unscaled[x] for x in range(40)},
        (20, 40, 20, 40),
        'horizontal'
    )
    u_tr, _ = subdomain_solver.solve_subdomain(
        theta[20:, 20:], f[20:, 20:],
        {
            'left': bc_left,
            'right': lambda y: bc_dict['right']((y+20)/40),
            'bottom': bc_bottom,
            'top': lambda x: bc_dict['top']((x+20)/40)
        }
    )
    u_ml_interface[20:, 20:] = u_tr
    
    # Scale solutions for comparison
    # First scale the full domain solution
    u_full_scaled = np.zeros_like(u_full)
    for i in range(u_full.shape[0]):
        for j in range(u_full.shape[1]):
            u_full_scaled[i,j] = unscale_value(u_full[i,j], 
                                             scale_factors['solution']['min'],
                                             scale_factors['solution']['max'],
                                             -1, 1)
    
    # Scale the ML interface solution
    u_ml_interface_scaled = np.zeros_like(u_ml_interface)
    for i in range(u_ml_interface.shape[0]):
        for j in range(u_ml_interface.shape[1]):
            u_ml_interface_scaled[i,j] = unscale_value(u_ml_interface[i,j],
                                                      scale_factors['solution']['min'],
                                                      scale_factors['solution']['max'],
                                                      -1, 1)
    
    # Verify interface values match predictions
    print("\nVerifying interface values...")
    vertical_interface_error = np.mean(np.abs(u_ml_interface_scaled[:, 20] - 
                                            [vertical_predictions[y] for y in range(40)]))
    horizontal_interface_error = np.mean(np.abs(u_ml_interface_scaled[20, :] - 
                                              [horizontal_predictions[x] for x in range(40)]))
    print(f"Mean vertical interface error: {vertical_interface_error:.6f}")
    print(f"Mean horizontal interface error: {horizontal_interface_error:.6f}")
    
    # Calculate errors
    print("\nCalculating errors...")
    
    def calculate_errors(solution, reference):
        diff = np.abs(solution - reference)
        return {
            'mae': np.mean(diff),
            'rmse': np.sqrt(np.mean(diff**2)),
            'max_error': np.max(diff),
            'rel_error': np.mean(diff / (np.abs(reference) + 1e-8))
        }
    
    full_errors = calculate_errors(u_full_scaled, u_analytical_scaled)
    ml_errors = calculate_errors(u_ml_interface_scaled, u_analytical_scaled)
    
    # Save results
    print("\nSaving results...")
    solutions = {
        'Analytical': u_analytical_scaled,
        'Full Domain': u_full_scaled,
        'ML Interface': u_ml_interface_scaled
    }
    
    # Store the scaled predictions for plotting
    interface_lines = {
        'vertical': {20: {y: vertical_predictions[y] for y in range(40)}},  # x=20 is the vertical interface
        'horizontal': {20: {x: horizontal_predictions[x] for x in range(40)}}  # y=20 is the horizontal interface
    }
    
    # Plot solution comparison with interface lines
    fig = plot_comparison(solutions, interface_lines)
    fig.savefig(os.path.join(results_dir, 'solution_comparison.png'))
    plt.close(fig)
    
    # Plot interface values
    plot_interface_values(solutions, interface_lines, results_dir)
    
    # Save error metrics
    with open(os.path.join(results_dir, 'error_metrics.txt'), 'w') as f:
        f.write("Full Domain Solution Errors:\n")
        f.write(f"Mean Absolute Error: {full_errors['mae']:.6f}\n")
        f.write(f"Root Mean Square Error: {full_errors['rmse']:.6f}\n")
        f.write(f"Maximum Error: {full_errors['max_error']:.6f}\n")
        f.write(f"Relative Error: {full_errors['rel_error']:.6f}\n\n")
        
        f.write("ML Interface Solution Errors:\n")
        f.write(f"Mean Absolute Error: {ml_errors['mae']:.6f}\n")
        f.write(f"Root Mean Square Error: {ml_errors['rmse']:.6f}\n")
        f.write(f"Maximum Error: {ml_errors['max_error']:.6f}\n")
        f.write(f"Relative Error: {ml_errors['rel_error']:.6f}\n")
    
    # Print summary
    print("\nResults Summary:")
    print("\nFull Domain Solution Errors:")
    print(f"Mean Absolute Error: {full_errors['mae']:.6f}")
    print(f"Root Mean Square Error: {full_errors['rmse']:.6f}")
    print(f"Maximum Error: {full_errors['max_error']:.6f}")
    print(f"Relative Error: {full_errors['rel_error']:.6f}")
    
    print("\nML Interface Solution Errors:")
    print(f"Mean Absolute Error: {ml_errors['mae']:.6f}")
    print(f"Root Mean Square Error: {ml_errors['rmse']:.6f}")
    print(f"Maximum Error: {ml_errors['max_error']:.6f}")
    print(f"Relative Error: {ml_errors['rel_error']:.6f}")
    
    print(f"\nResults have been saved to '{results_dir}' directory")

def plot_interface_values(solutions: dict, interface_predictions: dict, results_dir: str):
    """Plot the interface values along the vertical and horizontal lines."""
    # Create figure for interface value plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot vertical interface values
    y_coords = np.arange(40)
    vertical_x = 20  # x-coordinate of vertical interface
    
    # Get the predicted values for vertical interface
    vertical_pred_values = np.array([interface_predictions['vertical'][20].get(y, 0) for y in y_coords])
    
    ax1.plot(y_coords, solutions['Analytical'][:, vertical_x], 'b-', label='Analytical')
    ax1.plot(y_coords, vertical_pred_values, 'r--', label='ML Predicted')
    ax1.plot(y_coords, solutions['ML Interface'][:, vertical_x], 'g:', label='ML Solution')
    ax1.set_xlabel('Y coordinate')
    ax1.set_ylabel('Solution value')
    ax1.set_title('Values along Vertical Interface (x=20)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot horizontal interface values
    x_coords = np.arange(40)
    horizontal_y = 20  # y-coordinate of horizontal interface
    
    # Get the predicted values for horizontal interface
    horizontal_pred_values = np.array([interface_predictions['horizontal'][20].get(x, 0) for x in x_coords])
    
    ax2.plot(x_coords, solutions['Analytical'][horizontal_y, :], 'b-', label='Analytical')
    ax2.plot(x_coords, horizontal_pred_values, 'r--', label='ML Predicted')
    ax2.plot(x_coords, solutions['ML Interface'][horizontal_y, :], 'g:', label='ML Solution')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Solution value')
    ax2.set_title('Values along Horizontal Interface (y=20)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, 'interface_values.png'))
    plt.close(fig)

if __name__ == "__main__":
    compare_solutions() 