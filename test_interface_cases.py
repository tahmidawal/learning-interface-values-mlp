import numpy as np
import matplotlib.pyplot as plt
from data_generator_v2 import DataGeneratorV2
from ml_model_v2 import InterfacePredictorV2
import torch
from typing import Dict, List, Tuple
import os

def generate_in_distribution_case(data_gen: DataGeneratorV2) -> Dict:
    """Generate a case that follows the training data distribution"""
    return data_gen.generate_dataset(1)[0]

def generate_out_of_distribution_case(data_gen: DataGeneratorV2) -> Dict:
    """
    Generate a case that's outside the training distribution.
    Uses higher frequencies and amplitudes than training data.
    """
    # Save original k_values
    original_k_values = data_gen.k_values.copy()
    
    # Use higher frequencies
    data_gen.k_values = np.arange(9.0, 12.0, 0.5)  # Higher frequencies
    
    # Generate random coefficients and k values for the solution
    n_terms = np.random.randint(3, 5)  # More terms than training
    coefficients = np.random.uniform(0.3, 0.5, n_terms)  # Larger coefficients
    k_indices = np.random.choice(len(data_gen.k_values), size=(n_terms, 2), replace=True)
    
    # Initialize solution and source term
    u = np.zeros((data_gen.ny, data_gen.nx))
    f = np.zeros((data_gen.ny, data_gen.nx))
    
    # Build solution and source term
    for i in range(n_terms):
        k1, k2 = data_gen.k_values[k_indices[i]]
        coef = coefficients[i]
        
        # Solution term
        u_term = coef * np.sin(k1 * 2 * np.pi * data_gen.X) * np.sin(k2 * 2 * np.pi * data_gen.Y)
        u += u_term
        
        # Source term
        f_term = -coef * np.sin(k1 * 2 * np.pi * data_gen.X) * np.sin(k2 * 2 * np.pi * data_gen.Y)
        f += f_term
    
    # Generate theta with larger variations
    theta = np.ones((data_gen.ny, data_gen.nx)) + 0.1 * np.random.randn(data_gen.ny, data_gen.nx)
    theta = np.clip(theta, 0.8, 1.2)  # Larger variations around 1
    
    # Scale the arrays
    u_scaled = data_gen.scale_array(u)
    f_scaled = data_gen.scale_array(f)
    theta_scaled = data_gen.scale_array(theta, data_gen.scale_factors['theta']['min'], 
                                      data_gen.scale_factors['theta']['max'])
    
    # Generate boundary conditions from scaled solution
    bc_dict = data_gen.generate_boundary_conditions(u_scaled)
    
    # Extract interface data
    interface_data = data_gen.extract_interface_data(theta_scaled, f_scaled, u_scaled, bc_dict)
    
    # Restore original k_values
    data_gen.k_values = original_k_values
    
    return {
        'theta': theta_scaled,
        'f': f_scaled,
        'solution': u_scaled,
        'interface_data': interface_data,
        'bc_dict': bc_dict,
        'scale_factors': data_gen.scale_factors
    }

def plot_field(field: np.ndarray, title: str, ax=None):
    """Plot a 2D field with colorbar"""
    if ax is None:
        plt.figure(figsize=(6, 5))
        ax = plt.gca()
    
    im = ax.imshow(field, cmap='viridis', origin='lower')
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

def plot_case_comparison(case: Dict, predicted_solution: np.ndarray, case_type: str):
    """Plot comparison between exact and predicted solutions"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot theta
    plot_field(case['theta'], 'θ (Diffusion Coefficient)', axes[0, 0])
    
    # Plot source term
    plot_field(case['f'], 'f (Source Term)', axes[0, 1])
    
    # Plot exact solution
    plot_field(case['solution'], 'u_exact (Exact Solution)', axes[0, 2])
    
    # Plot predicted solution
    plot_field(predicted_solution, 'u_predicted (Predicted Solution)', axes[1, 0])
    
    # Plot absolute error
    error = np.abs(predicted_solution - case['solution'])
    plot_field(error, 'Absolute Error', axes[1, 1])
    
    # Plot relative error
    rel_error = error / (np.abs(case['solution']) + 1e-10)
    plot_field(rel_error, 'Relative Error', axes[1, 2])
    
    plt.suptitle(f'{case_type} Case Analysis')
    plt.tight_layout()
    return fig

def calculate_metrics(predicted: np.ndarray, exact: np.ndarray) -> Dict[str, float]:
    """Calculate error metrics"""
    abs_error = np.abs(predicted - exact)
    rel_error = abs_error / (np.abs(exact) + 1e-10)
    
    return {
        'mae': np.mean(abs_error),
        'rmse': np.sqrt(np.mean(abs_error**2)),
        'max_error': np.max(abs_error),
        'mean_rel_error': np.mean(rel_error),
        'median_rel_error': np.median(rel_error),
        'max_rel_error': np.max(rel_error)
    }

def main():
    # Create results directory
    os.makedirs('test_results', exist_ok=True)
    
    # Initialize data generator and model
    print("Initializing data generator and loading model...")
    data_gen = DataGeneratorV2(nx=40, ny=40, n_subdomains=(2, 2), patch_size=5)
    
    model_info = torch.load('models/interface_predictor_v2.pth')
    predictor = InterfacePredictorV2(patch_size=5)
    predictor.model.load_state_dict(model_info['model_state'])
    
    # Generate and test cases
    n_cases = 5
    in_dist_metrics = []
    out_dist_metrics = []
    
    print(f"\nGenerating and testing {n_cases} cases for each distribution...")
    
    # Test in-distribution cases
    for i in range(n_cases):
        print(f"\nTesting in-distribution case {i+1}")
        case = generate_in_distribution_case(data_gen)
        
        # Get predictions for all interface points
        predicted_values = []
        positions = []
        for point in case['interface_data']:
            pred = predictor.predict(point)
            predicted_values.append(pred)
            positions.append(point['position'])
        
        # Reconstruct the solution using predicted interface values
        predicted_solution = case['solution'].copy()
        for pos, val in zip(positions, predicted_values):
            predicted_solution[pos[1], pos[0]] = val
        
        # Calculate metrics
        metrics = calculate_metrics(predicted_solution, case['solution'])
        in_dist_metrics.append(metrics)
        
        # Plot and save results
        fig = plot_case_comparison(case, predicted_solution, f'In-Distribution Case {i+1}')
        plt.savefig(f'test_results/in_dist_case_{i+1}.png')
        plt.close(fig)
    
    # Test out-of-distribution cases
    for i in range(n_cases):
        print(f"\nTesting out-of-distribution case {i+1}")
        case = generate_out_of_distribution_case(data_gen)
        
        # Get predictions for all interface points
        predicted_values = []
        positions = []
        for point in case['interface_data']:
            pred = predictor.predict(point)
            predicted_values.append(pred)
            positions.append(point['position'])
        
        # Reconstruct the solution using predicted interface values
        predicted_solution = case['solution'].copy()
        for pos, val in zip(positions, predicted_values):
            predicted_solution[pos[1], pos[0]] = val
        
        # Calculate metrics
        metrics = calculate_metrics(predicted_solution, case['solution'])
        out_dist_metrics.append(metrics)
        
        # Plot and save results
        fig = plot_case_comparison(case, predicted_solution, f'Out-of-Distribution Case {i+1}')
        plt.savefig(f'test_results/out_dist_case_{i+1}.png')
        plt.close(fig)
    
    # Print average metrics
    print("\nAverage Metrics for In-Distribution Cases:")
    avg_metrics = {k: np.mean([m[k] for m in in_dist_metrics]) for k in in_dist_metrics[0].keys()}
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.6f}")
    
    print("\nAverage Metrics for Out-of-Distribution Cases:")
    avg_metrics = {k: np.mean([m[k] for m in out_dist_metrics]) for k in out_dist_metrics[0].keys()}
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.6f}")

if __name__ == "__main__":
    main() 