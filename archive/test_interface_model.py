import numpy as np
import matplotlib.pyplot as plt
from data_generator_v2 import DataGeneratorV2
from ml_model_v2 import InterfacePredictorV2
import torch
from typing import Dict, List, Tuple
import seaborn as sns
import os

def unscale_value(value: float, scale_min: float, scale_max: float,
                  target_min: float, target_max: float) -> float:
    """
    Unscale a value from the scaled range back to the original range.
    
    Args:
        value: Scaled value between scale_min and scale_max
        scale_min: Minimum of the scaled range
        scale_max: Maximum of the scaled range
        target_min: Minimum of the original range
        target_max: Maximum of the original range
    
    Returns:
        float: Unscaled value in the original range
    """
    return (value - scale_min) * (target_max - target_min) / (scale_max - scale_min) + target_min

def plot_solution_comparison(true_solution: np.ndarray, predicted_solution: np.ndarray,
                           title: str = "Solution Comparison"):
    """Plot true vs predicted solution."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot true solution
    im1 = ax1.imshow(true_solution, cmap='viridis')
    ax1.set_title('True Solution')
    plt.colorbar(im1, ax=ax1)
    
    # Plot predicted solution
    im2 = ax2.imshow(predicted_solution, cmap='viridis')
    ax2.set_title('Predicted Solution')
    plt.colorbar(im2, ax=ax2)
    
    # Plot difference
    diff = np.abs(true_solution - predicted_solution)
    im3 = ax3.imshow(diff, cmap='viridis')
    ax3.set_title('Absolute Difference')
    plt.colorbar(im3, ax=ax3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_interface_predictions(predictions: List[float], targets: List[float],
                             positions: List[Tuple[int, int]], interface_types: List[str],
                             scale_factors: Dict, save_path: str = None) -> None:
    """
    Plot interface predictions as continuous lines.
    Data is unscaled for visualization.
    
    Args:
        predictions: List of predicted values
        targets: List of target values
        positions: List of interface positions
        interface_types: List of interface types
        scale_factors: Dictionary of scaling factors
        save_path: Path to save the plot (if None, display instead)
    """
    # Separate vertical and horizontal interface points
    vertical_points = [(pos, pred, target) for pos, pred, target, itype 
                      in zip(positions, predictions, targets, interface_types)
                      if itype == 'vertical']
    horizontal_points = [(pos, pred, target) for pos, pred, target, itype 
                        in zip(positions, predictions, targets, interface_types)
                        if itype == 'horizontal']
    
    # Sort points by position for continuous lines
    vertical_points.sort(key=lambda x: x[0][1])  # Sort by y-coordinate
    horizontal_points.sort(key=lambda x: x[0][0])  # Sort by x-coordinate
    
    # Unscale values for visualization
    solution_scale = scale_factors['solution']
    unscale_sol = lambda x: unscale_value(x, -1, 1, solution_scale['min'], solution_scale['max'])
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot vertical interfaces
    if vertical_points:
        positions_v, preds_v, targets_v = zip(*vertical_points)
        y_coords = [pos[1]/40 for pos in positions_v]  # Normalize to [0,1]
        ax1.plot(y_coords, [unscale_sol(t) for t in targets_v], 'b-', label='True')
        ax1.plot(y_coords, [unscale_sol(p) for p in preds_v], 'r--', label='Predicted')
        ax1.set_xlabel('Normalized Y Position')
        ax1.set_ylabel('Interface Value')
        ax1.set_title('Vertical Interface Values')
        ax1.grid(True)
        ax1.legend()
    
    # Plot horizontal interfaces
    if horizontal_points:
        positions_h, preds_h, targets_h = zip(*horizontal_points)
        x_coords = [pos[0]/40 for pos in positions_h]  # Normalize to [0,1]
        ax2.plot(x_coords, [unscale_sol(t) for t in targets_h], 'b-', label='True')
        ax2.plot(x_coords, [unscale_sol(p) for p in preds_h], 'r--', label='Predicted')
        ax2.set_xlabel('Normalized X Position')
        ax2.set_ylabel('Interface Value')
        ax2.set_title('Horizontal Interface Values')
        ax2.grid(True)
        ax2.legend()
    
    # Plot error distribution
    errors = np.array(predictions) - np.array(targets)
    sns.histplot(errors, kde=True, ax=ax3)
    ax3.set_xlabel('Prediction Error (Scaled)')
    ax3.set_ylabel('Count')
    ax3.set_title('Error Distribution')
    
    # Plot predicted vs true values
    ax4.scatter(targets, predictions, alpha=0.5)
    ax4.plot([-1, 1], [-1, 1], 'r--')  # Perfect prediction line
    ax4.set_xlabel('True Values (Scaled)')
    ax4.set_ylabel('Predicted Values (Scaled)')
    ax4.set_title('Predicted vs True Values')
    ax4.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def analyze_error_by_position(predictor: InterfacePredictorV2, test_data: List[Dict]):
    """Analyze prediction error based on interface position."""
    vertical_errors = []
    horizontal_errors = []
    
    for problem in test_data:
        for interface_point in problem['interface_data']:
            pred = predictor.predict(interface_point)
            true = interface_point['target']
            error = np.abs(pred - true)
            
            if interface_point['type'] == 'vertical':
                vertical_errors.append((interface_point['position'], error))
            else:
                horizontal_errors.append((interface_point['position'], error))
    
    # Plot error distribution for vertical interfaces
    if vertical_errors:
        positions, errors = zip(*vertical_errors)
        y_positions = [pos[1] for pos in positions]
        
        plt.figure(figsize=(10, 5))
        plt.scatter(y_positions, errors, alpha=0.6)
        plt.xlabel('Y Position')
        plt.ylabel('Absolute Error')
        plt.title('Error Distribution Along Vertical Interfaces')
        plt.show()
    
    # Plot error distribution for horizontal interfaces
    if horizontal_errors:
        positions, errors = zip(*horizontal_errors)
        x_positions = [pos[0] for pos in positions]
        
        plt.figure(figsize=(10, 5))
        plt.scatter(x_positions, errors, alpha=0.6)
        plt.xlabel('X Position')
        plt.ylabel('Absolute Error')
        plt.title('Error Distribution Along Horizontal Interfaces')
        plt.show()

def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Calculate error metrics for scaled predictions."""
    abs_errors = np.abs(predictions - targets)
    squared_errors = (predictions - targets) ** 2
    
    metrics = {
        'mae': np.mean(abs_errors),
        'rmse': np.sqrt(np.mean(squared_errors)),
        'max_error': np.max(abs_errors),
        'mean_rel_error': np.mean(abs_errors / (np.abs(targets) + 1e-8)),
        'median_rel_error': np.median(abs_errors / (np.abs(targets) + 1e-8)),
        '90th_percentile_rel_error': np.percentile(abs_errors / (np.abs(targets) + 1e-8), 90)
    }
    return metrics

def analyze_by_interface_type(predictions: List[float], targets: List[float],
                            positions: List[Tuple[int, int]], interface_types: List[str]) -> Dict:
    """Analyze predictions separately for vertical and horizontal interfaces."""
    vertical_pred = []
    vertical_target = []
    horizontal_pred = []
    horizontal_target = []
    
    for pred, target, pos, itype in zip(predictions, targets, positions, interface_types):
        if itype == 'vertical':
            vertical_pred.append(pred)
            vertical_target.append(target)
        else:
            horizontal_pred.append(pred)
            horizontal_target.append(target)
    
    return {
        'vertical': calculate_metrics(np.array(vertical_pred), np.array(vertical_target)),
        'horizontal': calculate_metrics(np.array(horizontal_pred), np.array(horizontal_target))
    }

def plot_error_heatmap(predictor: InterfacePredictorV2, test_data: List[Dict],
                      n_samples: int = 5):
    """Plot error heatmap for interface predictions."""
    for i in range(n_samples):
        problem = test_data[i]
        solution = problem['solution']
        
        # Initialize error arrays
        errors = np.zeros_like(solution)
        mask = np.zeros_like(solution, dtype=bool)
        
        # Get predictions and calculate errors
        for interface_point in problem['interface_data']:
            x, y = interface_point['position']
            pred = predictor.predict(interface_point)
            true = interface_point['target']
            errors[y, x] = np.abs(pred - true)
            mask[y, x] = True
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(np.ma.masked_array(errors, ~mask), cmap='viridis')
        plt.colorbar(label='Absolute Error')
        plt.title(f'Error Heatmap (Sample {i+1})')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.show()

def test_model():
    """Test the trained model on new data."""
    # Create results directory
    results_dir = 'test_results'
    os.makedirs(results_dir, exist_ok=True)
    
    print("Generating test data...")
    data_gen = DataGeneratorV2(nx=40, ny=40, n_subdomains=(2, 2), patch_size=5)
    n_test_cases = 50  # Generate 50 test problems
    test_data = data_gen.generate_dataset(n_test_cases)
    
    print("Loading trained model...")
    model_info = torch.load('models/interface_predictor_v2.pth')
    predictor = InterfacePredictorV2(patch_size=5)
    predictor.model.load_state_dict(model_info['model_state'])
    scale_factors = model_info['scale_factors']
    
    print("Calculating test metrics...")
    all_predictions = []
    all_targets = []
    all_positions = []
    all_types = []
    
    # Process all test cases
    for i, problem in enumerate(test_data):
        problem_predictions = []
        problem_targets = []
        problem_positions = []
        problem_types = []
        
        for interface_point in problem['interface_data']:
            pred = predictor.predict(interface_point)
            problem_predictions.append(pred)
            problem_targets.append(interface_point['target'])
            problem_positions.append(interface_point['position'])
            problem_types.append(interface_point['type'])
        
        # Save individual problem results
        if i < 10:  # Save detailed plots for first 10 problems
            save_path = os.path.join(results_dir, f'problem_{i+1}_results.png')
            plot_interface_predictions(
                problem_predictions, problem_targets,
                problem_positions, problem_types,
                scale_factors, save_path
            )
        
        # Accumulate all results
        all_predictions.extend(problem_predictions)
        all_targets.extend(problem_targets)
        all_positions.extend(problem_positions)
        all_types.extend(problem_types)
    
    # Calculate overall metrics
    metrics = calculate_metrics(np.array(all_predictions), np.array(all_targets))
    
    # Save metrics to file
    metrics_path = os.path.join(results_dir, 'test_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Overall Test Metrics:\n")
        f.write(f"Number of test points: {len(all_predictions)}\n")
        f.write(f"Mean Absolute Error: {metrics['mae']:.6f}\n")
        f.write(f"Root Mean Square Error: {metrics['rmse']:.6f}\n")
        f.write(f"Maximum Error: {metrics['max_error']:.6f}\n")
        f.write(f"Mean Relative Error: {metrics['mean_rel_error']:.6f}\n")
        f.write(f"Median Relative Error: {metrics['median_rel_error']:.6f}\n")
        f.write(f"90th Percentile Relative Error: {metrics['90th_percentile_rel_error']:.6f}\n\n")
        
        # Add interface type metrics
        type_metrics = analyze_by_interface_type(all_predictions, all_targets,
                                               all_positions, all_types)
        
        f.write("\nVertical Interface Metrics:\n")
        f.write(f"Number of points: {len([t for t in all_types if t == 'vertical'])}\n")
        f.write(f"Mean Absolute Error: {type_metrics['vertical']['mae']:.6f}\n")
        f.write(f"Root Mean Square Error: {type_metrics['vertical']['rmse']:.6f}\n")
        
        f.write("\nHorizontal Interface Metrics:\n")
        f.write(f"Number of points: {len([t for t in all_types if t == 'horizontal'])}\n")
        f.write(f"Mean Absolute Error: {type_metrics['horizontal']['mae']:.6f}\n")
        f.write(f"Root Mean Square Error: {type_metrics['horizontal']['rmse']:.6f}\n")
    
    # Save overall visualization
    plot_interface_predictions(all_predictions, all_targets, all_positions,
                             all_types, scale_factors,
                             os.path.join(results_dir, 'overall_results.png'))
    
    print(f"Testing complete. Results have been saved to '{results_dir}' directory")
    
    # Print summary to console
    print("\nOverall Test Metrics:")
    print(f"Number of test points: {len(all_predictions)}")
    print(f"Mean Absolute Error: {metrics['mae']:.6f}")
    print(f"Root Mean Square Error: {metrics['rmse']:.6f}")
    print(f"Maximum Error: {metrics['max_error']:.6f}")
    print(f"Mean Relative Error: {metrics['mean_rel_error']:.6f}")
    print(f"Median Relative Error: {metrics['median_rel_error']:.6f}")
    print(f"90th Percentile Relative Error: {metrics['90th_percentile_rel_error']:.6f}")

if __name__ == "__main__":
    test_model() 