import numpy as np
import matplotlib.pyplot as plt
from data_generator_v2 import DataGeneratorV2
from ml_model_v2 import InterfacePredictorV2
import torch
from typing import Dict, List, Tuple
import seaborn as sns

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

def plot_interface_predictions(predictor: InterfacePredictorV2, test_data: List[Dict],
                             n_samples: int = 5):
    """Plot interface predictions for a few test cases."""
    for i in range(n_samples):
        problem = test_data[i]
        solution = problem['solution']
        
        # Separate vertical and horizontal interface points
        vertical_points = []
        horizontal_points = []
        
        for interface_point in problem['interface_data']:
            x, y = interface_point['position']
            pred = predictor.predict(interface_point)
            true = interface_point['target']
            
            if interface_point['type'] == 'vertical':
                vertical_points.append((y, true, pred))  # y-position, true value, predicted value
            else:
                horizontal_points.append((x, true, pred))  # x-position, true value, predicted value
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Plot vertical interfaces
        if vertical_points:
            ax1 = plt.subplot(221)
            # Sort by y-position
            vertical_points.sort(key=lambda x: x[0])
            y_pos, true_vals, pred_vals = zip(*vertical_points)
            
            ax1.plot(y_pos, true_vals, '-', label='True', linewidth=2)
            ax1.plot(y_pos, pred_vals, '--', label='Predicted', linewidth=2)
            ax1.set_xlabel('Y Position')
            ax1.set_ylabel('Interface Value')
            ax1.set_title('Vertical Interface Values')
            ax1.legend()
            ax1.grid(True)
        
        # Plot horizontal interfaces
        if horizontal_points:
            ax2 = plt.subplot(222)
            # Sort by x-position
            horizontal_points.sort(key=lambda x: x[0])
            x_pos, true_vals, pred_vals = zip(*horizontal_points)
            
            ax2.plot(x_pos, true_vals, '-', label='True', linewidth=2)
            ax2.plot(x_pos, pred_vals, '--', label='Predicted', linewidth=2)
            ax2.set_xlabel('X Position')
            ax2.set_ylabel('Interface Value')
            ax2.set_title('Horizontal Interface Values')
            ax2.legend()
            ax2.grid(True)
        
        # Error distribution
        all_errors = []
        if vertical_points:
            all_errors.extend([p - t for _, t, p in vertical_points])
        if horizontal_points:
            all_errors.extend([p - t for _, t, p in horizontal_points])
        
        ax3 = plt.subplot(223)
        sns.histplot(all_errors, kde=True, ax=ax3)
        ax3.set_title('Error Distribution')
        ax3.set_xlabel('Error')
        ax3.grid(True)
        
        # Scatter plot of predicted vs true values
        ax4 = plt.subplot(224)
        all_true = []
        all_pred = []
        if vertical_points:
            all_true.extend([t for _, t, _ in vertical_points])
            all_pred.extend([p for _, _, p in vertical_points])
        if horizontal_points:
            all_true.extend([t for _, t, _ in horizontal_points])
            all_pred.extend([p for _, _, p in horizontal_points])
        
        ax4.scatter(all_true, all_pred, alpha=0.6)
        min_val = min(min(all_true), min(all_pred))
        max_val = max(max(all_true), max(all_pred))
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
        ax4.set_xlabel('True Values')
        ax4.set_ylabel('Predicted Values')
        ax4.set_title('Predicted vs True Values')
        ax4.legend()
        ax4.grid(True)
        
        plt.suptitle(f'Interface Predictions Analysis (Sample {i+1})')
        plt.tight_layout()
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
    """Calculate various error metrics with better handling of relative errors."""
    abs_errors = np.abs(predictions - targets)
    
    # Handle relative errors more carefully
    rel_errors = abs_errors / (np.abs(targets) + 1e-6)  # Increased epsilon
    rel_errors = rel_errors[np.abs(targets) > 1e-6]  # Only consider non-tiny targets
    
    return {
        'mae': np.mean(abs_errors),
        'rmse': np.sqrt(np.mean(abs_errors**2)),
        'max_error': np.max(abs_errors),
        'rel_error_mean': np.mean(rel_errors),
        'rel_error_median': np.median(rel_errors),
        'rel_error_90th': np.percentile(rel_errors, 90),
        'n_samples': len(predictions)
    }

def analyze_by_interface_type(predictions: List[float], targets: List[float],
                            types: List[str]) -> Dict[str, Dict[str, float]]:
    """Analyze errors separately for vertical and horizontal interfaces."""
    vert_pred = [p for p, t in zip(predictions, types) if t == 'vertical']
    vert_targ = [t for t, typ in zip(targets, types) if typ == 'vertical']
    horz_pred = [p for p, t in zip(predictions, types) if t == 'horizontal']
    horz_targ = [t for t, typ in zip(targets, types) if typ == 'horizontal']
    
    return {
        'vertical': calculate_metrics(np.array(vert_pred), np.array(vert_targ)),
        'horizontal': calculate_metrics(np.array(horz_pred), np.array(horz_targ))
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
    """Test the trained model and generate visualizations."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create data generator and generate test data
    print("Generating test data...")
    data_gen = DataGeneratorV2(nx=40, ny=40, n_subdomains=(2, 2), patch_size=5)
    test_data = data_gen.generate_dataset(n_samples=20)
    
    # Load trained model
    print("Loading trained model...")
    predictor = InterfacePredictorV2(patch_size=5)
    predictor.load_model('models/interface_predictor_v2.pth')
    
    # Collect predictions and targets
    print("\nCalculating test metrics...")
    all_predictions = []
    all_targets = []
    all_types = []
    
    for problem in test_data:
        for interface_point in problem['interface_data']:
            pred = predictor.predict(interface_point)
            target = interface_point['target']
            all_predictions.append(pred)
            all_targets.append(target)
            all_types.append(interface_point['type'])
    
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(predictions, targets)
    type_metrics = analyze_by_interface_type(all_predictions, all_targets, all_types)
    
    # Print metrics
    print("\nOverall Test Metrics:")
    print(f"Number of test points: {overall_metrics['n_samples']}")
    print(f"Mean Absolute Error: {overall_metrics['mae']:.6f}")
    print(f"Root Mean Square Error: {overall_metrics['rmse']:.6f}")
    print(f"Maximum Error: {overall_metrics['max_error']:.6f}")
    print(f"Mean Relative Error: {overall_metrics['rel_error_mean']:.6f}")
    print(f"Median Relative Error: {overall_metrics['rel_error_median']:.6f}")
    print(f"90th Percentile Relative Error: {overall_metrics['rel_error_90th']:.6f}")
    
    print("\nVertical Interface Metrics:")
    print(f"Number of points: {type_metrics['vertical']['n_samples']}")
    print(f"Mean Absolute Error: {type_metrics['vertical']['mae']:.6f}")
    print(f"Root Mean Square Error: {type_metrics['vertical']['rmse']:.6f}")
    
    print("\nHorizontal Interface Metrics:")
    print(f"Number of points: {type_metrics['horizontal']['n_samples']}")
    print(f"Mean Absolute Error: {type_metrics['horizontal']['mae']:.6f}")
    print(f"Root Mean Square Error: {type_metrics['horizontal']['rmse']:.6f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Plot interface predictions
    print("\nPlotting interface predictions...")
    plot_interface_predictions(predictor, test_data, n_samples=3)
    
    # Plot error heatmaps
    print("\nPlotting error heatmaps...")
    plot_error_heatmap(predictor, test_data, n_samples=3)
    
    # Analyze error by position
    print("\nAnalyzing error distribution by position...")
    analyze_error_by_position(predictor, test_data)
    
    print("\nTesting complete!")

if __name__ == "__main__":
    test_model() 