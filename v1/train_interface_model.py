import numpy as np
from data_generator_v2 import DataGeneratorV2
from ml_model_v2 import InterfacePredictorV2
import torch
import matplotlib.pyplot as plt
from typing import Dict, List
import os
from test_interface_model import plot_interface_predictions

def split_dataset(data: List[Dict], train_ratio: float = 0.8,
                 val_ratio: float = 0.1) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset into train, validation, and test sets."""
    n_samples = len(data)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Shuffle data
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]
    
    return train_data, val_data, test_data

def evaluate_model(predictor: InterfacePredictorV2, test_data: List[Dict]) -> Dict[str, float]:
    """
    Evaluate model performance on test data.
    
    Returns dictionary with various error metrics.
    """
    all_predictions = []
    all_targets = []
    
    # Collect predictions and targets
    for problem in test_data:
        for interface_point in problem['interface_data']:
            pred = predictor.predict(interface_point)
            target = interface_point['target']
            all_predictions.append(pred)
            all_targets.append(target)
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # Calculate errors
    abs_errors = np.abs(predictions - targets)
    rel_errors = abs_errors / (np.abs(targets) + 1e-10)
    
    return {
        'mae': np.mean(abs_errors),
        'rmse': np.sqrt(np.mean(abs_errors**2)),
        'max_error': np.max(abs_errors),
        'mean_rel_error': np.mean(rel_errors),
        'median_rel_error': np.median(rel_errors)
    }

def plot_prediction_comparison(predictions: np.ndarray, targets: np.ndarray):
    """Plot predicted vs true values."""
    plt.figure(figsize=(10, 10))
    plt.scatter(targets, predictions, alpha=0.5)
    
    # Plot perfect prediction line
    min_val = min(predictions.min(), targets.min())
    max_val = max(predictions.max(), targets.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True Interface Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    """
    Main training function.
    All data is scaled between -1 and 1 for better training stability.
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create data generator with scaled data
    print("Initializing data generator...")
    data_gen = DataGeneratorV2(nx=40, ny=40, n_subdomains=(2, 2), patch_size=5)
    
    # Generate dataset with scaled values
    print("Generating scaled dataset...")
    n_samples = 2000
    dataset = data_gen.generate_dataset(n_samples)
    print(f"Generated {n_samples} samples with values scaled between -1 and 1")
    
    # Split dataset
    print("Splitting dataset...")
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    print(f"Dataset sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Initialize model
    print("Initializing model...")
    predictor = InterfacePredictorV2(patch_size=5)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train model with scaled data
    print("Training model...")
    predictor.train(
        train_data=train_data,
        val_data=val_data,
        n_epochs=200,
        batch_size=64,
        patience=15
    )
    
    # Save model and scaling factors
    print("Saving model and scaling information...")
    model_info = {
        'model_state': predictor.model.state_dict(),
        'scale_factors': data_gen.scale_factors
    }
    torch.save(model_info, 'models/interface_predictor_v2.pth')
    
    # Plot training history
    predictor.plot_training_history()
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    metrics = evaluate_model(predictor, test_data)
    
    print("\nTest Set Metrics (on scaled data):")
    print(f"Mean Absolute Error: {metrics['mae']:.6f}")
    print(f"Root Mean Square Error: {metrics['rmse']:.6f}")
    print(f"Maximum Error: {metrics['max_error']:.6f}")
    print(f"Mean Relative Error: {metrics['mean_rel_error']:.6f}")
    print(f"Median Relative Error: {metrics['median_rel_error']:.6f}")
    
    # Collect predictions for visualization
    all_predictions = []
    all_targets = []
    all_positions = []
    all_types = []
    
    for problem in test_data[:10]:  # Use first 10 test problems
        for interface_point in problem['interface_data']:
            pred = predictor.predict(interface_point)
            all_predictions.append(pred)
            all_targets.append(interface_point['target'])
            all_positions.append(interface_point['position'])
            all_types.append(interface_point['type'])
    
    # Plot predictions
    plot_interface_predictions(all_predictions, all_targets, all_positions,
                             all_types, data_gen.scale_factors)

if __name__ == "__main__":
    main() 