import numpy as np
from data_generator_v2 import DataGeneratorV2
from ml_model_v2 import InterfacePredictorV2
import torch
import matplotlib.pyplot as plt
from typing import Dict, List
import os

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
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create data generator
    data_gen = DataGeneratorV2(nx=40, ny=40, n_subdomains=(2, 2), patch_size=5)
    
    # Generate dataset
    print("Generating dataset...")
    n_samples = 1000  # Number of PDE problems to generate
    dataset = data_gen.generate_dataset(n_samples)
    
    # Split dataset
    print("Splitting dataset...")
    train_data, val_data, test_data = split_dataset(dataset)
    print(f"Dataset sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Initialize model
    print("Initializing model...")
    predictor = InterfacePredictorV2(patch_size=5)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train model
    print("Training model...")
    predictor.train(
        train_data=train_data,
        val_data=val_data,
        n_epochs=100,
        batch_size=32,
        patience=10
    )
    
    # Save model
    print("Saving model...")
    predictor.save_model('models/interface_predictor_v2.pth')
    
    # Plot training history
    predictor.plot_training_history()
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    metrics = evaluate_model(predictor, test_data)
    
    print("\nTest Set Metrics:")
    print(f"Mean Absolute Error: {metrics['mae']:.6f}")
    print(f"Root Mean Square Error: {metrics['rmse']:.6f}")
    print(f"Maximum Error: {metrics['max_error']:.6f}")
    print(f"Mean Relative Error: {metrics['mean_rel_error']:.6f}")
    print(f"Median Relative Error: {metrics['median_rel_error']:.6f}")
    
    # Plot some predictions
    all_predictions = []
    all_targets = []
    
    for problem in test_data[:10]:  # Use first 10 test problems
        for interface_point in problem['interface_data']:
            pred = predictor.predict(interface_point)
            target = interface_point['target']
            all_predictions.append(pred)
            all_targets.append(target)
    
    plot_prediction_comparison(
        np.array(all_predictions),
        np.array(all_targets)
    )

if __name__ == "__main__":
    main() 