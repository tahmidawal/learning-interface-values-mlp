import numpy as np
import torch
from data_generator_v3 import DataGeneratorV3
import os
from typing import Dict, List
import json

def generate_dataset(data_gen: DataGeneratorV3, n_samples: int) -> List[Dict]:
    """Generate a dataset with n_samples."""
    dataset = []
    
    for i in range(n_samples):
        if i % 10 == 0:
            print(f"Generating sample {i+1}/{n_samples}")
            
        # Generate random problem
        theta, f, u = data_gen.generate_random_problem()
        bc_dict = data_gen.generate_boundary_conditions(u)
        
        # Extract interface data
        interface_data = data_gen.extract_interface_data(theta, f, u, bc_dict)
        dataset.extend(interface_data)
    
    return dataset

def save_dataset(dataset: List[Dict], filename: str):
    """Save dataset to file."""
    # Convert numpy arrays to tensors and save
    processed_data = []
    for item in dataset:
        processed_item = {
            'position': item['position'],
            'type': item['type'],
            'theta_patch': torch.FloatTensor(item['theta_patch']),
            'f_patch': torch.FloatTensor(item['f_patch']),
            'target': torch.FloatTensor([item['target']])
        }
        processed_data.append(processed_item)
    
    torch.save(processed_data, filename)

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Initialize data generator
    print("Initializing data generator...")
    data_gen = DataGeneratorV3(nx=40, ny=40, n_subdomains=(2, 2), patch_size=5)
    
    # Generate training dataset (100 samples)
    print("\nGenerating training dataset...")
    train_dataset = generate_dataset(data_gen, n_samples=100)
    
    # Generate test dataset (20 samples)
    print("\nGenerating test dataset...")
    test_dataset = generate_dataset(data_gen, n_samples=20)
    
    # Save datasets
    print("\nSaving datasets...")
    save_dataset(train_dataset, 'data/train_dataset_v3.pth')
    save_dataset(test_dataset, 'data/test_dataset_v3.pth')
    
    # Save data generator configuration
    config = {
        'nx': data_gen.nx,
        'ny': data_gen.ny,
        'n_subdomains': data_gen.n_subdomains,
        'patch_size': data_gen.patch_size,
        'k_values': data_gen.k_values.tolist(),
        'scale_factors': data_gen.scale_factors
    }
    
    with open('data/generator_config_v3.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Print statistics
    print("\nDataset statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"k values used: {data_gen.k_values}")
    print("\nDone!")

if __name__ == "__main__":
    main() 