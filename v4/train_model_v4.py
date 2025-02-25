import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import time
from typing import Dict, List, Tuple
from data_generator_v4 import DataGeneratorV4
import json

class InterfaceDataset(Dataset):
    """Dataset class for interface prediction."""
    
    def __init__(self, data: List[Dict]):
        """
        Initialize the dataset.
        
        Args:
            data: List of dictionaries containing interface data
        """
        self.interface_data = data
    
    def __len__(self) -> int:
        return len(self.interface_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Returns dictionary with:
            - theta_patch: Local theta values
            - f_patch: Local f values
            - target: True interface value
        """
        data = self.interface_data[idx]
        
        # Convert numpy arrays to torch tensors
        theta_patch = torch.FloatTensor(data['theta_patch']).unsqueeze(0)  # Add channel dim
        f_patch = torch.FloatTensor(data['f_patch']).unsqueeze(0)
        target = torch.FloatTensor([data['target']])
        
        return {
            'theta_patch': theta_patch,
            'f_patch': f_patch,
            'target': target
        }

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

def generate_dataset(data_gen: DataGeneratorV4, n_samples: int) -> List[Dict]:
    """
    Generate a dataset of interface data.
    
    Args:
        data_gen: Data generator
        n_samples: Number of problems to generate
        
    Returns:
        List of interface data points
    """
    dataset = []
    
    print(f"Generating {n_samples} samples...")
    for i in range(n_samples):
        if i % 10 == 0:
            print(f"  Sample {i}/{n_samples}")
        
        # Generate random problem
        theta, f, u = data_gen.generate_random_problem()
        bc_dict = data_gen.generate_boundary_conditions(u)
        
        # Extract interface data
        interface_data = data_gen.extract_interface_data(theta, f, u, bc_dict)
        dataset.extend(interface_data)
    
    print(f"Generated {len(dataset)} interface data points from {n_samples} problems")
    return dataset

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
               device: torch.device, n_epochs: int = 100, lr: float = 0.001) -> Tuple[List[float], List[float]]:
    """
    Train the interface prediction model.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on (CPU/GPU)
        n_epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        Training and validation losses
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            theta_patch = batch['theta_patch'].to(device)
            f_patch = batch['f_patch'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            output = model(theta_patch, f_patch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                theta_patch = batch['theta_patch'].to(device)
                f_patch = batch['f_patch'].to(device)
                target = batch['target'].to(device)
                
                output = model(theta_patch, f_patch)
                loss = criterion(output, target)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return train_losses, val_losses

def save_dataset(dataset: List[Dict], filename: str):
    """Save dataset to file."""
    # Convert numpy arrays to lists for JSON serialization
    serializable_dataset = []
    for item in dataset:
        serializable_item = {
            'position': item['position'],
            'type': item['type'],
            'theta_patch': item['theta_patch'].tolist(),
            'f_patch': item['f_patch'].tolist(),
            'target': float(item['target'])
        }
        serializable_dataset.append(serializable_item)
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(serializable_dataset, f)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Initialize data generator
    print("Initializing data generator...")
    data_gen = DataGeneratorV4(nx=40, ny=40, n_subdomains=(2, 2), patch_size=5)
    
    # Generate datasets
    train_data = generate_dataset(data_gen, n_samples=800)  # 800 samples for training
    val_data = generate_dataset(data_gen, n_samples=200)    # 200 samples for validation
    
    # Save datasets
    print("Saving datasets...")
    save_dataset(train_data, 'data/train_data_v4.json')
    save_dataset(val_data, 'data/val_data_v4.json')
    
    # Save generator configuration
    config = {
        'nx': data_gen.nx,
        'ny': data_gen.ny,
        'n_subdomains': data_gen.n_subdomains,
        'patch_size': data_gen.patch_size,
        'k_values': data_gen.k_values.tolist(),
        'scale_factors': data_gen.scale_factors
    }
    with open('data/generator_config_v4.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create datasets
    train_dataset = InterfaceDataset(train_data)
    val_dataset = InterfaceDataset(val_data)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InterfacePredictor(patch_size=5).to(device)
    
    # Train model
    print(f"Training model on {device}...")
    start_time = time.time()
    train_losses, val_losses = train_model(model, train_loader, val_loader, device, n_epochs=100)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save model
    torch.save({
        'model_state': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'training_time': training_time
    }, 'models/interface_predictor_v4.pth')
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/training_history_v4.png')
    
    print("Model saved to models/interface_predictor_v4.pth")
    print("Training history saved to models/training_history_v4.png")

if __name__ == "__main__":
    main() 