import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from data_generator_v3 import DataGeneratorV3
import os
import json
from typing import Dict, List, Tuple
import time

class InterfaceDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'theta_patch': torch.tensor(item['theta_patch'], dtype=torch.float32),
            'f_patch': torch.tensor(item['f_patch'], dtype=torch.float32),
            'target': torch.tensor(item['target'], dtype=torch.float32)
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

def train_model(model: nn.Module, 
                train_loader: DataLoader,
                val_loader: DataLoader,
                n_epochs: int,
                learning_rate: float,
                device: torch.device) -> Tuple[List[float], List[float]]:
    """Train the model and return training history."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                   patience=5, verbose=True)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            theta_patch = batch['theta_patch'].to(device)
            f_patch = batch['f_patch'].to(device)
            target = batch['target'].to(device).view(-1, 1)  # Ensure correct shape
            
            optimizer.zero_grad()
            output = model(theta_patch, f_patch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                theta_patch = batch['theta_patch'].to(device)
                f_patch = batch['f_patch'].to(device)
                target = batch['target'].to(device).view(-1, 1)  # Ensure correct shape
                
                output = model(theta_patch, f_patch)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print("--------------------")
    
    return train_losses, val_losses

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize data generator
    print("Initializing data generator...")
    data_gen = DataGeneratorV3(nx=40, ny=40, n_subdomains=(2, 2), patch_size=5)
    
    # Generate dataset
    print("\nGenerating training dataset...")
    train_data = []
    for i in range(100):  # 100 training samples
        if i % 10 == 0:
            print(f"Generating sample {i+1}/100")
        theta, f, u = data_gen.generate_random_problem()
        bc_dict = data_gen.generate_boundary_conditions(u)
        interface_data = data_gen.extract_interface_data(theta, f, u, bc_dict)
        train_data.extend(interface_data)
    
    print("\nGenerating test dataset...")
    test_data = []
    for i in range(20):  # 20 test samples
        if i % 5 == 0:
            print(f"Generating sample {i+1}/20")
        theta, f, u = data_gen.generate_random_problem()
        bc_dict = data_gen.generate_boundary_conditions(u)
        interface_data = data_gen.extract_interface_data(theta, f, u, bc_dict)
        test_data.extend(interface_data)
    
    # Split training data into train and validation
    train_size = int(0.8 * len(train_data))
    train_dataset = train_data[:train_size]
    val_dataset = train_data[train_size:]
    
    print(f"\nDataset sizes:")
    print(f"Training: {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")
    print(f"Test: {len(test_data)}")
    
    # Create data loaders
    train_loader = DataLoader(InterfaceDataset(train_dataset), batch_size=32, shuffle=True)
    val_loader = DataLoader(InterfaceDataset(val_dataset), batch_size=32)
    test_loader = DataLoader(InterfaceDataset(test_data), batch_size=32)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InterfacePredictor(patch_size=5).to(device)
    
    # Train model
    print("\nTraining model...")
    start_time = time.time()
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=100,
        learning_rate=0.001,
        device=device
    )
    training_time = time.time() - start_time
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            theta_patch = batch['theta_patch'].to(device)
            f_patch = batch['f_patch'].to(device)
            target = batch['target'].to(device)
            
            output = model(theta_patch, f_patch)
            loss = criterion(output, target)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('models/training_history_v3.png')
    plt.close()
    
    # Save model and training info
    print("\nSaving model and training info...")
    model_info = {
        'model_state': model.state_dict(),
        'config': {
            'patch_size': 5,
            'k_values': data_gen.k_values.tolist(),
            'scale_factors': data_gen.scale_factors
        },
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_loss': test_loss,
            'training_time': training_time
        }
    }
    torch.save(model_info, 'models/interface_predictor_v3.pth')
    
    # Print final statistics
    print("\nTraining complete!")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    print(f"Test loss: {test_loss:.6f}")

if __name__ == "__main__":
    main() 