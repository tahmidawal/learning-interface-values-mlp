import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

class InterfaceDataset(Dataset):
    """Dataset class for interface prediction."""
    
    def __init__(self, data: List[Dict], patch_size: int = 3):
        """
        Initialize the dataset.
        
        Args:
            data: List of dictionaries containing problem data
            patch_size: Size of local context patch
        """
        self.interface_data = []
        self.patch_size = patch_size
        
        # Flatten and process all interface data
        for problem in data:
            self.interface_data.extend(problem['interface_data'])
    
    def __len__(self) -> int:
        return len(self.interface_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.
        
        Returns dictionary with:
            - theta_patch: Local theta values
            - f_patch: Local f values
            - boundary_values: Global boundary values
            - target: True interface value
            - position: Interface position
            - type: Interface type (vertical/horizontal)
        """
        data = self.interface_data[idx]
        
        # Convert numpy arrays to torch tensors
        theta_patch = torch.FloatTensor(data['theta_patch']).unsqueeze(0)  # Add channel dim
        f_patch = torch.FloatTensor(data['f_patch']).unsqueeze(0)
        boundary_values = torch.FloatTensor(data['boundary_values'])
        target = torch.FloatTensor([data['target']])
        
        return {
            'theta_patch': theta_patch,
            'f_patch': f_patch,
            'boundary_values': boundary_values,
            'target': target,
            'position': data['position'],
            'type': data['type']
        }

class InterfaceNetV2(nn.Module):
    """
    Enhanced neural network for predicting interface values.
    Uses a CNN to process local context and combines with global boundary information.
    All input and output values are expected to be scaled between -1 and 1.
    """
    
    def __init__(self, patch_size: int = 3, n_boundary_values: int = 4):
        """
        Initialize the network.
        
        Args:
            patch_size: Size of input patches
            n_boundary_values: Number of boundary values provided
        """
        super().__init__()
        
        # CNN for processing theta patches
        self.theta_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Added batch normalization
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Added batch normalization
            nn.ReLU()
        )
        
        # CNN for processing f patches
        self.f_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Added batch normalization
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Added batch normalization
            nn.ReLU()
        )
        
        # Calculate flattened size after convolutions
        self.flat_size = 64 * patch_size * patch_size
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(2 * self.flat_size + n_boundary_values, 256),
            nn.BatchNorm1d(256),  # Added batch normalization
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # Added batch normalization
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),   # Added batch normalization
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Added tanh activation to ensure output is between -1 and 1
        )
    
    def forward(self, theta_patch: torch.Tensor, f_patch: torch.Tensor,
                boundary_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        All inputs are expected to be scaled between -1 and 1.
        Output will be scaled between -1 and 1.
        
        Args:
            theta_patch: Patch of theta values (B, 1, H, W), scaled between -1 and 1
            f_patch: Patch of f values (B, 1, H, W), scaled between -1 and 1
            boundary_values: Global boundary values (B, N), scaled between -1 and 1
            
        Returns:
            torch.Tensor: Predicted interface values (B, 1), scaled between -1 and 1
        """
        # Process patches through CNNs
        theta_features = self.theta_conv(theta_patch)
        f_features = self.f_conv(f_patch)
        
        # Flatten CNN outputs
        theta_flat = theta_features.view(-1, self.flat_size)
        f_flat = f_features.view(-1, self.flat_size)
        
        # Concatenate all features
        combined = torch.cat([theta_flat, f_flat, boundary_values], dim=1)
        
        # Final prediction (will be between -1 and 1 due to tanh activation)
        return self.fc(combined)

class InterfacePredictorV2:
    """
    Enhanced predictor class for training and using the InterfaceNetV2 model.
    Handles scaled data between -1 and 1.
    """
    
    def __init__(self, patch_size: int = 3, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the predictor.
        
        Args:
            patch_size: Size of context patches
            device: Device to use for computations
        """
        self.device = device
        self.patch_size = patch_size
        self.model = InterfaceNetV2(patch_size=patch_size).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def train(self, train_data: List[Dict], val_data: Optional[List[Dict]] = None,
             n_epochs: int = 100, batch_size: int = 32, patience: int = 10):
        """
        Train the model on scaled data.
        
        Args:
            train_data: Training dataset (with scaled values)
            val_data: Validation dataset (with scaled values)
            n_epochs: Number of training epochs
            batch_size: Batch size
            patience: Early stopping patience
        """
        # Create datasets
        train_dataset = InterfaceDataset(train_data, self.patch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_data is not None:
            val_dataset = InterfaceDataset(val_data, self.patch_size)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training
            self.model.train()
            total_loss = 0.0
            n_batches = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}'):
                self.optimizer.zero_grad()
                
                # Move data to device
                theta_patch = batch['theta_patch'].to(self.device)
                f_patch = batch['f_patch'].to(self.device)
                boundary_values = batch['boundary_values'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass
                outputs = self.model(theta_patch, f_patch, boundary_values)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = total_loss / n_batches
            self.train_losses.append(avg_train_loss)
            
            # Validation
            if val_data is not None:
                val_loss = self.evaluate(val_loader)
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)  # Update learning rate based on validation loss
                
                print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, '
                      f'Val Loss = {val_loss:.6f}')
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'Early stopping after {epoch+1} epochs')
                        break
            else:
                print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}')
    
    def evaluate(self, data_loader: DataLoader) -> float:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            
        Returns:
            float: Average loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Move data to device
                theta_patch = batch['theta_patch'].to(self.device)
                f_patch = batch['f_patch'].to(self.device)
                boundary_values = batch['boundary_values'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass
                outputs = self.model(theta_patch, f_patch, boundary_values)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def predict(self, interface_data: Dict) -> np.ndarray:
        """
        Predict interface values for a single interface point.
        
        Args:
            interface_data: Dictionary containing interface data
            
        Returns:
            np.ndarray: Predicted interface value
        """
        self.model.eval()
        
        with torch.no_grad():
            # Prepare input data
            theta_patch = torch.FloatTensor(interface_data['theta_patch']).unsqueeze(0).unsqueeze(0)
            f_patch = torch.FloatTensor(interface_data['f_patch']).unsqueeze(0).unsqueeze(0)
            boundary_values = torch.FloatTensor(interface_data['boundary_values']).unsqueeze(0)
            
            # Move to device
            theta_patch = theta_patch.to(self.device)
            f_patch = f_patch.to(self.device)
            boundary_values = boundary_values.to(self.device)
            
            # Forward pass
            output = self.model(theta_patch, f_patch, boundary_values)
            
            return output.cpu().numpy().squeeze()
    
    def plot_training_history(self):
        """Plot training and validation loss history."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        plt.show()
    
    def save_model(self, path: str):
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
    
    def load_model(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses'] 