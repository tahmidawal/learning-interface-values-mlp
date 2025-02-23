import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np

class InterfaceNet(nn.Module):
    """
    Neural network for predicting interface boundary conditions.
    Uses a CNN architecture to process local context around interfaces.
    """
    
    def __init__(self, context_size: int = 3, interface_size: int = 20):
        """
        Initialize the network.
        
        Args:
            context_size (int): Size of context window on each side of interface
            interface_size (int): Length of the interface
        """
        super().__init__()
        
        # Convolutional layers for processing context
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * context_size * interface_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, interface_size)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, context_size, interface_size)
            
        Returns:
            torch.Tensor: Predicted interface values
        """
        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class InterfacePredictor:
    """
    Class for training and using the InterfaceNet model.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the predictor.
        
        Args:
            device (str): Device to use for computations ('cuda' or 'cpu')
        """
        self.device = device
        # Initialize two separate models for vertical and horizontal interfaces
        self.vertical_model = InterfaceNet(context_size=6, interface_size=20).to(device)  # 20 is the height of subdomain
        self.horizontal_model = InterfaceNet(context_size=6, interface_size=20).to(device)  # 20 is the width of subdomain
        
        # Initialize optimizers
        self.vertical_optimizer = torch.optim.Adam(self.vertical_model.parameters())
        self.horizontal_optimizer = torch.optim.Adam(self.horizontal_model.parameters())
        self.criterion = nn.MSELoss()
    
    def prepare_batch(self, interface_data: List[Dict]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Prepare a batch of interface data for training, separating vertical and horizontal interfaces.
        
        Args:
            interface_data (List[Dict]): List of interface data dictionaries
            
        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: Input tensors and target values for both types
        """
        vertical_inputs = []
        vertical_targets = []
        horizontal_inputs = []
        horizontal_targets = []
        
        for data in interface_data:
            if data['type'] == 'vertical':
                # Combine left and right context for vertical interfaces
                context = np.concatenate([data['context_left'], data['context_right']], axis=1)
                context = context.reshape(1, 1, *context.shape)
                vertical_inputs.append(context)
                vertical_targets.append(data['interface_values'])
            else:  # horizontal
                # Combine top and bottom context for horizontal interfaces
                context = np.concatenate([data['context_bottom'], data['context_top']], axis=0)
                context = context.reshape(1, 1, *context.shape)
                horizontal_inputs.append(context)
                horizontal_targets.append(data['interface_values'])
        
        # Prepare vertical interface tensors if we have any
        if vertical_inputs:
            vertical_inputs = torch.FloatTensor(np.concatenate(vertical_inputs, axis=0)).to(self.device)
            vertical_targets = torch.FloatTensor(np.stack(vertical_targets)).to(self.device)
        else:
            vertical_inputs = None
            vertical_targets = None
        
        # Prepare horizontal interface tensors if we have any
        if horizontal_inputs:
            horizontal_inputs = torch.FloatTensor(np.concatenate(horizontal_inputs, axis=0)).to(self.device)
            horizontal_targets = torch.FloatTensor(np.stack(horizontal_targets)).to(self.device)
        else:
            horizontal_inputs = None
            horizontal_targets = None
        
        inputs = {
            'vertical': vertical_inputs,
            'horizontal': horizontal_inputs
        }
        targets = {
            'vertical': vertical_targets,
            'horizontal': horizontal_targets
        }
        
        return inputs, targets
    
    def train(self, dataset: List[Dict], n_epochs: int = 100, batch_size: int = 32):
        """
        Train the model on a dataset.
        
        Args:
            dataset (List[Dict]): List of dictionaries containing problem data
            n_epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        self.vertical_model.train()
        self.horizontal_model.train()
        
        # Flatten interface data from all problems
        all_interface_data = []
        for problem in dataset:
            all_interface_data.extend(problem['interface_data'])
        
        n_samples = len(all_interface_data)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(n_epochs):
            total_vertical_loss = 0.0
            total_horizontal_loss = 0.0
            n_vertical = 0
            n_horizontal = 0
            
            # Shuffle data
            np.random.shuffle(all_interface_data)
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch_data = all_interface_data[start_idx:end_idx]
                
                # Prepare batch
                inputs, targets = self.prepare_batch(batch_data)
                
                # Train vertical interfaces
                if inputs['vertical'] is not None:
                    self.vertical_optimizer.zero_grad()
                    outputs = self.vertical_model(inputs['vertical'])
                    loss = self.criterion(outputs, targets['vertical'])
                    loss.backward()
                    self.vertical_optimizer.step()
                    total_vertical_loss += loss.item()
                    n_vertical += 1
                
                # Train horizontal interfaces
                if inputs['horizontal'] is not None:
                    self.horizontal_optimizer.zero_grad()
                    outputs = self.horizontal_model(inputs['horizontal'])
                    loss = self.criterion(outputs, targets['horizontal'])
                    loss.backward()
                    self.horizontal_optimizer.step()
                    total_horizontal_loss += loss.item()
                    n_horizontal += 1
            
            # Calculate average losses
            avg_vertical_loss = total_vertical_loss / n_vertical if n_vertical > 0 else 0
            avg_horizontal_loss = total_horizontal_loss / n_horizontal if n_horizontal > 0 else 0
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}]')
                print(f'  Vertical Loss: {avg_vertical_loss:.6f}')
                print(f'  Horizontal Loss: {avg_horizontal_loss:.6f}')
    
    def predict(self, interface_data: Dict) -> np.ndarray:
        """
        Predict interface values for a single interface.
        
        Args:
            interface_data (Dict): Dictionary containing interface data
            
        Returns:
            np.ndarray: Predicted interface values
        """
        if interface_data['type'] == 'vertical':
            model = self.vertical_model
            context = np.concatenate([interface_data['context_left'], interface_data['context_right']], axis=1)
        else:  # horizontal
            model = self.horizontal_model
            context = np.concatenate([interface_data['context_bottom'], interface_data['context_top']], axis=0)
        
        model.eval()
        with torch.no_grad():
            context = context.reshape(1, 1, *context.shape)
            inputs = torch.FloatTensor(context).to(self.device)
            outputs = model(inputs)
            return outputs[0].cpu().numpy()
    
    def save_model(self, path_prefix: str):
        """Save both models to files."""
        torch.save({
            'model_state_dict': self.vertical_model.state_dict(),
            'optimizer_state_dict': self.vertical_optimizer.state_dict(),
        }, f'{path_prefix}_vertical_model.pth')
        
        torch.save({
            'model_state_dict': self.horizontal_model.state_dict(),
            'optimizer_state_dict': self.horizontal_optimizer.state_dict(),
        }, f'{path_prefix}_horizontal_model.pth')
    
    def load_model(self, path_prefix: str):
        """Load both models from files."""
        vertical_checkpoint = torch.load(f'{path_prefix}_vertical_model.pth', map_location=self.device)
        self.vertical_model.load_state_dict(vertical_checkpoint['model_state_dict'])
        self.vertical_optimizer.load_state_dict(vertical_checkpoint['optimizer_state_dict'])
        
        horizontal_checkpoint = torch.load(f'{path_prefix}_horizontal_model.pth', map_location=self.device)
        self.horizontal_model.load_state_dict(horizontal_checkpoint['model_state_dict'])
        self.horizontal_optimizer.load_state_dict(horizontal_checkpoint['optimizer_state_dict']) 