import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm
import time
import os
import json

from pde_solver import PoissonSolver
from data_generator import DataGenerator
from ml_model import InterfacePredictor

class Experiment:
    """
    Class for running experiments with ML-predicted interface boundary conditions.
    """
    
    def __init__(self, nx: int = 40, ny: int = 40, n_subdomains: Tuple[int, int] = (2, 2)):
        """
        Initialize the experiment.
        
        Args:
            nx (int): Number of points in x direction
            ny (int): Number of points in y direction
            n_subdomains (Tuple[int, int]): Number of subdomains in (x, y) directions
        """
        self.nx = nx
        self.ny = ny
        self.n_subdomains = n_subdomains
        
        # Initialize components
        self.data_generator = DataGenerator(nx, ny, n_subdomains)
        self.predictor = InterfacePredictor()
        
        # Initialize full domain solver
        self.full_solver = PoissonSolver(nx, ny)
        
        # Initialize subdomain solvers
        self.subdomain_nx = nx // n_subdomains[0]
        self.subdomain_ny = ny // n_subdomains[1]
        self.subdomain_solvers = [
            [PoissonSolver(self.subdomain_nx, self.subdomain_ny) 
             for _ in range(n_subdomains[0])]
            for _ in range(n_subdomains[1])
        ]
    
    def save_problem_visualization(self, sample_idx: int, theta: np.ndarray, f: np.ndarray, 
                                 solution: np.ndarray, interface_data: List[Dict]):
        """
        Save visualization of a generated problem.
        
        Args:
            sample_idx (int): Index of the sample
            theta (np.ndarray): Diffusion coefficient field
            f (np.ndarray): Source term
            solution (np.ndarray): Solution field
            interface_data (List[Dict]): Interface data
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot diffusion coefficient (theta)
        im1 = ax1.imshow(theta)
        ax1.set_title('Diffusion Coefficient (θ)')
        plt.colorbar(im1, ax=ax1)
        
        # Plot source term (f)
        im2 = ax2.imshow(f)
        ax2.set_title('Source Term (f)')
        plt.colorbar(im2, ax=ax2)
        
        # Plot solution
        im3 = ax3.imshow(solution)
        ax3.set_title('Solution (u)')
        plt.colorbar(im3, ax=ax3)
        
        # Plot interface locations
        ax4.imshow(solution, alpha=0.3)
        for data in interface_data:
            if data['type'] == 'vertical':
                x, y_start = data['position']
                y_end = y_start + len(data['interface_values'])
                ax4.plot([x, x], [y_start, y_end], 'r-', linewidth=2)
            else:  # horizontal
                x_start, y = data['position']
                x_end = x_start + len(data['interface_values'])
                ax4.plot([x_start, x_end], [y, y], 'b-', linewidth=2)
        ax4.set_title('Interface Locations\n(Red: Vertical, Blue: Horizontal)')
        
        plt.tight_layout()
        plt.savefig(f'plots/sample_{sample_idx:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def save_generated_data(self, dataset: List[Dict], sample_idx: int):
        """
        Save generated data sample to disk.
        
        Args:
            dataset (List[Dict]): List of generated data samples
            sample_idx (int): Index of the current sample
        """
        data = dataset[-1]  # Get the last generated sample
        
        # Create a directory for this sample
        sample_dir = f'data/generated_samples/sample_{sample_idx:04d}'
        os.makedirs(sample_dir, exist_ok=True)
        
        # Save arrays
        np.save(f'{sample_dir}/theta.npy', data['theta'])
        np.save(f'{sample_dir}/f.npy', data['f'])
        np.save(f'{sample_dir}/solution.npy', data['solution'])
        
        # Save interface data as JSON
        interface_data = []
        for iface in data['interface_data']:
            interface_dict = {
                'type': iface['type'],
                'position': iface['position'],
                'interface_values': iface['interface_values'].tolist()
            }
            if 'context_left' in iface:
                interface_dict['context_left'] = iface['context_left'].tolist()
                interface_dict['context_right'] = iface['context_right'].tolist()
            if 'context_bottom' in iface:
                interface_dict['context_bottom'] = iface['context_bottom'].tolist()
                interface_dict['context_top'] = iface['context_top'].tolist()
            interface_data.append(interface_dict)
        
        with open(f'{sample_dir}/interface_data.json', 'w') as f:
            json.dump(interface_data, f)
        
        # Save metadata
        metadata = {
            'iterations': data['iterations'],
            'grid_size': (self.nx, self.ny),
            'n_subdomains': self.n_subdomains
        }
        with open(f'{sample_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f)

    def train_model(self, n_samples: int = 1000, n_epochs: int = 100, force_retrain: bool = False):
        """
        Train the ML model on generated data. Skip training if model weights already exist.
        
        Args:
            n_samples (int): Number of training samples to generate
            n_epochs (int): Number of training epochs
            force_retrain (bool): If True, retrain even if model weights exist
        """
        # Check if model weights already exist
        if not force_retrain and os.path.exists('models/best_vertical_model.pth') and os.path.exists('models/best_horizontal_model.pth'):
            print("\n=== Loading existing model weights ===")
            # Load the best models
            self.predictor.load_model('models/best')
            
            # Load and print the best losses
            vertical_checkpoint = torch.load('models/best_vertical_model.pth', map_location=self.predictor.device)
            horizontal_checkpoint = torch.load('models/best_horizontal_model.pth', map_location=self.predictor.device)
            
            print(f"Loaded pre-trained models:")
            print(f"  Best vertical loss: {vertical_checkpoint['loss']:.6f} at epoch {vertical_checkpoint['epoch']}")
            print(f"  Best horizontal loss: {horizontal_checkpoint['loss']:.6f} at epoch {horizontal_checkpoint['epoch']}")
            return

        print("\n=== Training Phase ===")
        start_time = time.time()
        
        # Load all interface data from saved files
        all_interface_data = []
        print("Loading saved interface data...")
        for i in tqdm(range(n_samples), desc="Processing samples"):
            sample_dir = f'data/generated_samples/sample_{i:04d}'
            
            # Load interface data
            with open(f'{sample_dir}/interface_data.json', 'r') as f:
                interface_data = json.load(f)
            
            # Convert lists back to numpy arrays
            for iface in interface_data:
                iface['interface_values'] = np.array(iface['interface_values'])
                if 'context_left' in iface:
                    iface['context_left'] = np.array(iface['context_left'])
                    iface['context_right'] = np.array(iface['context_right'])
                if 'context_bottom' in iface:
                    iface['context_bottom'] = np.array(iface['context_bottom'])
                    iface['context_top'] = np.array(iface['context_top'])
            
            all_interface_data.extend(interface_data)
        
        n_samples = len(all_interface_data)
        batch_size = 32
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        print(f"\nStarting training with:")
        print(f"  Total interface samples: {n_samples}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches per epoch: {n_batches}")
        
        # Training loop with progress tracking
        epoch_pbar = tqdm(range(n_epochs), desc="Training epochs")
        best_vertical_loss = float('inf')
        best_horizontal_loss = float('inf')
        
        for epoch in epoch_pbar:
            total_vertical_loss = 0.0
            total_horizontal_loss = 0.0
            n_vertical = 0
            n_horizontal = 0
            
            # Shuffle data
            np.random.shuffle(all_interface_data)
            
            # Batch progress bar
            batch_pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
            
            for i in batch_pbar:
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                batch_data = all_interface_data[start_idx:end_idx]
                
                # Prepare batch
                inputs, targets = self.predictor.prepare_batch(batch_data)
                
                # Train vertical interfaces
                if inputs['vertical'] is not None:
                    self.predictor.vertical_optimizer.zero_grad()
                    outputs = self.predictor.vertical_model(inputs['vertical'])
                    loss = self.predictor.criterion(outputs, targets['vertical'])
                    loss.backward()
                    self.predictor.vertical_optimizer.step()
                    total_vertical_loss += loss.item()
                    n_vertical += 1
                
                # Train horizontal interfaces
                if inputs['horizontal'] is not None:
                    self.predictor.horizontal_optimizer.zero_grad()
                    outputs = self.predictor.horizontal_model(inputs['horizontal'])
                    loss = self.predictor.criterion(outputs, targets['horizontal'])
                    loss.backward()
                    self.predictor.horizontal_optimizer.step()
                    total_horizontal_loss += loss.item()
                    n_horizontal += 1
                
                # Update batch progress
                avg_batch_loss = (total_vertical_loss / max(1, n_vertical) + 
                                total_horizontal_loss / max(1, n_horizontal)) / 2
                batch_pbar.set_postfix({'loss': f'{avg_batch_loss:.6f}'})
            
            # Calculate average losses
            avg_vertical_loss = total_vertical_loss / n_vertical if n_vertical > 0 else float('inf')
            avg_horizontal_loss = total_horizontal_loss / n_horizontal if n_horizontal > 0 else float('inf')
            
            # Track best models
            if avg_vertical_loss < best_vertical_loss:
                best_vertical_loss = avg_vertical_loss
                print(f"\nNew best vertical loss: {best_vertical_loss:.6f} at epoch {epoch+1}")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.predictor.vertical_model.state_dict(),
                    'optimizer_state_dict': self.predictor.vertical_optimizer.state_dict(),
                    'loss': best_vertical_loss,
                }, f'models/best_vertical_model.pth')
            
            if avg_horizontal_loss < best_horizontal_loss:
                best_horizontal_loss = avg_horizontal_loss
                print(f"\nNew best horizontal loss: {best_horizontal_loss:.6f} at epoch {epoch+1}")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.predictor.horizontal_model.state_dict(),
                    'optimizer_state_dict': self.predictor.horizontal_optimizer.state_dict(),
                    'loss': best_horizontal_loss,
                }, f'models/best_horizontal_model.pth')
            
            # Save checkpoints every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.predictor.save_model(f'models/checkpoint_epoch_{epoch+1:03d}')
                print(f"\nEpoch {epoch+1} Statistics:")
                print(f"  Vertical Loss: {avg_vertical_loss:.6f}")
                print(f"  Horizontal Loss: {avg_horizontal_loss:.6f}")
        
        train_time = time.time() - start_time
        print(f"\nTraining completed in {train_time:.2f} seconds")
        print(f"Best vertical loss: {best_vertical_loss:.6f}")
        print(f"Best horizontal loss: {best_horizontal_loss:.6f}")
        print(f"Model weights saved in models/")
    
    def solve_with_ml_interface(self, theta: np.ndarray, f: np.ndarray,
                              bc_dict: Dict[str, callable]) -> Tuple[np.ndarray, float]:
        """
        Solve the PDE using ML-predicted interface conditions.
        
        Args:
            theta (np.ndarray): Diffusion coefficient field
            f (np.ndarray): Source term
            bc_dict (Dict[str, callable]): External boundary conditions
            
        Returns:
            Tuple[np.ndarray, float]: Solution array and maximum error
        """
        # First solve the full domain problem for comparison
        full_solver = PoissonSolver(self.nx, self.ny)
        u_true, _ = full_solver.solve_full_domain(
            theta, f,
            bc_dict['left'], bc_dict['right'],
            bc_dict['bottom'], bc_dict['top']
        )
        
        # Initialize solution array
        u_ml = np.zeros((self.ny, self.nx))
        
        # Extract interface data from true solution
        interface_data = self.data_generator.extract_interface_data(u_true)
        
        # Predict interface conditions using ML model
        predicted_interfaces = {}
        for data in interface_data:
            pred_values = self.predictor.predict(data)
            key = (data['position'], data['type'])
            predicted_interfaces[key] = pred_values
        
        # Solve each subdomain with ML-predicted interface conditions
        for i in range(self.n_subdomains[1]):
            for j in range(self.n_subdomains[0]):
                # Get subdomain indices
                x_start = j * self.subdomain_nx
                x_end = (j + 1) * self.subdomain_nx
                y_start = i * self.subdomain_ny
                y_end = (i + 1) * self.subdomain_ny
                
                # Get subdomain theta and f
                theta_sub = theta[y_start:y_end, x_start:x_end]
                f_sub = f[y_start:y_end, x_start:x_end]
                
                # Prepare boundary conditions for subdomain
                bc_sub = {}
                
                # Left boundary
                if j == 0:
                    bc_sub['left'] = lambda y: bc_dict['left'](y + y_start)
                else:
                    key = ((x_start, y_start), 'vertical')
                    values = predicted_interfaces[key]
                    bc_sub['left'] = lambda y: float(values[int(y)])
                
                # Right boundary
                if j == self.n_subdomains[0] - 1:
                    bc_sub['right'] = lambda y: bc_dict['right'](y + y_start)
                else:
                    key = ((x_end, y_start), 'vertical')
                    values = predicted_interfaces[key]
                    bc_sub['right'] = lambda y: float(values[int(y)])
                
                # Bottom boundary
                if i == 0:
                    bc_sub['bottom'] = lambda x: bc_dict['bottom'](x + x_start)
                else:
                    key = ((x_start, y_start), 'horizontal')
                    values = predicted_interfaces[key]
                    bc_sub['bottom'] = lambda x: float(values[int(x)])
                
                # Top boundary
                if i == self.n_subdomains[1] - 1:
                    bc_sub['top'] = lambda x: bc_dict['top'](x + x_start)
                else:
                    key = ((x_start, y_end), 'horizontal')
                    values = predicted_interfaces[key]
                    bc_sub['top'] = lambda x: float(values[int(x)])
                
                # Solve subdomain
                u_sub, _ = self.subdomain_solvers[i][j].solve_subdomain(
                    theta_sub, f_sub, bc_sub
                )
                
                # Store solution in global array
                u_ml[y_start:y_end, x_start:x_end] = u_sub
        
        # Calculate maximum error
        max_error = np.max(np.abs(u_ml - u_true))
        
        return u_ml, max_error
    
    def run_convergence_study(self, n_problems: int = 10) -> List[float]:
        """
        Run a convergence study comparing ML-predicted interface conditions
        with true interface conditions.
        
        Args:
            n_problems (int): Number of test problems to solve
            
        Returns:
            List[float]: List of maximum errors for each problem
        """
        errors = []
        
        for i in tqdm(range(n_problems), desc="Running convergence study"):
            # Generate random problem
            theta, f = self.data_generator.generate_random_problem()
            bc_dict = self.data_generator.generate_boundary_conditions()
            
            # Solve with ML-predicted interface conditions
            _, error = self.solve_with_ml_interface(theta, f, bc_dict)
            errors.append(error)
        
        return errors
    
    def visualize_solution(self, u_ml: np.ndarray, u_true: np.ndarray):
        """
        Visualize the ML solution and compare with true solution.
        
        Args:
            u_ml (np.ndarray): Solution with ML-predicted interface conditions
            u_true (np.ndarray): True solution
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot ML solution
        im1 = ax1.imshow(u_ml)
        ax1.set_title('ML Solution')
        plt.colorbar(im1, ax=ax1)
        
        # Plot true solution
        im2 = ax2.imshow(u_true)
        ax2.set_title('True Solution')
        plt.colorbar(im2, ax=ax2)
        
        # Plot error
        error = np.abs(u_ml - u_true)
        im3 = ax3.imshow(error)
        ax3.set_title('Absolute Error')
        plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        plt.show()

    def visualize_samples(self, n_samples: int = 5):
        """
        Visualize multiple random samples and their solutions.
        
        Args:
            n_samples (int): Number of samples to visualize
        """
        print(f"\nVisualizing {n_samples} random samples...")
        
        for i in range(n_samples):
            # Generate random problem
            theta, f = self.data_generator.generate_random_problem()
            bc_dict = self.data_generator.generate_boundary_conditions()
            
            # Get ML solution
            u_ml, error = self.solve_with_ml_interface(theta, f, bc_dict)
            
            # Get true solution
            u_true, _ = self.full_solver.solve_full_domain(
                theta, f,
                bc_dict['left'], bc_dict['right'],
                bc_dict['bottom'], bc_dict['top']
            )
            
            # Create figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot theta
            im1 = ax1.imshow(theta)
            ax1.set_title('Diffusion Coefficient (θ)')
            plt.colorbar(im1, ax=ax1)
            
            # Plot source term
            im2 = ax2.imshow(f)
            ax2.set_title('Source Term (f)')
            plt.colorbar(im2, ax=ax2)
            
            # Plot ML solution
            im3 = ax3.imshow(u_ml)
            ax3.set_title(f'ML Solution (Max Error: {error:.6f})')
            plt.colorbar(im3, ax=ax3)
            
            # Plot error
            error_field = np.abs(u_ml - u_true)
            im4 = ax4.imshow(error_field)
            ax4.set_title('Absolute Error')
            plt.colorbar(im4, ax=ax4)
            
            plt.suptitle(f'Sample {i+1}', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'plots/random_sample_{i+1}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Sample {i+1} - Max Error: {error:.6f}")

    def save_convergence_plot(self, errors: List[float]):
        """
        Save a plot of the convergence study results.
        
        Args:
            errors (List[float]): List of errors from convergence study
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(errors) + 1), errors, 'bo-', linewidth=2, markersize=8)
        plt.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'Mean Error: {np.mean(errors):.2f}')
        plt.fill_between(range(1, len(errors) + 1), 
                        np.mean(errors) - np.std(errors), 
                        np.mean(errors) + np.std(errors), 
                        color='r', alpha=0.2, label=f'Std Dev: {np.std(errors):.2f}')
        
        plt.xlabel('Problem Number')
        plt.ylabel('Maximum Error')
        plt.title('Convergence Study Results')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/convergence_study.png', dpi=150, bbox_inches='tight')
        plt.close()

    def run_timing_comparison(self, n_problems: int = 50):
        """
        Compare solution times between ML-predicted interface conditions (parallelized)
        and full domain solution.
        
        Args:
            n_problems (int): Number of test problems to solve
        """
        ml_times = []
        full_times = []
        ml_errors = []
        
        print(f"\nRunning timing comparison over {n_problems} problems...")
        
        for i in tqdm(range(n_problems), desc="Solving problems"):
            # Generate random problem
            theta, f = self.data_generator.generate_random_problem()
            bc_dict = self.data_generator.generate_boundary_conditions()
            
            # Time ML solution
            ml_start = time.time()
            u_ml, error = self.solve_with_ml_interface(theta, f, bc_dict)
            ml_time = time.time() - ml_start
            ml_times.append(ml_time)
            ml_errors.append(error)
            
            # Time full domain solution
            full_start = time.time()
            u_true, _ = self.full_solver.solve_full_domain(
                theta, f,
                bc_dict['left'], bc_dict['right'],
                bc_dict['bottom'], bc_dict['top']
            )
            full_time = time.time() - full_start
            full_times.append(full_time)
        
        # Calculate statistics
        avg_ml_time = np.mean(ml_times)
        std_ml_time = np.std(ml_times)
        avg_full_time = np.mean(full_times)
        std_full_time = np.std(full_times)
        avg_speedup = avg_full_time / avg_ml_time
        
        # Print results
        print("\nTiming Comparison Results:")
        print(f"ML Solution (with interface prediction):")
        print(f"  Average time: {avg_ml_time:.6f} seconds")
        print(f"  Std dev time: {std_ml_time:.6f} seconds")
        print(f"  Average error: {np.mean(ml_errors):.6f}")
        print(f"\nFull Domain Solution:")
        print(f"  Average time: {avg_full_time:.6f} seconds")
        print(f"  Std dev time: {std_full_time:.6f} seconds")
        print(f"\nAverage speedup: {avg_speedup:.2f}x")
        
        # Create timing comparison plot
        plt.figure(figsize=(12, 6))
        
        # Plot timing distributions
        plt.subplot(1, 2, 1)
        plt.hist(ml_times, bins=20, alpha=0.5, label='ML Solution')
        plt.hist(full_times, bins=20, alpha=0.5, label='Full Solution')
        plt.xlabel('Solution Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Solution Times')
        plt.legend()
        
        # Plot timing comparison
        plt.subplot(1, 2, 2)
        plt.boxplot([ml_times, full_times], labels=['ML Solution', 'Full Solution'])
        plt.ylabel('Time (seconds)')
        plt.title('Solution Time Comparison')
        
        plt.tight_layout()
        plt.savefig('plots/timing_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'ml_times': ml_times,
            'full_times': full_times,
            'ml_errors': ml_errors,
            'avg_speedup': avg_speedup
        }

def main():
    """Run a sample experiment."""
    # Initialize experiment
    exp = Experiment()
    
    # Train the model (will load existing weights if available)
    exp.train_model(n_samples=1000, n_epochs=100, force_retrain=False)
    
    # Run timing comparison
    timing_results = exp.run_timing_comparison(n_problems=50)
    
    # Visualize a few random samples
    exp.visualize_samples(n_samples=5)
    
    # Run convergence study
    errors = exp.run_convergence_study(n_problems=10)
    
    # Save convergence plot
    exp.save_convergence_plot(errors)
    
    # Print statistics
    print(f"\nConvergence Study Results:")
    print(f"Mean Error: {np.mean(errors):.6f}")
    print(f"Std Error: {np.std(errors):.6f}")
    print(f"Max Error: {np.max(errors):.6f}")

if __name__ == "__main__":
    main() 