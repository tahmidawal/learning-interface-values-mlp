import numpy as np
import matplotlib.pyplot as plt
import torch
from data_generator_v2 import DataGeneratorV2
import os

def plot_field(field: np.ndarray, title: str, ax: plt.Axes):
    """Plot a 2D field with proper formatting."""
    im = ax.imshow(field, cmap='viridis', aspect='equal', origin='lower')
    ax.set_title(title)
    return plt.colorbar(im, ax=ax)

def plot_interface_points(interface_data: list, nx: int, ny: int, ax: plt.Axes):
    """Plot interface points with their values."""
    positions = [(p['position'], p['target']) for p in interface_data]
    x_pos = [x for (x, y), _ in positions]
    y_pos = [y for (x, y), _ in positions]
    values = [v for _, v in positions]
    
    scatter = ax.scatter(x_pos, y_pos, c=values, cmap='viridis', s=100)
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_title('Interface Points and Values')
    return plt.colorbar(scatter, ax=ax)

def main():
    # Create results directory
    results_dir = 'training_data_samples'
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize data generator
    print("Initializing data generator...")
    data_gen = DataGeneratorV2(nx=40, ny=40, n_subdomains=(2, 2), patch_size=5)
    
    # Generate and visualize samples
    n_samples = 5
    print(f"Generating {n_samples} training data samples...")
    
    for i in range(n_samples):
        print(f"Processing sample {i+1}...")
        
        # Generate sample
        theta, f, u_exact = data_gen.generate_random_problem()
        bc_dict = data_gen.generate_boundary_conditions(u_exact)
        interface_data = data_gen.extract_interface_data(theta, f, u_exact, bc_dict)
        
        # Create figure
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 4)
        
        # Plot full fields
        ax_theta = fig.add_subplot(gs[0, 0])
        plot_field(theta, 'Theta Field', ax_theta)
        
        ax_f = fig.add_subplot(gs[0, 1])
        plot_field(f, 'Source Term f', ax_f)
        
        ax_u = fig.add_subplot(gs[0, 2])
        plot_field(u_exact, 'Exact Solution u', ax_u)
        
        ax_interface = fig.add_subplot(gs[0, 3])
        plot_interface_points(interface_data, data_gen.nx, data_gen.ny, ax_interface)
        
        # Plot some example patches
        n_patches = min(6, len(interface_data))
        for j in range(n_patches):
            ax = fig.add_subplot(gs[1 + j//3, j%3])
            point = interface_data[j]
            plot_field(point['theta_patch'], 
                      f"Patch at {point['position']}\n"
                      f"Type: {point['type']}\n"
                      f"Target: {point['target']:.4f}", ax)
        
        # Add global title
        k_values = data_gen.k_values
        plt.suptitle(f'Training Sample {i+1}\n'
                    f'Available k values: {k_values[0]:.1f} to {k_values[-1]:.1f}\n'
                    f'Scale factors: theta [{data_gen.scale_factors["theta"]["min"]}, '
                    f'{data_gen.scale_factors["theta"]["max"]}], '
                    f'solution/f [{data_gen.scale_factors["solution"]["min"]}, '
                    f'{data_gen.scale_factors["solution"]["max"]}]')
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(results_dir, f'training_sample_{i+1}.png')
        print(f"Saving plot to {save_path}")
        plt.savefig(save_path)
        plt.close()
        
        # Print sample statistics
        print(f"  Sample {i+1} Statistics:")
        print(f"    Theta range: [{theta.min():.4f}, {theta.max():.4f}]")
        print(f"    f range: [{f.min():.4f}, {f.max():.4f}]")
        print(f"    Solution range: [{u_exact.min():.4f}, {u_exact.max():.4f}]")
        print(f"    Number of interface points: {len(interface_data)}")
        
        # Print interface point statistics
        interface_vals = [p['target'] for p in interface_data]
        print(f"    Interface values range: [{min(interface_vals):.4f}, {max(interface_vals):.4f}]")
        print(f"    Interface values mean: {np.mean(interface_vals):.4f}")
        print(f"    Interface values std: {np.std(interface_vals):.4f}")
        
        # Count interface types
        interface_types = {}
        for point in interface_data:
            itype = point['type']
            interface_types[itype] = interface_types.get(itype, 0) + 1
        print("    Interface point types:")
        for itype, count in interface_types.items():
            print(f"      {itype}: {count}")
        print()

if __name__ == "__main__":
    main() 