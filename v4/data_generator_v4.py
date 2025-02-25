import numpy as np
from typing import Tuple, List, Dict, Callable
from pde_solver import PoissonSolver
import matplotlib.pyplot as plt

class DataGeneratorV4:
    """
    Data generator for ML interface prediction using numerical solutions.
    Uses constant theta=1 and extended range of k values (1.0 to 6.0).
    All values are scaled to be between -1 and 1.
    """
    
    def __init__(self, nx: int = 40, ny: int = 40, n_subdomains: Tuple[int, int] = (2, 2),
                 patch_size: int = 5):
        """
        Initialize the data generator.
        
        Args:
            nx (int): Number of points in x direction for full domain
            ny (int): Number of points in y direction for full domain
            n_subdomains (Tuple[int, int]): Number of subdomains in (x, y) directions
            patch_size (int): Size of local context patch around interface points
        """
        self.nx = nx
        self.ny = ny
        self.n_subdomains = n_subdomains
        self.patch_size = patch_size
        
        # Calculate subdomain sizes
        self.subdomain_nx = nx // n_subdomains[0]
        self.subdomain_ny = ny // n_subdomains[1]
        
        # Create coordinate grids
        self.x = np.linspace(0, 1, nx)
        self.y = np.linspace(0, 1, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Wavenumbers for solution generation (extended range)
        self.k_values = np.arange(1.0, 6.5, 0.5)  # 1.0, 1.5, 2.0, ..., 6.0
        
        # Initialize scaling factors
        self.scale_factors = {
            'theta': {'min': 1.0, 'max': 1.0},  # Theta is constant at 1
            'solution': {'min': -1.0, 'max': 1.0}  # Solution and BCs between -1 and 1
        }
        
        # Initialize PDE solver
        self.solver = PoissonSolver(nx, ny)
    
    def scale_array(self, arr: np.ndarray, min_val: float = -1.0, max_val: float = 1.0) -> np.ndarray:
        """Scale array to target range."""
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        
        # Handle constant arrays
        if arr_min == arr_max:
            return np.zeros_like(arr) if arr_min == 0 else np.ones_like(arr) * min_val
        
        return (arr - arr_min) * (max_val - min_val) / (arr_max - arr_min) + min_val
    
    def generate_random_problem(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a random problem with numerical solution.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: theta, f, u_numerical
        """
        # Use constant theta=1
        theta = np.ones((self.ny, self.nx))
        
        # Randomly select k values
        k1 = np.random.choice(self.k_values)
        k2 = np.random.choice(self.k_values)
        coef = np.random.uniform(0.1, 0.2)
        
        # Generate source term f
        f = coef * np.sin(k1 * 2 * np.pi * self.X) * np.sin(k2 * 2 * np.pi * self.Y)
        
        # Generate boundary conditions from analytical solution
        u_analytical = coef * np.sin(k1 * 2 * np.pi * self.X) * np.sin(k2 * 2 * np.pi * self.Y)
        bc_dict = self.generate_boundary_conditions(u_analytical)
        
        # Solve PDE numerically
        u_numerical, _ = self.solver.solve_full_domain(
            theta, -f,  # Negate f to match ∇·(θ∇u) = f
            bc_dict['left'], bc_dict['right'],
            bc_dict['bottom'], bc_dict['top']
        )
        
        # Scale the arrays
        u_scaled = self.scale_array(u_numerical)
        f_scaled = self.scale_array(f)
        theta_scaled = self.scale_array(theta, self.scale_factors['theta']['min'], 
                                      self.scale_factors['theta']['max'])
        
        return theta_scaled, -f_scaled, u_scaled  # Negate f to match ∇·(θ∇u) = f
    
    def generate_boundary_conditions(self, u: np.ndarray) -> Dict[str, Callable]:
        """
        Generate boundary condition functions from a solution array.
        The solution u is already scaled between -1 and 1.
        
        Args:
            u: Scaled numerical solution array
            
        Returns:
            Dict[str, callable]: Dictionary of boundary condition functions
        """
        def bc_left(y):
            if isinstance(y, (int, float)):
                y_idx = int(y)
                return float(u[y_idx, 0])
            return np.array([float(u[int(yi), 0]) for yi in y])
        
        def bc_right(y):
            if isinstance(y, (int, float)):
                y_idx = int(y)
                return float(u[y_idx, -1])
            return np.array([float(u[int(yi), -1]) for yi in y])
        
        def bc_bottom(x):
            if isinstance(x, (int, float)):
                x_idx = int(x)
                return float(u[0, x_idx])
            return np.array([float(u[0, int(xi)]) for xi in x])
        
        def bc_top(x):
            if isinstance(x, (int, float)):
                x_idx = int(x)
                return float(u[-1, x_idx])
            return np.array([float(u[-1, int(xi)]) for xi in x])
        
        return {
            'left': bc_left,
            'right': bc_right,
            'bottom': bc_bottom,
            'top': bc_top
        }
    
    def extract_patch(self, field: np.ndarray, center_x: int, center_y: int) -> np.ndarray:
        """Extract a patch from a field centered at (center_x, center_y)."""
        half_size = self.patch_size // 2
        patch = field[center_y - half_size:center_y + half_size + 1,
                     center_x - half_size:center_x + half_size + 1]
        return patch
    
    def extract_interface_data(self, theta: np.ndarray, f: np.ndarray, u: np.ndarray,
                             bc_dict: Dict[str, Callable]) -> List[Dict]:
        """Extract interface data for training."""
        interface_data = []
        
        # Extract vertical interface points
        for i in range(1, self.n_subdomains[0]):
            x = i * self.subdomain_nx
            for y in range(self.patch_size//2, self.ny - self.patch_size//2):
                theta_patch = self.extract_patch(theta, x, y)
                f_patch = self.extract_patch(f, x, y)
                target = u[y, x]
                
                interface_data.append({
                    'position': (x, y),
                    'type': 'vertical',
                    'theta_patch': theta_patch,
                    'f_patch': f_patch,
                    'target': target
                })
        
        # Extract horizontal interface points
        for i in range(1, self.n_subdomains[1]):
            y = i * self.subdomain_ny
            for x in range(self.patch_size//2, self.nx - self.patch_size//2):
                theta_patch = self.extract_patch(theta, x, y)
                f_patch = self.extract_patch(f, x, y)
                target = u[y, x]
                
                interface_data.append({
                    'position': (x, y),
                    'type': 'horizontal',
                    'theta_patch': theta_patch,
                    'f_patch': f_patch,
                    'target': target
                })
        
        return interface_data

if __name__ == "__main__":
    # Test the data generator
    data_gen = DataGeneratorV4()
    theta, f, u = data_gen.generate_random_problem()
    bc_dict = data_gen.generate_boundary_conditions(u)
    interface_data = data_gen.extract_interface_data(theta, f, u, bc_dict)
    
    # Plot the generated fields
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(theta, origin='lower')
    axes[0].set_title('Theta (Constant=1)')
    
    axes[1].imshow(f, origin='lower')
    axes[1].set_title('Source Term f')
    
    axes[2].imshow(u, origin='lower')
    axes[2].set_title('Solution u')
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"Number of interface points: {len(interface_data)}")
    print(f"k values available: {data_gen.k_values}")
    print(f"Solution range: [{u.min():.4f}, {u.max():.4f}]")
    print(f"f range: [{f.min():.4f}, {f.max():.4f}]") 