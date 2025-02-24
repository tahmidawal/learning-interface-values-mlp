import numpy as np
from typing import Tuple, List, Dict, Callable

class DataGeneratorV2:
    """
    Enhanced data generator for ML interface prediction using analytical solutions.
    Uses constant theta=1 and analytical solutions from sine functions.
    All values are scaled to be between -1 and 1.
    """
    
    def __init__(self, nx: int = 40, ny: int = 40, n_subdomains: Tuple[int, int] = (2, 2),
                 patch_size: int = 3):
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
        
        # Wavenumbers for solution generation
        self.k_values = np.arange(0.5, 8.5, 0.5)
        
        # Initialize scaling factors
        self.scale_factors = {
            'theta': {'min': 0.5, 'max': 1.5},  # Theta will be scaled around 1
            'solution': {'min': -1.0, 'max': 1.0},  # Solution and BCs between -1 and 1
            'f': {'min': -1.0, 'max': 1.0}  # Source term between -1 and 1
        }
    
    def scale_array(self, arr: np.ndarray, target_min: float = -1.0, target_max: float = 1.0) -> np.ndarray:
        """Scale array to target range."""
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        
        # Handle constant arrays
        if arr_min == arr_max:
            return np.zeros_like(arr) if arr_min == 0 else np.ones_like(arr) * target_min
        
        return (arr - arr_min) * (target_max - target_min) / (arr_max - arr_min) + target_min
    
    def generate_random_problem(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate random problem with analytical solution.
        All values are scaled to be between -1 and 1.
        The PDE is properly scaled to ensure consistent solutions.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: theta, f, and analytical solution
        """
        # Generate random coefficients and k values for the solution
        n_terms = np.random.randint(1, 3)  # Use 1-2 terms for simpler solutions
        coefficients = np.random.uniform(0.1, 0.2, n_terms)  # Further reduced coefficient range
        k_indices = np.random.choice(len(self.k_values), size=(n_terms, 2), replace=True)
        
        # Initialize solution and source term
        u = np.zeros((self.ny, self.nx))
        f = np.zeros((self.ny, self.nx))
        
        # Build solution and source term
        for i in range(n_terms):
            k1, k2 = self.k_values[k_indices[i]]
            coef = coefficients[i]
            
            # Solution term: A*sin(k₁*2πx)*sin(k₂*2πy)
            u_term = coef * np.sin(k1 * 2 * np.pi * self.X) * np.sin(k2 * 2 * np.pi * self.Y)
            u += u_term
            
            # Source term: -A*(k₁²+k₂²)*(2π)²*sin(k₁*2πx)*sin(k₂*2πy)
            # Scale down the source term to match the solution scale
            k_factor = (k1**2 + k2**2) * (2 * np.pi)**2
            f_term = -coef * np.sin(k1 * 2 * np.pi * self.X) * np.sin(k2 * 2 * np.pi * self.Y)
            f += f_term
        
        # Generate theta very close to 1 for better scaling
        theta = np.ones((self.ny, self.nx)) + 0.01 * np.random.randn(self.ny, self.nx)
        theta = np.clip(theta, 0.95, 1.05)  # Very small variations around 1
        
        # Scale the arrays
        u_scaled = self.scale_array(u)
        f_scaled = self.scale_array(f)
        theta_scaled = self.scale_array(theta, self.scale_factors['theta']['min'], self.scale_factors['theta']['max'])
        
        return theta_scaled, f_scaled, u_scaled
    
    def generate_boundary_conditions(self, u: np.ndarray) -> Dict[str, Callable]:
        """
        Generate boundary conditions from the analytical solution.
        The solution u is already scaled between -1 and 1.
        
        Args:
            u: Scaled analytical solution array
            
        Returns:
            Dict[str, callable]: Dictionary of boundary condition functions
        """
        # Create interpolation functions for the boundaries
        def bc_left(y):
            y_idx = int(y * (self.ny - 1))
            return float(u[y_idx, 0])
        
        def bc_right(y):
            y_idx = int(y * (self.ny - 1))
            return float(u[y_idx, -1])
        
        def bc_bottom(x):
            x_idx = int(x * (self.nx - 1))
            return float(u[0, x_idx])
        
        def bc_top(x):
            x_idx = int(x * (self.nx - 1))
            return float(u[-1, x_idx])
        
        return {
            'left': bc_left,
            'right': bc_right,
            'bottom': bc_bottom,
            'top': bc_top
        }
    
    def extract_patch(self, field: np.ndarray, center_x: int, center_y: int) -> np.ndarray:
        """
        Extract a patch from a field centered at (center_x, center_y).
        
        Args:
            field: Field to extract patch from
            center_x: x-coordinate of patch center
            center_y: y-coordinate of patch center
            
        Returns:
            np.ndarray: Extracted patch
        """
        half_size = self.patch_size // 2
        patch = field[max(0, center_y - half_size):min(self.ny, center_y + half_size + 1),
                     max(0, center_x - half_size):min(self.nx, center_x + half_size + 1)]
        
        # Pad if necessary
        if patch.shape[0] < self.patch_size:
            pad_top = max(0, half_size - center_y)
            pad_bottom = max(0, center_y + half_size + 1 - self.ny)
            patch = np.pad(patch, ((pad_top, pad_bottom), (0, 0)), mode='edge')
        
        if patch.shape[1] < self.patch_size:
            pad_left = max(0, half_size - center_x)
            pad_right = max(0, center_x + half_size + 1 - self.nx)
            patch = np.pad(patch, ((0, 0), (pad_left, pad_right)), mode='edge')
        
        return patch
    
    def extract_interface_data(self, theta: np.ndarray, f: np.ndarray, solution: np.ndarray,
                             bc_dict: Dict[str, Callable]) -> List[Dict]:
        """
        Extract interface data including local context and global boundary information.
        
        Args:
            theta: Diffusion coefficient field
            f: Source term field
            solution: Analytical solution
            bc_dict: Dictionary of boundary conditions
            
        Returns:
            List[Dict]: List of interface data dictionaries
        """
        interface_data = []
        
        # Helper function to get boundary values
        def get_boundary_values(bc_func, size):
            return np.array([bc_func(i/size) for i in range(size)])
        
        # Extract vertical interfaces
        for i in range(1, self.n_subdomains[0]):
            x_idx = i * self.subdomain_nx
            for y_idx in range(self.ny):
                # Extract local patches
                theta_patch = self.extract_patch(theta, x_idx, y_idx)
                f_patch = self.extract_patch(f, x_idx, y_idx)
                
                # Get corresponding boundary values
                left_bc = get_boundary_values(bc_dict['left'], self.ny)[y_idx]
                right_bc = get_boundary_values(bc_dict['right'], self.ny)[y_idx]
                bottom_bc = get_boundary_values(bc_dict['bottom'], self.nx)[x_idx]
                top_bc = get_boundary_values(bc_dict['top'], self.nx)[x_idx]
                
                interface_data.append({
                    'type': 'vertical',
                    'position': (x_idx, y_idx),
                    'theta_patch': theta_patch,
                    'f_patch': f_patch,
                    'boundary_values': np.array([left_bc, right_bc, bottom_bc, top_bc]),
                    'target': solution[y_idx, x_idx]
                })
        
        # Extract horizontal interfaces
        for j in range(1, self.n_subdomains[1]):
            y_idx = j * self.subdomain_ny
            for x_idx in range(self.nx):
                # Extract local patches
                theta_patch = self.extract_patch(theta, x_idx, y_idx)
                f_patch = self.extract_patch(f, x_idx, y_idx)
                
                # Get corresponding boundary values
                left_bc = get_boundary_values(bc_dict['left'], self.ny)[y_idx]
                right_bc = get_boundary_values(bc_dict['right'], self.ny)[y_idx]
                bottom_bc = get_boundary_values(bc_dict['bottom'], self.nx)[x_idx]
                top_bc = get_boundary_values(bc_dict['top'], self.nx)[x_idx]
                
                interface_data.append({
                    'type': 'horizontal',
                    'position': (x_idx, y_idx),
                    'theta_patch': theta_patch,
                    'f_patch': f_patch,
                    'boundary_values': np.array([left_bc, right_bc, bottom_bc, top_bc]),
                    'target': solution[y_idx, x_idx]
                })
        
        return interface_data
    
    def generate_dataset(self, n_samples: int) -> List[Dict]:
        """
        Generate a dataset of n_samples problems and their interface data.
        
        Args:
            n_samples (int): Number of samples to generate
            
        Returns:
            List[Dict]: List of dictionaries containing problem data and interface values
        """
        dataset = []
        
        for _ in range(n_samples):
            # Generate random problem with scaled values
            theta, f, solution = self.generate_random_problem()
            
            # Generate boundary conditions from scaled solution
            bc_dict = self.generate_boundary_conditions(solution)
            
            # Extract interface data
            interface_data = self.extract_interface_data(theta, f, solution, bc_dict)
            
            # Store problem data
            dataset.append({
                'theta': theta,
                'f': f,
                'solution': solution,
                'interface_data': interface_data,
                'bc_dict': bc_dict,
                'scale_factors': self.scale_factors
            })
        
        return dataset 