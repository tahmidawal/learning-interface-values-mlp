import numpy as np
from typing import Tuple, List, Dict, Callable

class DataGeneratorV2:
    """
    Enhanced data generator for ML interface prediction using analytical solutions.
    Uses constant theta=1 and analytical solutions from sine functions.
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
    
    def generate_random_problem(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate random problem with analytical solution.
        Uses constant theta=1 and generates solution as sum of sine functions.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: theta, f, and analytical solution
        """
        # Constant theta = 1
        theta = np.ones((self.ny, self.nx))
        
        # Generate random coefficients and k values for the solution
        n_terms = np.random.randint(1, 4)  # Use 1-3 terms
        coefficients = np.random.uniform(0.1, 1.0, n_terms)
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
            # This comes from -∇²u when theta=1
            f_term = -coef * (k1**2 + k2**2) * (2 * np.pi)**2 * np.sin(k1 * 2 * np.pi * self.X) * np.sin(k2 * 2 * np.pi * self.Y)
            f += f_term
        
        return theta, f, u
    
    def generate_boundary_conditions(self, u: np.ndarray) -> Dict[str, Callable]:
        """
        Generate boundary conditions from the analytical solution.
        
        Args:
            u: Analytical solution array
            
        Returns:
            Dict[str, callable]: Dictionary of boundary condition functions
        """
        # Create interpolation functions for the boundaries
        def bc_left(y):
            if isinstance(y, (float, int)):
                y_idx = int(y * (self.ny - 1))
                return float(u[y_idx, 0])
            else:
                y_idx = np.clip((y * (self.ny - 1)).astype(int), 0, self.ny - 1)
                return u[y_idx, 0]
        
        def bc_right(y):
            if isinstance(y, (float, int)):
                y_idx = int(y * (self.ny - 1))
                return float(u[y_idx, -1])
            else:
                y_idx = np.clip((y * (self.ny - 1)).astype(int), 0, self.ny - 1)
                return u[y_idx, -1]
        
        def bc_bottom(x):
            if isinstance(x, (float, int)):
                x_idx = int(x * (self.nx - 1))
                return float(u[0, x_idx])
            else:
                x_idx = np.clip((x * (self.nx - 1)).astype(int), 0, self.nx - 1)
                return u[0, x_idx]
        
        def bc_top(x):
            if isinstance(x, (float, int)):
                x_idx = int(x * (self.nx - 1))
                return float(u[-1, x_idx])
            else:
                x_idx = np.clip((x * (self.nx - 1)).astype(int), 0, self.nx - 1)
                return u[-1, x_idx]
        
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
            # Generate random problem with analytical solution
            theta, f, solution = self.generate_random_problem()
            
            # Generate boundary conditions from solution
            bc_dict = self.generate_boundary_conditions(solution)
            
            # Extract interface data
            interface_data = self.extract_interface_data(theta, f, solution, bc_dict)
            
            # Store problem data
            dataset.append({
                'theta': theta,
                'f': f,
                'solution': solution,
                'interface_data': interface_data,
                'bc_dict': bc_dict
            })
        
        return dataset 