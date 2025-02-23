import numpy as np
from typing import Tuple, List, Dict
from pde_solver import PoissonSolver

class DataGenerator:
    """
    Class for generating training data for the ML model by solving the full domain
    problem and extracting interface conditions.
    """
    
    def __init__(self, nx: int = 40, ny: int = 40, n_subdomains: Tuple[int, int] = (2, 2)):
        """
        Initialize the data generator.
        
        Args:
            nx (int): Number of points in x direction for full domain
            ny (int): Number of points in y direction for full domain
            n_subdomains (Tuple[int, int]): Number of subdomains in (x, y) directions
        """
        self.nx = nx
        self.ny = ny
        self.n_subdomains = n_subdomains
        
        # Calculate subdomain sizes
        self.subdomain_nx = nx // n_subdomains[0]
        self.subdomain_ny = ny // n_subdomains[1]
        
        # Initialize solvers
        self.full_solver = PoissonSolver(nx, ny)
        
        # Create coordinate grids
        self.x = np.linspace(0, 1, nx)
        self.y = np.linspace(0, 1, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
    
    def generate_random_problem(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random theta and f fields for the Poisson equation.
        The solution will be of the form: u(x,y) = sin(k₁*2πx)*sin(k₂*2πy)
        where k₁ and k₂ are random integers between 1 and 3.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Random theta and f fields
        """
        # Generate random wavenumbers (integers between 1 and 3)
        k1 = np.random.randint(1, 4)
        k2 = np.random.randint(1, 4)
        
        # Generate random diffusion coefficient (theta)
        # We'll keep theta close to 1 with small variations
        theta = 1.0 + 0.1 * np.random.rand(self.ny, self.nx)
        
        # Calculate the analytical solution
        u = np.sin(k1 * 2 * np.pi * self.X) * np.sin(k2 * 2 * np.pi * self.Y)
        
        # Calculate source term f = -∇·(θ∇u)
        # For u = sin(k₁*2πx)*sin(k₂*2πy), we need:
        # f = ((k₁*2π)² + (k₂*2π)²)*θ*u + ∇θ·∇u
        
        # Calculate ∇u
        du_dx = k1 * 2 * np.pi * np.cos(k1 * 2 * np.pi * self.X) * np.sin(k2 * 2 * np.pi * self.Y)
        du_dy = k2 * 2 * np.pi * np.sin(k1 * 2 * np.pi * self.X) * np.cos(k2 * 2 * np.pi * self.Y)
        
        # Calculate ∇θ
        dtheta_dx = np.gradient(theta, self.x[1] - self.x[0], axis=1)
        dtheta_dy = np.gradient(theta, self.y[1] - self.y[0], axis=0)
        
        # Calculate f
        laplacian_coef = -((k1 * 2 * np.pi)**2 + (k2 * 2 * np.pi)**2)
        f = laplacian_coef * theta * u - (dtheta_dx * du_dx + dtheta_dy * du_dy)
        
        return theta, f
    
    def generate_boundary_conditions(self) -> Dict[str, callable]:
        """
        Generate boundary conditions for the full domain based on the last generated problem.
        
        Returns:
            Dict[str, callable]: Dictionary of boundary condition functions
        """
        # Get the last generated k1 and k2 values
        k1 = np.random.randint(1, 4)  # We'll use new random values for each BC
        k2 = np.random.randint(1, 4)
        
        # Define boundary conditions using the analytical solution
        def bc_left(y): return np.sin(k2 * 2 * np.pi * y)  # x = 0
        def bc_right(y): return np.sin(k2 * 2 * np.pi * y)  # x = 1
        def bc_bottom(x): return np.sin(k1 * 2 * np.pi * x)  # y = 0
        def bc_top(x): return np.sin(k1 * 2 * np.pi * x)  # y = 1
        
        return {
            'left': bc_left,
            'right': bc_right,
            'bottom': bc_bottom,
            'top': bc_top
        }
    
    def extract_interface_data(self, solution: np.ndarray) -> List[Dict]:
        """
        Extract interface data from a full domain solution.
        
        Args:
            solution (np.ndarray): Full domain solution
            
        Returns:
            List[Dict]: List of dictionaries containing interface data and local context
        """
        interface_data = []
        
        # Extract vertical interfaces
        for i in range(1, self.n_subdomains[0]):
            x_idx = i * self.subdomain_nx
            for j in range(self.n_subdomains[1]):
                y_start = j * self.subdomain_ny
                y_end = (j + 1) * self.subdomain_ny
                
                # Extract local context (including interface)
                context_left = solution[y_start:y_end, x_idx-2:x_idx+1]
                context_right = solution[y_start:y_end, x_idx:x_idx+3]
                
                interface_data.append({
                    'type': 'vertical',
                    'position': (x_idx, y_start),
                    'interface_values': solution[y_start:y_end, x_idx],
                    'context_left': context_left,
                    'context_right': context_right
                })
        
        # Extract horizontal interfaces
        for j in range(1, self.n_subdomains[1]):
            y_idx = j * self.subdomain_ny
            for i in range(self.n_subdomains[0]):
                x_start = i * self.subdomain_nx
                x_end = (i + 1) * self.subdomain_nx
                
                # Extract local context (including interface)
                context_bottom = solution[y_idx-2:y_idx+1, x_start:x_end]
                context_top = solution[y_idx:y_idx+3, x_start:x_end]
                
                interface_data.append({
                    'type': 'horizontal',
                    'position': (x_start, y_idx),
                    'interface_values': solution[y_idx, x_start:x_end],
                    'context_bottom': context_bottom,
                    'context_top': context_top
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
            # Generate random problem
            theta, f = self.generate_random_problem()
            bc_dict = self.generate_boundary_conditions()
            
            # Solve full domain problem
            solution, _ = self.full_solver.solve_full_domain(
                theta, f,
                bc_dict['left'], bc_dict['right'],
                bc_dict['bottom'], bc_dict['top']
            )
            
            # Extract interface data
            interface_data = self.extract_interface_data(solution)
            
            # Store problem data
            dataset.append({
                'theta': theta,
                'f': f,
                'solution': solution,
                'interface_data': interface_data
            })
        
        return dataset 