import numpy as np
from typing import Tuple, Optional, Callable

class PoissonSolver:
    """
    A class to solve the Poisson equation: -∇·(θ∇u) = f
    Implements both full domain and subdomain solvers using finite differences.
    All input and output values are expected to be scaled between -1 and 1.
    """
    
    def __init__(self, nx: int, ny: int, dx: float = 1.0, dy: float = 1.0):
        """
        Initialize the Poisson solver.
        
        Args:
            nx (int): Number of points in x direction
            ny (int): Number of points in y direction
            dx (float): Grid spacing in x direction
            dy (float): Grid spacing in y direction
        """
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        
        # Initialize grid
        self.x = np.linspace(0, (nx-1)*dx, nx)
        self.y = np.linspace(0, (ny-1)*dy, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
    
    def solve_full_domain(self, 
                         theta: np.ndarray,
                         f: np.ndarray,
                         bc_left: Callable[[np.ndarray], np.ndarray],
                         bc_right: Callable[[np.ndarray], np.ndarray],
                         bc_bottom: Callable[[np.ndarray], np.ndarray],
                         bc_top: Callable[[np.ndarray], np.ndarray],
                         max_iter: int = 1000,
                         tol: float = 1e-6) -> Tuple[np.ndarray, int]:
        """
        Solve the Poisson equation on the full domain using Jacobi iteration.
        All input and output values are expected to be scaled between -1 and 1.
        
        Args:
            theta (np.ndarray): Diffusion coefficient field (scaled)
            f (np.ndarray): Source term (scaled)
            bc_left, bc_right, bc_bottom, bc_top: Boundary condition functions (return scaled values)
            max_iter (int): Maximum number of iterations
            tol (float): Convergence tolerance
            
        Returns:
            Tuple[np.ndarray, int]: Solution array (scaled) and number of iterations
        """
        # Initialize solution with scaled boundary conditions
        u = np.zeros((self.ny, self.nx))
        
        # Apply initial boundary conditions (already scaled)
        u[:, 0] = bc_left(self.y)
        u[:, -1] = bc_right(self.y)
        u[0, :] = bc_bottom(self.x)
        u[-1, :] = bc_top(self.x)
        
        # Iteration coefficients
        dx2 = self.dx * self.dx
        dy2 = self.dy * self.dy
        
        # Jacobi iteration
        for iter_count in range(max_iter):
            u_old = u.copy()
            
            # Update interior points
            for i in range(1, self.ny-1):
                for j in range(1, self.nx-1):
                    theta_avg_x = (theta[i,j] + theta[i,j+1])/2
                    theta_avg_x_m = (theta[i,j] + theta[i,j-1])/2
                    theta_avg_y = (theta[i,j] + theta[i+1,j])/2
                    theta_avg_y_m = (theta[i,j] + theta[i-1,j])/2
                    
                    # Scale the coefficients to maintain solution scale
                    scale = (theta_avg_x + theta_avg_x_m)/dx2 + (theta_avg_y + theta_avg_y_m)/dy2
                    
                    u[i,j] = (
                        (theta_avg_x * u_old[i,j+1] + theta_avg_x_m * u_old[i,j-1])/dx2 +
                        (theta_avg_y * u_old[i+1,j] + theta_avg_y_m * u_old[i-1,j])/dy2 +
                        f[i,j]
                    ) / scale
            
            # Check convergence
            if np.max(np.abs(u - u_old)) < tol:
                return u, iter_count + 1
                
        return u, max_iter
    
    def solve_subdomain(self,
                       theta: np.ndarray,
                       f: np.ndarray,
                       bc_dict: dict,
                       max_iter: int = 1000,
                       tol: float = 1e-6) -> Tuple[np.ndarray, int]:
        """
        Solve the Poisson equation on a subdomain using Jacobi iteration.
        All input and output values are expected to be scaled between -1 and 1.
        
        Args:
            theta (np.ndarray): Diffusion coefficient field for subdomain (scaled)
            f (np.ndarray): Source term for subdomain (scaled)
            bc_dict (dict): Dictionary containing boundary condition functions (return scaled values)
            max_iter (int): Maximum number of iterations
            tol (float): Convergence tolerance
            
        Returns:
            Tuple[np.ndarray, int]: Solution array (scaled) and number of iterations
        """
        # Initialize solution
        u = np.zeros((self.ny, self.nx))
        
        # Create coordinate arrays for boundary conditions
        y_coords = np.arange(self.ny)
        x_coords = np.arange(self.nx)
        
        # Apply initial boundary conditions (already scaled)
        u[:, 0] = np.array([bc_dict['left'](y) for y in y_coords])  # Left boundary
        u[:, -1] = np.array([bc_dict['right'](y) for y in y_coords])  # Right boundary
        u[0, :] = np.array([bc_dict['bottom'](x) for x in x_coords])  # Bottom boundary
        u[-1, :] = np.array([bc_dict['top'](x) for x in x_coords])  # Top boundary
        
        # Iteration coefficients
        dx2 = self.dx * self.dx
        dy2 = self.dy * self.dy
        
        # Jacobi iteration
        for iter_count in range(max_iter):
            u_old = u.copy()
            
            # Update interior points
            for i in range(1, self.ny-1):
                for j in range(1, self.nx-1):
                    theta_avg_x = (theta[i,j] + theta[i,j+1])/2
                    theta_avg_x_m = (theta[i,j] + theta[i,j-1])/2
                    theta_avg_y = (theta[i,j] + theta[i+1,j])/2
                    theta_avg_y_m = (theta[i,j] + theta[i-1,j])/2
                    
                    # Scale the coefficients to maintain solution scale
                    scale = (theta_avg_x + theta_avg_x_m)/dx2 + (theta_avg_y + theta_avg_y_m)/dy2
                    
                    u[i,j] = (
                        (theta_avg_x * u_old[i,j+1] + theta_avg_x_m * u_old[i,j-1])/dx2 +
                        (theta_avg_y * u_old[i+1,j] + theta_avg_y_m * u_old[i-1,j])/dy2 +
                        f[i,j]
                    ) / scale
            
            # Check convergence
            if np.max(np.abs(u - u_old)) < tol:
                return u, iter_count + 1
                
        return u, max_iter 