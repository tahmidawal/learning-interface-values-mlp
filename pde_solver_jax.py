import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Callable
from functools import partial

class PoissonSolverJax:
    """
    A JAX-based solver for the Poisson equation: -∇·(θ∇u) = f
    Implements both full domain and subdomain solvers using finite differences.
    """
    
    def __init__(self, nx: int, ny: int, dx: float = 1.0, dy: float = 1.0):
        """
        Initialize the JAX-based Poisson solver.
        
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
        self.x = jnp.linspace(0, (nx-1)*dx, nx)
        self.y = jnp.linspace(0, (ny-1)*dy, ny)
        self.X, self.Y = jnp.meshgrid(self.x, self.y)

    @partial(jax.jit, static_argnums=(0,))
    def _jacobi_update(self, u: jnp.ndarray, u_old: jnp.ndarray, 
                      theta: jnp.ndarray, f: jnp.ndarray,
                      dx2: float, dy2: float) -> jnp.ndarray:
        """Single Jacobi iteration step implemented in a vectorized way."""
        # Compute theta averages using padding and slicing
        theta_padded = jnp.pad(theta, ((0,1), (0,1)))
        theta_avg_x = (theta_padded[:-1,1:] + theta_padded[:-1,:-1])/2
        theta_avg_y = (theta_padded[1:,:-1] + theta_padded[:-1,:-1])/2

        # Compute the update for interior points using explicit slicing
        # Define interior indices: 1:-1 for both rows and columns (shape: (n-2, n-2))
        u_right = u_old[1:-1, 2:]       # right neighbor: column shifted left
        u_left = u_old[1:-1, :-2]        # left neighbor: column shifted right
        u_up = u_old[2:, 1:-1]           # up neighbor: row shifted up
        u_down = u_old[:-2, 1:-1]        # down neighbor: row shifted down

        # Extract interior portions of theta averages corresponding to these shifts
        t_ax_center = theta_avg_x[1:-1, 1:-1]  # center coefficients for horizontal update
        t_ax_left = theta_avg_x[1:-1, :-2]       # left coefficients
        t_ay_center = theta_avg_y[1:-1, 1:-1]      # center coefficients for vertical update
        t_ay_top = theta_avg_y[:-2, 1:-1]         # top coefficients

        num_x = (t_ax_center * u_right + t_ax_left * u_left) / dx2
        num_y = (t_ay_center * u_up + t_ay_top * u_down) / dy2
        numerator = num_x + num_y + f[1:-1, 1:-1]

        denom_x = (t_ax_center + t_ax_left) / dx2
        denom_y = (t_ay_center + t_ay_top) / dy2
        denominator = denom_x + denom_y

        u_new_interior = numerator / denominator
        return u_old.at[1:-1, 1:-1].set(u_new_interior)

    @partial(jax.jit, static_argnums=(0,))
    def _solve_with_bc(self, 
                      theta: jnp.ndarray,
                      f: jnp.ndarray,
                      bc_left: jnp.ndarray,
                      bc_right: jnp.ndarray,
                      bc_bottom: jnp.ndarray,
                      bc_top: jnp.ndarray,
                      max_iter: int = 1000,
                      tol: float = 1e-6) -> Tuple[jnp.ndarray, int]:
        """JAX-optimized solver implementation."""
        dx2 = self.dx * self.dx
        dy2 = self.dy * self.dy
        
        # Initialize solution with boundary conditions
        u = jnp.zeros((self.ny, self.nx))
        u = u.at[:, 0].set(bc_left)
        u = u.at[:, -1].set(bc_right)
        u = u.at[0, :].set(bc_bottom)
        u = u.at[-1, :].set(bc_top)

        def cond_fun(state):
            i, u, u_old, _ = state
            return jnp.logical_and(
                i < max_iter,
                jnp.max(jnp.abs(u - u_old)) >= tol
            )

        def body_fun(state):
            i, u, _, _ = state
            u_new = self._jacobi_update(u, u, theta, f, dx2, dy2)
            return i + 1, u_new, u, jnp.max(jnp.abs(u_new - u))

        # Run the solver using jax.lax.while_loop
        init_state = (0, u, jnp.zeros_like(u), jnp.inf)
        final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
        
        return final_state[1], final_state[0]

    def solve_full_domain(self, 
                         theta: jnp.ndarray,
                         f: jnp.ndarray,
                         bc_left: jnp.ndarray,
                         bc_right: jnp.ndarray,
                         bc_bottom: jnp.ndarray,
                         bc_top: jnp.ndarray,
                         max_iter: int = 1000,
                         tol: float = 1e-6) -> Tuple[jnp.ndarray, int]:
        """
        Solve the Poisson equation on the full domain using JAX-optimized Jacobi iteration.
        
        Args:
            theta: Diffusion coefficient field
            f: Source term
            bc_left, bc_right, bc_bottom, bc_top: Boundary condition arrays
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            Tuple[jnp.ndarray, int]: Solution array and number of iterations
        """
        return self._solve_with_bc(theta, f, bc_left, bc_right, bc_bottom, bc_top, max_iter, tol)

    def solve_subdomain(self,
                       theta: jnp.ndarray,
                       f: jnp.ndarray,
                       bc_dict: Dict[str, jnp.ndarray],
                       max_iter: int = 1000,
                       tol: float = 1e-6) -> Tuple[jnp.ndarray, int]:
        """
        Solve the Poisson equation on a subdomain using JAX-optimized Jacobi iteration.
        
        Args:
            theta: Diffusion coefficient field for subdomain
            f: Source term for subdomain
            bc_dict: Dictionary containing boundary condition arrays
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            
        Returns:
            Tuple[jnp.ndarray, int]: Solution array and number of iterations
        """
        return self._solve_with_bc(
            theta, f,
            bc_dict['left'], bc_dict['right'],
            bc_dict['bottom'], bc_dict['top'],
            max_iter, tol
        )

# Vectorized solver for multiple subdomains
@partial(jax.jit, static_argnums=(0,))
def solve_multiple_subdomains(solver: PoissonSolverJax,
                            thetas: jnp.ndarray,
                            fs: jnp.ndarray,
                            bc_dicts: Dict[str, jnp.ndarray],
                            max_iter: int = 1000,
                            tol: float = 1e-6) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve multiple subdomains in parallel using JAX's vmap.
    
    Args:
        solver: PoissonSolverJax instance (static argument)
        thetas: Stacked theta arrays for each subdomain [n_domains, ny, nx]
        fs: Stacked f arrays for each subdomain [n_domains, ny, nx]
        bc_dicts: Dictionary with stacked boundary conditions [n_domains, n_points]
        max_iter: Maximum iterations
        tol: Convergence tolerance
    
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Stacked solutions and iteration counts
    """
    solve_batch = jax.vmap(solver._solve_with_bc, in_axes=(0, 0, 0, 0, 0, 0, None, None))
    solutions, iterations = solve_batch(
        thetas, fs,
        bc_dicts['left'], bc_dicts['right'],
        bc_dicts['bottom'], bc_dicts['top'],
        max_iter, tol
    )
    return solutions, iterations 