import numpy as np
import time
from mpi4py import MPI
from typing import Dict, Tuple, List
import sys

from pde_solver import PoissonSolver
from experiments import Experiment

class BoundaryCondition:
    def __init__(self, values):
        self.values = values
    
    def __call__(self, x):
        return float(self.values[int(x)])

def solve_subdomain(solver: PoissonSolver, theta: np.ndarray, f: np.ndarray, bc_dict: Dict) -> Tuple[np.ndarray, int]:
    """Solve a single subdomain."""
    return solver.solve_subdomain(theta, f, bc_dict)

def combine_solutions(solutions: List[Tuple[Tuple[np.ndarray, int], Tuple[int, int]]], shape: Tuple[int, int]) -> np.ndarray:
    """Combine subdomain solutions into full domain solution."""
    full_solution = np.zeros(shape)
    for (solution_tuple, position) in solutions:
        sol, _ = solution_tuple
        y_start, x_start = position
        ny, nx = sol.shape
        full_solution[y_start:y_start+ny, x_start:x_start+nx] = sol
    return full_solution

def create_boundary_array(bc_func, size: int) -> np.ndarray:
    """Convert boundary function to array of values."""
    return np.array([bc_func(i) for i in range(size)])

def solve_pde_ml_parallel(theta: np.ndarray, f: np.ndarray, bc_arrays: Dict[str, np.ndarray], exp: Experiment = None) -> Tuple[np.ndarray, float, float]:
    """
    Solve PDE using ML for interface prediction and MPI for parallel subdomain solving.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < 5:
        print(f"Error: Need at least 5 MPI processes (1 master + 4 workers), but got {size}")
        comm.Abort(1)
        return None, 0, 0
    
    if rank == 0:
        # Master process
        if exp is None:
            exp = Experiment(nx=40, ny=40, n_subdomains=(2, 2))
        
        # ML inference for interface conditions
        start_ml = time.time()
        interface_data = exp.data_generator.extract_interface_data(np.zeros((40, 40)))
        interface_predictions = {}
        for data in interface_data:
            pred_values = exp.predictor.predict(data)
            key = (data['position'], data['type'])
            interface_predictions[key] = pred_values
        ml_time = time.time() - start_ml
        
        # Prepare subdomain problems
        subdomain_tasks = []
        positions = []
        
        for i in range(2):
            for j in range(2):
                x_start = j * 20
                x_end = (j + 1) * 20
                y_start = i * 20
                y_end = (i + 1) * 20
                
                theta_sub = theta[y_start:y_end, x_start:x_end].copy()
                f_sub = f[y_start:y_end, x_start:x_end].copy()
                
                # Prepare boundary conditions for subdomain
                bc_sub = {}
                
                # Left boundary
                if j == 0:
                    bc_values = bc_arrays['left'][y_start:y_end].copy()
                    bc_sub['left'] = BoundaryCondition(bc_values)
                else:
                    key = ((x_start, y_start), 'vertical')
                    values = interface_predictions[key].copy()
                    bc_sub['left'] = BoundaryCondition(values)
                
                # Right boundary
                if j == 1:
                    bc_values = bc_arrays['right'][y_start:y_end].copy()
                    bc_sub['right'] = BoundaryCondition(bc_values)
                else:
                    key = ((x_end, y_start), 'vertical')
                    values = interface_predictions[key].copy()
                    bc_sub['right'] = BoundaryCondition(values)
                
                # Bottom boundary
                if i == 0:
                    bc_values = bc_arrays['bottom'][x_start:x_end].copy()
                    bc_sub['bottom'] = BoundaryCondition(bc_values)
                else:
                    key = ((x_start, y_start), 'horizontal')
                    values = interface_predictions[key].copy()
                    bc_sub['bottom'] = BoundaryCondition(values)
                
                # Top boundary
                if i == 1:
                    bc_values = bc_arrays['top'][x_start:x_end].copy()
                    bc_sub['top'] = BoundaryCondition(bc_values)
                else:
                    key = ((x_start, y_end), 'horizontal')
                    values = interface_predictions[key].copy()
                    bc_sub['top'] = BoundaryCondition(values)
                
                subdomain_tasks.append((exp.subdomain_solvers[i][j], theta_sub, f_sub, bc_sub))
                positions.append((y_start, x_start))
        
        # Distribute tasks to workers
        start_solve = time.time()
        for i, task in enumerate(subdomain_tasks):
            worker_rank = i + 1
            comm.send((task, positions[i]), dest=worker_rank, tag=0)
        
        # Collect results
        solutions = []
        for i in range(len(subdomain_tasks)):
            worker_rank = i + 1
            result = comm.recv(source=worker_rank, tag=0)
            solutions.append((result, positions[i]))
        
        solve_time = time.time() - start_solve
        
        # Combine solutions
        solution = combine_solutions(solutions, (40, 40))
        return solution, ml_time, solve_time
    
    elif rank <= 4:
        # Worker processes
        task, position = comm.recv(source=0, tag=0)
        result = solve_subdomain(*task)
        comm.send(result, dest=0, tag=0)
        return None, 0, 0
    
    else:
        # Extra processes do nothing
        return None, 0, 0

if __name__ == "__main__":
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Only rank 0 initializes the problem
    if rank == 0:
        print(f"Initializing with {comm.Get_size()} processes")
        exp = Experiment()
        
        # Generate random problem
        theta, f = exp.data_generator.generate_random_problem()
        bc_dict = exp.data_generator.generate_boundary_conditions()
        
        # Convert boundary conditions to arrays
        nx, ny = 40, 40
        bc_arrays = {
            'left': create_boundary_array(bc_dict['left'], ny),
            'right': create_boundary_array(bc_dict['right'], ny),
            'bottom': create_boundary_array(bc_dict['bottom'], nx),
            'top': create_boundary_array(bc_dict['top'], nx)
        }
        
        # Get reference solution
        print("Computing reference solution...")
        u_ref, _ = exp.full_solver.solve_full_domain(
            theta, f,
            bc_dict['left'], bc_dict['right'],
            bc_dict['bottom'], bc_dict['top']
        )
    else:
        exp = None
        theta = None
        f = None
        bc_arrays = None
        u_ref = None
    
    # Broadcast problem data
    if rank == 0:
        print("Broadcasting problem data...")
    theta = comm.bcast(theta, root=0)
    f = comm.bcast(f, root=0)
    bc_arrays = comm.bcast(bc_arrays, root=0)
    
    # Solve using ML and parallel computation
    if rank == 0:
        print("Starting parallel solution...")
    solution, ml_time, solve_time = solve_pde_ml_parallel(theta, f, bc_arrays, exp)
    
    # Only rank 0 processes results
    if rank == 0:
        print(f"\nResults:")
        print(f"ML prediction time: {ml_time:.6f} seconds")
        print(f"Parallel solve time: {solve_time:.6f} seconds")
        print(f"Total time: {ml_time + solve_time:.6f} seconds")
        
        if solution is not None:
            error = np.abs(solution - u_ref)
            print(f"\nError metrics:")
            print(f"Mean absolute error: {np.mean(error):.6e}")
            print(f"Max absolute error: {np.max(error):.6e}")
            print(f"Relative L2 error: {np.sqrt(np.sum(error**2)) / np.sqrt(np.sum(u_ref**2)):.6e}")
    
    # Clean exit
    comm.Barrier()
    if rank == 0:
        print("\nDone!") 