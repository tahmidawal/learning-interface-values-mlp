import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class DataGeneratorV3:
    def __init__(self, nx: int = 40, ny: int = 40, n_subdomains: Tuple[int, int] = (2, 2),
                 patch_size: int = 5):
        self.nx = nx
        self.ny = ny
        self.n_subdomains = n_subdomains
        self.patch_size = patch_size
        
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        self.X, self.Y = np.meshgrid(x, y)
        
        self.k_values = np.arange(2.0, 5.0, 0.5)
        
        self.scale_factors = {
            "theta": {"min": 1.0, "max": 1.0},
            "solution": {"min": -1.0, "max": 1.0}
        }
        
        self.subdomain_nx = nx // n_subdomains[1]
        self.subdomain_ny = ny // n_subdomains[0]
        
        if patch_size % 2 == 0:
            raise ValueError("patch_size must be odd")
        if patch_size > min(self.subdomain_nx, self.subdomain_ny):
            raise ValueError("patch_size too large for subdomain size")

    def scale_array(self, arr: np.ndarray, min_val: Optional[float] = None, 
                   max_val: Optional[float] = None) -> np.ndarray:
        if min_val is None:
            min_val = self.scale_factors["solution"]["min"]
        if max_val is None:
            max_val = self.scale_factors["solution"]["max"]
            
        arr_min, arr_max = arr.min(), arr.max()
        if arr_min == arr_max:
            return np.ones_like(arr) * min_val
        
        return (arr - arr_min) * (max_val - min_val) / (arr_max - arr_min) + min_val

    def generate_random_problem(self):
        k1 = np.random.choice(self.k_values)
        k2 = np.random.choice(self.k_values)
        
        coef = np.random.uniform(0.1, 0.2)
        
        u = coef * np.sin(k1 * 2 * np.pi * self.X) * np.sin(k2 * 2 * np.pi * self.Y)
        f = coef * np.sin(k1 * 2 * np.pi * self.X) * np.sin(k2 * 2 * np.pi * self.Y)
        
        theta = np.ones((self.ny, self.nx))
        
        u_scaled = self.scale_array(u)
        f_scaled = self.scale_array(f)
        
        return theta, -f_scaled, u_scaled

    def generate_boundary_conditions(self, u: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "left": u[:, 0],
            "right": u[:, -1],
            "bottom": u[0, :],
            "top": u[-1, :]
        }

    def extract_patch(self, field: np.ndarray, center_x: int, center_y: int) -> np.ndarray:
        half_size = self.patch_size // 2
        patch = field[center_y - half_size:center_y + half_size + 1,
                     center_x - half_size:center_x + half_size + 1]
        return patch

    def extract_interface_data(self, theta: np.ndarray, f: np.ndarray, u: np.ndarray,
                             bc_dict: Dict[str, np.ndarray]) -> List[Dict]:
        interface_data = []
        
        # Vertical interfaces
        for j in range(1, self.n_subdomains[1]):
            x = j * self.subdomain_nx
            for y in range(self.patch_size//2, self.ny - self.patch_size//2):
                theta_patch = self.extract_patch(theta, x, y)
                f_patch = self.extract_patch(f, x, y)
                target = u[y, x]  # Interface value
                
                interface_data.append({
                    "position": (x, y),
                    "type": "vertical",
                    "theta_patch": theta_patch,
                    "f_patch": f_patch,
                    "target": target
                })
        
        # Horizontal interfaces
        for i in range(1, self.n_subdomains[0]):
            y = i * self.subdomain_ny
            for x in range(self.patch_size//2, self.nx - self.patch_size//2):
                theta_patch = self.extract_patch(theta, x, y)
                f_patch = self.extract_patch(f, x, y)
                target = u[y, x]  # Interface value
                
                interface_data.append({
                    "position": (x, y),
                    "type": "horizontal",
                    "theta_patch": theta_patch,
                    "f_patch": f_patch,
                    "target": target
                })
        
        return interface_data

if __name__ == "__main__":
    # Test the data generator
    data_gen = DataGeneratorV3()
    theta, f, u = data_gen.generate_random_problem()
    bc_dict = data_gen.generate_boundary_conditions(u)
    interface_data = data_gen.extract_interface_data(theta, f, u, bc_dict)
    
    # Plot the generated fields
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(theta, origin="lower")
    axes[0].set_title("Theta (Constant=1)")
    
    axes[1].imshow(f, origin="lower")
    axes[1].set_title("Source Term f")
    
    axes[2].imshow(u, origin="lower")
    axes[2].set_title("Solution u")
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"Number of interface points: {len(interface_data)}")
    print(f"k values available: {data_gen.k_values}")
    print(f"Solution range: [{u.min():.4f}, {u.max():.4f}]")
    print(f"f range: [{f.min():.4f}, {f.max():.4f}]")
