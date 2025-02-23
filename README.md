# ML-Predicted Interface Boundary Conditions for Domain Decomposition

This project implements a machine learning approach to predict interface boundary conditions when solving partial differential equations (PDEs) using domain decomposition. The focus is on the Poisson equation, but the methodology can be extended to other PDEs.

## Overview

The project uses domain decomposition to split a large PDE problem into smaller subdomains, making the solution process more efficient. Machine learning is used to predict the boundary conditions at the interfaces between subdomains, enabling parallel computation and reducing the overall computational cost.

### Key Features

- Implementation of the Poisson equation solver using finite differences
- Domain decomposition with ML-predicted interface conditions
- CNN-based model for predicting interface values
- Comprehensive data generation and training pipeline
- Evaluation tools and visualization capabilities

## Project Structure

```
.
├── README.md
├── requirements.txt
├── pde_solver.py        # PDE solver implementation
├── data_generator.py    # Training data generation
├── ml_model.py         # Neural network model
└── experiments.py      # Experiment runner and evaluation
```

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

The main experiment script can be run with:

```bash
python experiments.py
```

This will:
1. Generate training data
2. Train the ML model
3. Run a convergence study
4. Visualize results

### Custom Usage

You can also use individual components:

```python
from pde_solver import PoissonSolver
from data_generator import DataGenerator
from ml_model import InterfacePredictor
from experiments import Experiment

# Initialize experiment
exp = Experiment(nx=40, ny=40, n_subdomains=(2, 2))

# Train model
exp.train_model(n_samples=1000, n_epochs=100)

# Solve a problem with ML-predicted interface conditions
theta, f = exp.data_generator.generate_random_problem()
bc_dict = exp.data_generator.generate_boundary_conditions()
solution, error = exp.solve_with_ml_interface(theta, f, bc_dict)
```

## Implementation Details

### PDE Solver

- Implements both full-domain and subdomain solvers
- Uses Jacobi iteration for solving the Poisson equation
- Handles various boundary condition types

### Data Generator

- Generates random PDE problems for training
- Extracts interface data from full-domain solutions
- Provides utilities for boundary condition generation

### ML Model

- CNN-based architecture for interface prediction
- Processes local context around interfaces
- Trained using MSE loss
- Includes data preprocessing and batch preparation

### Experiments

- Provides tools for model evaluation
- Implements convergence studies
- Includes visualization utilities

## Results

The project demonstrates:
- Accurate prediction of interface boundary conditions
- Efficient domain decomposition
- Reduced computational cost compared to full-domain solving
- Good convergence properties

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project is based on research in domain decomposition methods and machine learning applications in scientific computing. 