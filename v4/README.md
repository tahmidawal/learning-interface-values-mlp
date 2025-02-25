# Version 4: Extended K-Range with Larger Dataset

This version extends the low-k numerical solutions approach with a wider range of wavenumbers and a larger training dataset.

## Components

- `pde_solver.py`: Implementation of the Poisson equation solver
- `data_generator_v4.py`: Data generator for numerical solutions with extended k-range
- `train_model_v4.py`: Training script with 1000 samples across wider k-range
- `test_subdomain_solution_v4.py`: Test script for evaluating model performance
- `test_interface_values_v4.py`: Script for visualizing interface values
- `requirements.txt`: Required Python packages

## Features

- Constant theta=1 for all cases
- Extended wavenumber range (k=1.0 to 6.0)
- Uses numerical solutions instead of analytical solutions
- Larger training dataset (1000 samples)
- Comprehensive testing across different k values
- Comparison between ML-based and direct solutions
- Interface value visualization

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the model (includes data generation):
   ```
   python train_model_v4.py
   ```

3. Test the model:
   ```
   python test_subdomain_solution_v4.py
   ```

4. Visualize interface values:
   ```
   python test_interface_values_v4.py
   ```

## Improvements over v3

- Wider range of wavenumbers for better generalization
- Larger training dataset for improved model accuracy
- Enhanced testing and visualization capabilities
- Better handling of interface conditions 