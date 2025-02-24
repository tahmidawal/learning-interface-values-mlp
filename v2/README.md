# Version 2: Subdomain Integration

This version extends the basic interface prediction model to integrate with subdomain solutions.

## Components

- `pde_solver.py`: Implementation of the Poisson equation solver
- `data_generator_v2.py`: Data generator for creating training and test cases
- `ml_model_v2.py`: Enhanced ML model for interface prediction
- `test_subdomain_solution.py`: Test script for evaluating subdomain solutions
- `test_subdomain_solution_low_k.py`: Test script for low wavenumber cases
- `visualize_training_data.py`: Script for visualizing training data
- `requirements.txt`: Required Python packages

## Features

- Enhanced data generation with analytical solutions
- Variable theta (diffusion coefficient)
- Integration with subdomain solutions
- Comparison with direct solutions
- Support for in-distribution and out-of-distribution testing
- Visualization tools for training data and results

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Visualize training data:
   ```
   python visualize_training_data.py
   ```

3. Test subdomain solutions:
   ```
   python test_subdomain_solution.py
   ```

4. Test low wavenumber cases:
   ```
   python test_subdomain_solution_low_k.py
   ``` 