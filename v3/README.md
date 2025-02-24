# Version 3: Low-K Numerical Solutions

This version focuses on low wavenumber cases with constant theta and numerical solutions.

## Components

- `pde_solver.py`: Implementation of the Poisson equation solver
- `data_generator_v3.py`: Data generator for low-k numerical solutions
- `train_model_v3.py`: Training script with integrated data generation
- `test_subdomain_solution_v3.py`: Test script for evaluating model performance
- `generate_datasets.py`: Script for generating and saving datasets
- `requirements.txt`: Required Python packages

## Features

- Constant theta=1 for all cases
- Focus on low wavenumber (k=2.0 to 4.5)
- Uses numerical solutions instead of analytical solutions
- Integrated data generation during training
- Comprehensive testing across different k values
- Comparison between ML-based and direct solutions

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the model (includes data generation):
   ```
   python train_model_v3.py
   ```

3. Generate datasets separately (optional):
   ```
   python generate_datasets.py
   ```

4. Test the model:
   ```
   python test_subdomain_solution_v3.py
   ```

## Results

The model shows good performance on low wavenumber cases:
- Similar or better MAE compared to direct solutions
- More consistent performance across different k values
- Reliable convergence even when direct solver struggles
- Effective interface prediction for constant theta cases 