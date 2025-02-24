# Learning Interface Conditions for Subdomain Decomposition

This project explores machine learning approaches for predicting interface conditions between subdomains in partial differential equation (PDE) solvers. The goal is to accelerate PDE solutions by using ML to predict interface conditions between subdomains.

## Project Structure

The project is organized into three versions, each representing a different approach:

### [Version 1](v1/): Basic Interface Prediction

- Initial implementation of interface prediction
- Uses analytical solutions for training
- Simple neural network architecture
- No subdomain solution integration

### [Version 2](v2/): Subdomain Integration

- Enhanced data generation with analytical solutions
- Variable theta (diffusion coefficient)
- Integration with subdomain solutions
- Comparison with direct solutions
- Support for in-distribution and out-of-distribution testing

### [Version 3](v3/): Low-K Numerical Solutions

- Constant theta=1 for all cases
- Focus on low wavenumber (k=2.0 to 4.5)
- Uses numerical solutions instead of analytical solutions
- Integrated data generation during training
- Comprehensive testing across different k values

## Common Components

- `pde_solver.py`: Implementation of the Poisson equation solver
- Data generators: Create training and test data
- ML models: Neural networks for interface prediction
- Test scripts: Evaluate model performance

## Key Findings

1. ML-based interface prediction can achieve similar or better accuracy compared to direct solutions
2. The subdomain approach shows more consistent performance across different problem parameters
3. ML solutions typically converge more reliably than direct solvers in challenging cases
4. The approach is particularly effective for low wavenumber cases with constant diffusion coefficients

## Getting Started

Each version directory contains its own README with specific instructions. In general:

1. Navigate to the desired version directory
2. Install dependencies: `pip install -r requirements.txt`
3. Run the training script
4. Run the test script to evaluate performance

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- Seaborn (for visualization)

## License

[MIT License](LICENSE) 