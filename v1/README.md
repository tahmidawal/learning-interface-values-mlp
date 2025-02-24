# Version 1: Basic Interface Prediction

This version implements the initial approach for learning interface conditions between subdomains.

## Components

- `pde_solver.py`: Implementation of the Poisson equation solver
- `train_interface_model.py`: Training script for the interface prediction model
- `test_interface_model.py`: Testing script for evaluating the trained model
- `requirements.txt`: Required Python packages

## Features

- Basic interface prediction between subdomains
- Uses analytical solutions for training data
- Simple neural network architecture
- No subdomain solution integration

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the model:
   ```
   python train_interface_model.py
   ```

3. Test the model:
   ```
   python test_interface_model.py
   ``` 