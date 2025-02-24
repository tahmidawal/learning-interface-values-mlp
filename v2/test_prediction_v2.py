import numpy as np
import torch
import matplotlib.pyplot as plt
from data_generator_v2 import DataGeneratorV2
from ml_model_v2 import InterfacePredictorV2
from test_subdomain_solution import generate_in_distribution_case

# Initialize data generator
print("Initializing data generator...")
data_gen = DataGeneratorV2(nx=40, ny=40, n_subdomains=(2, 2), patch_size=5)

# Load trained model
print("Loading trained model...")
model_path = 'models/interface_predictor_v2.pth'
model_info = torch.load(model_path)
predictor = InterfacePredictorV2(patch_size=5)
predictor.load_state_dict(model_info['model_state'])
predictor.eval()  # Set to evaluation mode

# Generate a test case
print("Generating test case...")
theta, f, u_exact, bc_dict = generate_in_distribution_case(data_gen)

# Extract a sample patch for prediction
print("Extracting sample patch...")
# Get a point at the vertical interface
x = data_gen.nx // 2  # Middle of the domain (vertical interface)
y = data_gen.ny // 2  # Middle of the domain
theta_patch = data_gen.extract_patch(theta, x, y)
f_patch = data_gen.extract_patch(f, x, y)
true_value = u_exact[y, x]

# Get boundary values for this point
boundary_values = np.array([
    bc_dict['left'](y / data_gen.ny),
    bc_dict['right'](y / data_gen.ny),
    bc_dict['bottom'](x / data_gen.nx),
    bc_dict['top'](x / data_gen.nx)
])

# Make a prediction
print("Making prediction...")
sample_point = {
    'position': (x, y),
    'type': 'vertical',
    'theta_patch': theta_patch,
    'f_patch': f_patch,
    'boundary_values': boundary_values,
    'target': true_value
}
predicted_value = predictor.predict(sample_point)

# Print results
print(f"\nResults for point at ({x}, {y}):")
print(f"True value: {true_value:.6f}")
print(f"Predicted value: {predicted_value:.6f}")
print(f"Absolute error: {abs(true_value - predicted_value):.6f}")

# Visualize the patch and prediction
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot theta patch
im0 = axes[0].imshow(theta_patch, cmap='viridis')
axes[0].set_title('Theta Patch')
plt.colorbar(im0, ax=axes[0])

# Plot f patch
im1 = axes[1].imshow(f_patch, cmap='viridis')
axes[1].set_title('f Patch')
plt.colorbar(im1, ax=axes[1])

# Plot exact solution around the point
solution_patch = data_gen.extract_patch(u_exact, x, y)
im2 = axes[2].imshow(solution_patch, cmap='viridis')
axes[2].set_title(f'Solution Patch\nTrue: {true_value:.4f}, Pred: {predicted_value:.4f}')
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig('v2/prediction_test_v2.png')
print("Visualization saved to 'v2/prediction_test_v2.png'") 