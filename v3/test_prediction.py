import numpy as np
import torch
import matplotlib.pyplot as plt
from data_generator_v3 import DataGeneratorV3
from test_subdomain_solution_v3 import InterfacePredictor, generate_test_case

# Initialize data generator
print("Initializing data generator...")
data_gen = DataGeneratorV3(nx=40, ny=40, n_subdomains=(2, 2), patch_size=5)

# Load trained model
print("Loading trained model...")
model_path = 'models/interface_predictor_v3.pth'
model_info = torch.load(model_path)
predictor = InterfacePredictor(patch_size=5)
predictor.load_state_dict(model_info['model_state'])
predictor.eval()  # Set to evaluation mode

# Generate a test case
print("Generating test case...")
k = 3.0  # Use k=3.0 for this test
theta, f, u_exact = generate_test_case(data_gen, k)

# Extract a sample patch for prediction
print("Extracting sample patch...")
# Get a point at the vertical interface
x = data_gen.nx // 2  # Middle of the domain (vertical interface)
y = data_gen.ny // 2  # Middle of the domain
theta_patch = data_gen.extract_patch(theta, x, y)
f_patch = data_gen.extract_patch(f, x, y)
true_value = u_exact[y, x]

# Make a prediction
print("Making prediction...")
sample_point = {
    'position': (x, y),
    'type': 'vertical',
    'theta_patch': theta_patch,
    'f_patch': f_patch,
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
plt.savefig('prediction_test.png')
print("Visualization saved to 'prediction_test.png'") 