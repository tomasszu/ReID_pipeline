import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  # Get the directory above

sys.path.append(SCRIPT_DIR)  # Add current script's directory
sys.path.append(PARENT_DIR)  # Add parent directory

from vehicle_reid_repo2.vehicle_reid.load_model import load_model_from_opts
from counting_workspace.misc.feature_extract_AICity import extract_feature

device = "cuda"
model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.pth", remove_classifier=True)
model.eval()  # Set to evaluation mode

# Select a convolutional layer to visualize (e.g., layer 4)
layer = model.model.layer3[2].conv3  # Choose a conv layer inside layer 4

# print(model)

# Register a forward hook to capture activations
activations = {}

def hook_fn(module, input, output):
    activations["layer"] = output

hook = layer.register_forward_hook(hook_fn)

# Create a random input image (starting point for optimization)
input_img = torch.randn(1, 3, 224, 224, requires_grad=True, device="cpu")

# Optimizer to modify input image
optimizer = torch.optim.Adam([input_img], lr=0.1)

# Optimization loop
for i in range(100):
    optimizer.zero_grad()
    
    model(input_img)  # Forward pass
    loss = -activations["layer"].norm()  # Maximize activation
    loss.backward()
    
    optimizer.step()

    # Optional: Clamp values to keep image within valid range
    with torch.no_grad():
        input_img.clamp_(0, 1)

# Show the optimized image (what the neuron "sees" as important)
plt.imshow(input_img.squeeze().permute(1, 2, 0).detach().numpy())
plt.title("Optimized Input (AlexNet Approach)")
plt.axis("off")
plt.show()