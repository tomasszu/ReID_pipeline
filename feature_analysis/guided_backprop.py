import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  # Get the directory above

sys.path.append(SCRIPT_DIR)  # Add current script's directory
sys.path.append(PARENT_DIR)  # Add parent directory

# Load your custom model
from vehicle_reid_repo2.vehicle_reid.load_model import load_model_from_opts

device = "cuda"
model = load_model_from_opts(
    "/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/opts.yaml",
    ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.pth",
    remove_classifier=True
).to(device)
model.eval()

# Load an actual image
image_path = "/home/tomass/tomass/data/VeRi/image_test/0002_c002_00030625_1.jpg"  # <-- Replace with your image path
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img = Image.open(image_path).convert("RGB")
img = transform(img).unsqueeze(0).to(device)  # Add batch dimension

img.requires_grad = True  # Enable gradients for backpropagation

# Forward pass
output = model(img)
target_class = output.argmax(dim=1).item()  # Pick the highest scoring class

# Backward pass
model.zero_grad()
output[0, target_class].backward()

# Get the gradients
gradients = img.grad.detach().cpu().numpy().squeeze().transpose(1, 2, 0)  # (H, W, C)

# Normalize gradients using absolute values
gradients = np.abs(gradients)

# Normalize between 0 and 1
gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)

# Convert to grayscale
grayscale_grad = np.mean(gradients, axis=-1)

# Rescale grayscale to 0-255
grayscale_grad = (grayscale_grad * 255).astype(np.uint8)

# Apply contrast enhancement
grayscale_grad = cv2.equalizeHist(grayscale_grad)

# Show results
plt.imshow(grayscale_grad, cmap="gray")
plt.title("Guided Backpropagation (Enhanced)")
plt.axis("off")
plt.show()
