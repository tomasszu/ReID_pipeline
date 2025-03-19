import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import sys
import cv2
import torch.nn as nn

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
model.classifier.add_block[2] = nn.Sequential()

#print(model)

# Get the last linear layer
last_layer = model.classifier.add_block[0]  # Change this depending on your model's classifier layer
weights = last_layer.weight.detach().cpu().numpy()  # Shape: (num_classes, 256)

# Compute importance scores for each feature position
feature_importance = abs(weights).mean(axis=1)  # Averaging across classes

print(feature_importance)

# # Find least important indices
# least_important_indices = feature_importance.argsort()[:50]  # Lowest 50 features

# print(least_important_indices)

feature_importance /= feature_importance.sum()
plt.plot(feature_importance)
plt.title("Feature Importance Distribution")
plt.show()

least_important_indices = feature_importance.argsort()[:50]  # 50 least useful features
print("Least important features:", least_important_indices)
