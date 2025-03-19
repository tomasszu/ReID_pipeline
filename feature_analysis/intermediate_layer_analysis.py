import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import pandas as pd
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

# Define a transform to preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load a random image from the dataframe
def load_random_image(df, img_root, image_column="imageName"):
    path = df.sample(1)[image_column].values[0]  # Pick random image path
    print("Selected image:", path)
    img = Image.open(os.path.join(img_root, path)).convert("RGB")  # Open image
    return img, path

# Hook to extract feature maps

def hook_fn(module, input, output):
    activations[layer_name] = output.detach()


# Forward pass the image
def get_feature_maps(img):
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        _ = model(img_tensor)  # Run through model to trigger hook
    return activations[layer_name]

def plot_feature_maps(feature_maps, num_maps=6):
    feature_maps = feature_maps.squeeze(0)  # Remove batch dimension
    num_maps = min(num_maps, feature_maps.shape[0])  # Limit number of maps
    
    fig, axes = plt.subplots(1, num_maps, figsize=(15, 5))
    for i in range(num_maps):
        axes[i].imshow(feature_maps[i].cpu().numpy(), cmap="viridis")
        axes[i].axis("off")
    plt.show()


def parse_vehicle_data(xml_file):
    """
    Parses an XML file containing vehicle data and returns a DataFrame.
    
    :param xml_file: Path to the XML file.
    :return: DataFrame with columns: ['imageName', 'vehicleID', 'cameraID', 'colorID', 'typeID']
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    data = []
    for item in root.find("Items"):
        data.append({
            "imageName": item.get("imageName"),
            "vehicleID": int(item.get("vehicleID")),
            "cameraID": item.get("cameraID"),
            "colorID": int(item.get("colorID")),
            "typeID": int(item.get("typeID"))
        })
    
    return pd.DataFrame(data)

def plot_image_and_features(img, feature_maps, num_maps=6, grid_size=(3, 5), resize=(224, 224)):
    resize_transform = transforms.Resize(resize)
    img_resized = resize_transform(img)

    feature_maps = feature_maps.squeeze(0)  # Remove batch dimension
    num_maps = min(num_maps, feature_maps.shape[0])  # Limit number of feature maps
    
    rows, cols = grid_size  # Rows and columns for the grid
    fig, axes = plt.subplots(rows, cols + 1, figsize=(15, 5))  # Extra column for the original image
    
    axes = axes.flatten()  # Flatten axes array for easier indexing
    
    # Show original image
    axes[0].imshow(img_resized)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Show feature maps
    for i in range(num_maps):
        axes[i + 1].imshow(feature_maps[i].cpu().numpy(), cmap="viridis")
        axes[i + 1].set_title(f"Feature {i+1}")
        axes[i + 1].axis("off")
    
    # Turn off any unused axes
    for i in range(num_maps + 1, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()

def plot_image_and_features_grayscale(img, feature_maps, num_maps=6, grid_size=(3, 5), resize=(224, 224)):
    resize_transform = transforms.Resize(resize)
    img_resized = resize_transform(img)

    feature_maps = feature_maps.squeeze(0)  # Remove batch dimension
    num_maps = min(num_maps, feature_maps.shape[0])  # Limit number of feature maps
    
    rows, cols = grid_size  # Rows and columns for the grid
    fig, axes = plt.subplots(rows, cols + 1, figsize=(15, 5))  # Extra column for the original image
    
    axes = axes.flatten()  # Flatten axes array for easier indexing
    
    # Show original image
    axes[0].imshow(img_resized)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Show feature maps as grayscale images (no colormap)
    for i in range(num_maps):
        axes[i + 1].imshow(feature_maps[i].cpu().numpy(), cmap="gray")
        axes[i + 1].set_title(f"Feature {i+1}")
        axes[i + 1].axis("off")
    
    # Turn off any unused axes
    for i in range(num_maps + 1, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()

activations = {}

# Choose an intermediate layer
layer_name = "layer3"  
layer = dict(model.model.named_children())[layer_name]

hook = layer.register_forward_hook(hook_fn)


img_root = '/home/tomass/tomass/data/VeRi/image_test'
df = parse_vehicle_data("/home/tomass/tomass/data/VeRi/test_label.xml")


img, path = load_random_image(df,img_root)

# plt.imshow(img); plt.title("Random Image"); plt.show()

feature_maps = get_feature_maps(img)

# plot_feature_maps(feature_maps)

# Plot both original image and feature maps 

#COLOR

# plot_image_and_features(img, feature_maps, num_maps=17)


#GRAYSCALE
plot_image_and_features_grayscale(img, feature_maps, num_maps=17)