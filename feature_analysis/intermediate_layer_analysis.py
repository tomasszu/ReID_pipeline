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
# model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.pth", remove_classifier=True)
model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/Pidgeon_model_3_split_ids/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/Pidgeon_model_3_split_ids/net_12.pth", remove_classifier=True)


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

def plot_feature_maps(feature_maps, num_maps=8):
    feature_maps = feature_maps.squeeze(0)  # Remove batch dimension
    num_maps = min(num_maps, feature_maps.shape[0])  # Limit number of maps
    
    fig, axes = plt.subplots(1, num_maps, figsize=(5, 3))
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

def plot_image_and_features(img, feature_maps, num_maps=17, grid_size=(3, 4), resize=(224, 224)):
    resize_transform = transforms.Resize(resize)
    img_resized = resize_transform(img)

    feature_maps = feature_maps.squeeze(0)  # Remove batch dimension
    num_maps = min(num_maps, feature_maps.shape[0])  # Limit number of feature maps
    
    rows, cols = grid_size  # Rows and columns for the grid
    fig, axes = plt.subplots(rows, cols + 1, figsize=(3, 3))  # Extra column for the original image
    
    axes = axes.flatten()  # Flatten axes array for easier indexing
    
    # Show original image
    axes[0].imshow(img_resized)
    #axes[0].set_title("Orģī")
    axes[0].axis("off")
    
    # Show feature maps
    for i in range(num_maps):
        axes[i + 1].imshow(feature_maps[i].cpu().numpy(), cmap="viridis")
        #axes[i + 1].set_title(f"Feature {i+1}")
        axes[i + 1].axis("off")
    
    # Turn off any unused axes
    for i in range(num_maps + 1, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.show()

def plot_image_and_features_grayscale(img, feature_maps, num_maps=17, grid_size=(3, 5), resize=(224, 224)):
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

import torch
import matplotlib.pyplot as plt
from torchvision import transforms

def plot_image_and_features_based_on_variance_color(img, feature_maps, variance_threshold=0.2, resize=(224, 224)):
    resize_transform = transforms.Resize(resize)
    img_resized = resize_transform(img)

    feature_maps = feature_maps.squeeze(0)  # Remove batch dimension
    num_maps = feature_maps.shape[0]

    # Compute the variance for each feature map
    variances = [torch.var(feature_map).item() for feature_map in feature_maps]
    
    # Filter feature maps based on variance threshold
    selected_maps = [i for i, var in enumerate(variances) if var >= variance_threshold]
    
    if not selected_maps:
        print("No feature maps with variance above threshold.")
        return
    
    # Normalize the variances to [0, 1] range for color mapping
    variance_tensor = torch.tensor(variances)
    normalized_variance = (variance_tensor - variance_tensor.min()) / (variance_tensor.max() - variance_tensor.min())

    # Create grid for the selected maps
    rows = (len(selected_maps) // 8) + (1 if len(selected_maps) % 8 != 0 else 0)
    cols = 8

    # Create figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 2))
    axes = axes.flatten()

    # Show original image
    axes[0].imshow(img_resized)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Show selected feature maps with hue-based color map (e.g., "hsv" or "hls")
    for i, idx in enumerate(selected_maps):
        feature_map = feature_maps[idx].cpu().numpy()

        # Apply variance as hue
        cmap = plt.cm.hsv  # You can try other color maps like "hls", "jet", etc.
        normed_map = normalized_variance[idx]  # Get the normalized variance for this feature map
        colored_map = cmap(normed_map)  # Apply the colormap

        # Plot with colored variance heatmap (based on hue)
        axes[i + 1].imshow(feature_map, cmap=cmap)
        axes[i + 1].set_title(f"Feature {idx + 1} (Var: {variances[idx]:.4f})")
        axes[i + 1].axis("off")

    # Turn off any unused axes
    for i in range(len(selected_maps) + 1, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def plot_image_and_features_variance_heatmap(img, feature_maps, resize=(224, 224)):
    resize_transform = transforms.Resize(resize)
    img_resized = resize_transform(img)

    feature_maps = feature_maps.squeeze(0)  # Remove batch dimension
    num_maps = feature_maps.shape[0]

    # Compute the variance for each feature map
    variances = [torch.var(feature_map).item() for feature_map in feature_maps]

    # Normalize the variance values for better visualization
    variance_tensor = torch.tensor(variances)
    normalized_variance = (variance_tensor - variance_tensor.min()) / (variance_tensor.max() - variance_tensor.min())

    # Create a heatmap of the variances
    heatmap = normalized_variance.view(1, -1).cpu().numpy()  # Reshape to 1 x num_maps for heatmap

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(15, 1))
    
    # Plot the heatmap (color-coded variance)
    cax = ax.imshow(heatmap, cmap="hot", aspect="auto", interpolation="nearest")
    fig.colorbar(cax, ax=ax, orientation='horizontal', label='Variance')

    ax.set_title("Variance of Feature Maps")
    ax.axis("off")

    # Show the image and the heatmap side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(img_resized)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="hot", aspect="auto", interpolation="nearest")
    axes[1].set_title("Variance Heatmap")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    # Optional: If you want to show the actual variance values overlaid as text on the heatmap
    fig, ax = plt.subplots(figsize=(15, 2))
    cax = ax.imshow(heatmap, cmap="hot", aspect="auto", interpolation="nearest")
    fig.colorbar(cax, ax=ax, orientation='horizontal', label='Variance')

    for i, var in enumerate(variances):
        ax.text(i, 0, f'{var:.4f}', ha='center', va='center', color='white', fontsize=8)

    ax.set_title("Variance Heatmap with Actual Values")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

def plot_image_and_features_with_pagination(img, feature_maps, variance_threshold=0.005, num_maps=19, grid_size=(4, 5), resize=(224, 224), page=0):
    resize_transform = transforms.Resize(resize)
    img_resized = resize_transform(img)

    feature_maps = feature_maps.squeeze(0)  # Remove batch dimension
    num_maps = min(num_maps, feature_maps.shape[0])  # Limit number of feature maps
    
    # Compute the variance for each feature map
    variances = [torch.var(feature_map).item() for feature_map in feature_maps]
    
    # Filter feature maps based on variance threshold
    selected_maps = [i for i, var in enumerate(variances) if var >= variance_threshold]
    
    if not selected_maps:
        print("No feature maps with variance above threshold.")
        return
    
    # Define pagination
    maps_per_page = grid_size[0] * grid_size[1] - 1  # All spots except the first one for the image
    start_idx = page * maps_per_page  # Start index for the current page
    end_idx = start_idx + maps_per_page  # End index for the current page
    
    # Select the feature maps for the current page
    selected_maps_page = selected_maps[start_idx:end_idx]
    
    # If we are on the last page and there are fewer maps remaining than the grid can hold, adjust the grid size
    if len(selected_maps_page) < maps_per_page:
        grid_size = (len(selected_maps_page) // grid_size[1] + 1, grid_size[1])
    
    # Create figure and axes
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(15, 10))  # Adjust size for pagination
    axes = axes.flatten()

    # Show original image
    axes[0].imshow(img_resized)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Show selected feature maps with heatmap (viridis colormap)
    for i, idx in enumerate(selected_maps_page):
        feature_map = feature_maps[idx].cpu().numpy()
        axes[i + 1].imshow(feature_map, cmap="viridis")  # Apply 'viridis' colormap for heatmap
        axes[i + 1].set_title(f"Feature {idx + 1} (Var: {variances[idx]:.4f})")
        axes[i + 1].axis("off")
    
    # Turn off any unused axes
    for i in range(len(selected_maps_page) + 1, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    # Function to handle window close event
    def on_close(event):
        nonlocal page
        # Move to the next page when the window is closed
        page += 1
        total_pages = (len(selected_maps) // maps_per_page) + (1 if len(selected_maps) % maps_per_page > 0 else 0)
        if page < total_pages:
            plot_image_and_features_with_pagination(img, feature_maps, page=page)

    # Connect the window close event to the on_close function
    fig.canvas.mpl_connect('close_event', on_close)

    plt.show()


def parse_vehicle_csv(csv_file):
    """
    Parses a CSV file containing vehicle data and returns a DataFrame.

    Expected CSV structure: path,id

    :param csv_file: Path to the CSV file.
    :return: DataFrame with columns: ['imageName', 'vehicleID']
    """
    df = pd.read_csv(csv_file)  # Header is automatically inferred

    # Rename columns to match expected output
    df.rename(columns={'path': 'imageName', 'id': 'vehicleID'}, inplace=True)
    # Ensure vehicleID is int
    df['vehicleID'] = df['vehicleID'].astype(int)

    return df




activations = {}

# Choose an intermediate layer
layer_name = "layer3"  
layer = dict(model.model.named_children())[layer_name]

hook = layer.register_forward_hook(hook_fn)


# img_root = '/home/tomass/tomass/data/VeRi/image_test'
# df = parse_vehicle_data("/home/tomass/tomass/data/VeRi/test_label.xml")

img_root = '/home/tomass/tomass/magistrs/video_annotating'
df = parse_vehicle_csv("/home/tomass/tomass/magistrs/video_annotating/pidgeon_datasets/test_datasets/pidgeon_test_4/part2.csv")


img, path = load_random_image(df,img_root)

# plt.imshow(img); plt.title("Random Image"); plt.show()

feature_maps = get_feature_maps(img)

# plot_feature_maps(feature_maps)

# Plot both original image and feature maps 

#COLOR

plot_image_and_features(img, feature_maps, num_maps=14)

#plot_image_and_features_based_on_variance_color(img, feature_maps)


#GRAYSCALE
#plot_image_and_features_grayscale(img, feature_maps, num_maps=17)


# Option 3: Visualize Feature Maps Based on Criteria

# You could implement some automatic filtering of feature maps based on specific criteria like activation level or variance.
# For example, only display feature maps that have high variance,
# which may indicate that they are more important for detecting features in the image.
#plot_image_and_features_based_on_variance(img, feature_maps)

#Pageanated
# plot_image_and_features_with_pagination(img, feature_maps)