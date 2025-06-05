import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import sys
import os
from matplotlib import cm
import xml.etree.ElementTree as ET
import matplotlib.colors as mcolors

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  # Get the directory above

sys.path.append(SCRIPT_DIR)  # Add current script's directory
sys.path.append(PARENT_DIR)  # Add parent directory

from vehicle_reid_repo2.vehicle_reid.load_model import load_model_from_opts
from counting_workspace.misc.feature_extract_AICity import extract_feature

device = "cuda"

img_root = '/home/tomass/tomass/data/VeRi/image_train'

# Load CSV file
csv_path = "/home/tomass/tomass/data/VeRi/VeRi_train.csv"
df = pd.read_csv(csv_path)

model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.pth", remove_classifier=True)

#print(model)
model.eval()
model.to(device)
model.classifier = torch.nn.Sequential()  # Remove classifier


#print(model)

# Define transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

def load_mapping(file_path):
    """
    Loads a mapping file (color or type) and returns a dictionary {id: name}.
    """
    mapping = {}
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                mapping[int(parts[0])] = parts[1]
    return mapping

# Function to extract features from multiple images
def extract_features_from_images(image_paths, model, device):
    features, labels = [], []
    
    for path in tqdm(image_paths, desc="Extracting Features"):
        img = Image.open(os.path.join(img_root, path)).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = extract_feature(model, img, device).cpu().numpy()
        features.append(feature)
        labels.append(df[df["imageName"] == path]["vehicleID"].values[0])  # Get ID from CSV

    return np.vstack(features), np.array(labels)

def plot_tsne_col(features, labels, title="t-SNE Visualization", label_mapping=None):
    """
    Plots a t-SNE visualization of feature vectors with colors that match actual car colors.

    :param features: NumPy array of shape (N, feature_dim), extracted feature vectors.
    :param labels: List or array of shape (N,), category labels.
    :param title: Title of the plot.
    :param label_mapping: Dictionary mapping label IDs to actual color names.
    """
    if len(labels) != len(features):
        raise ValueError(f"Mismatch: {len(features)} features but {len(labels)} labels")

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_features = tsne.fit_transform(features)

    # Convert label IDs to actual color names
    color_names = np.array([label_mapping.get(l, "orange") for l in labels])  # Default to 'orange' if missing

    # Convert color names to RGB/HEX values
    color_palette = {name: mcolors.CSS4_COLORS.get(name, "orange") for name in np.unique(color_names)}

    # Plot
    plt.figure(figsize=(8, 6))
    for color_name in np.unique(color_names):
        idx = color_names == color_name
        plt.scatter(reduced_features[idx, 0], reduced_features[idx, 1],
                    color=color_palette[color_name], label=color_name, alpha=0.7, edgecolors='k', s=30)

    plt.title(title)
    plt.legend(title="Transp. krāsas", loc="best")
    plt.show()

def plot_tsne_type(features, labels, title="t-SNE Visualization", label_mapping=None):
    """
    Plots a t-SNE visualization of feature vectors for car types, where each type is assigned a color.

    :param features: NumPy array of shape (N, feature_dim), extracted feature vectors.
    :param labels: List or array of shape (N,), category labels (typeIDs).
    :param title: Title of the plot.
    :param label_mapping: Dictionary mapping label IDs to readable car type names.
    """
    if len(labels) != len(features):
        raise ValueError(f"Mismatch: {len(features)} features but {len(labels)} labels")

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_features = tsne.fit_transform(features)

    # Map label IDs to car type names
    type_names = np.array([label_mapping.get(l, "Unknown") for l in labels])  # Default to 'Unknown' if missing

    # Use distinct colors from a color palette (optional: adjust palette if needed)
    color_palette = list(mcolors.TABLEAU_COLORS.values())  # 10 distinct colors
    color_map = {name: color_palette[i % len(color_palette)] for i, name in enumerate(np.unique(type_names))}

    # Plot
    plt.figure(figsize=(8, 6))
    for type_name in np.unique(type_names):
        idx = type_names == type_name
        plt.scatter(reduced_features[idx, 0], reduced_features[idx, 1],
                    color=color_map[type_name], label=type_name, alpha=0.7, edgecolors='k', s=30)

    plt.title(title)
    plt.legend(title="Uzbūves veidi", loc="best")
    plt.show()

def plot_tsne_camera(features, labels, title="t-SNE by Camera", label_mapping=None):
    """
    Plots a t-SNE visualization of feature vectors, colored by camera ID.

    :param features: NumPy array of shape (N, feature_dim), extracted feature vectors.
    :param labels: List or array of shape (N,), category labels (cameraIDs).
    :param title: Title of the plot.
    :param label_mapping: Optional dictionary mapping cameraID to camera names (not used here).
    """
    if len(labels) != len(features):
        raise ValueError(f"Mismatch: {len(features)} features but {len(labels)} labels")

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_features = tsne.fit_transform(features)

    # Camera labels, if available, will be passed in `labels` and can be displayed directly
    unique_labels = np.unique(labels)

    # Use the tab20 colormap for better distinguishability
    colormap = plt.get_cmap('tab20')
    color_map = {label: colormap(i % 20) for i, label in enumerate(unique_labels)}  # Wrap color palette if >20

    # Plot
    plt.figure(figsize=(8, 6))
    for label in unique_labels:
        idx = labels == label
        plt.scatter(reduced_features[idx, 0], reduced_features[idx, 1],
                    color=color_map[label], label=f"{label}", alpha=0.7, edgecolors='k', s=30)

    plt.title(title)
    plt.legend(title="Kameras ID", loc="best")
    plt.show()

def plot_tsne_vehicle_id(features, labels, title="t-SNE by Vehicle ID"):
    """
    Plots a t-SNE visualization of feature vectors, colored by vehicle ID, with IDs annotated next to each dot.
    
    :param features: NumPy array of shape (N, feature_dim), extracted feature vectors.
    :param labels: List or array of shape (N,), category labels (vehicle IDs).
    :param title: Title of the plot.
    """
    if len(labels) != len(features):
        raise ValueError(f"Mismatch: {len(features)} features but {len(labels)} labels")

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_features = tsne.fit_transform(features)

    # Normalize labels (vehicle IDs) to the range [0, 1] for colormap mapping
    norm = plt.Normalize(vmin=min(labels), vmax=max(labels))
    colormap = plt.get_cmap('viridis')  # Colormap that can handle ~600 unique values

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                          c=labels, cmap=colormap, alpha=0.6, edgecolors='k', s=50)

    # Annotate each point with the corresponding vehicle ID using white background for visibility
    for i, vehicle_id in enumerate(labels):
        # Position the label slightly offset from the point (0.1 in both x and y direction)
        plt.text(reduced_features[i, 0] + 0.01, reduced_features[i, 1] + 0.005, 
                 str(vehicle_id), fontsize=8, ha='left', va='bottom', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    # Add color bar for reference
    plt.colorbar(scatter, label="Vehicle ID")
    plt.title(title)
    plt.show()



type_of_descriptor = 2 # 0 - color, 1 - type, 2 - camera

# Load & Parse the XML File
df = parse_vehicle_data("/home/tomass/tomass/data/VeRi/train_label.xml")
print(df.head())  # Check first rows

# Load mappings
if type_of_descriptor == 0:
    color_mapping = load_mapping("/home/tomass/tomass/data/VeRi/list_color.txt")
    # Apply mappings to DataFrame
    df["color"] = df["colorID"].map(color_mapping)
    df["colorID"] = df["colorID"].astype(int)  # Convert to integer
if type_of_descriptor == 1:
    type_mapping = load_mapping("/home/tomass/tomass/data/VeRi/list_type.txt")
    df["type"] = df["typeID"].map(type_mapping)

print(df.head())  # Check updated DataFrame


# Select a subset of images (e.g., 2000 from query set)
num_samples = 2000
subset = df.sample(n=num_samples, random_state=42)
image_paths = subset["imageName"].tolist()
labels = subset["vehicleID"].tolist()

# Extract features
features, labels = extract_features_from_images(image_paths, model, device)

# PCA to reduce dimensionality to 50D before t-SNE (optional but speeds up t-SNE)
pca = PCA(n_components=50)
features_pca = pca.fit_transform(features)

# Plot t-SNE for Color
if type_of_descriptor == 0:
    plot_tsne_col(features, subset["colorID"], title="t-SNE pēc mašīnas krāsas", label_mapping=color_mapping)
# Plot t-SNE for Car Type
elif type_of_descriptor == 1:
    plot_tsne_type(features, subset["typeID"], title="t-SNE pēc mašīnas tipa", label_mapping=type_mapping)
# Plot t-SNE for Camera
elif type_of_descriptor == 2:
    plot_tsne_camera(features, subset["cameraID"], title="t-SNE pēc kameras ID")
# Plot t-SNE for Vehicle ID
elif type_of_descriptor == 3:
    plot_tsne_vehicle_id(features, labels, title="t-SNE pēc mašīnas ID")