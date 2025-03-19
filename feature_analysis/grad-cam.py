import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
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

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook to capture activations
        self.target_layer.register_forward_hook(self.save_activation)
        # Hook to capture gradients
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        """
        Computes Grad-CAM heatmap for a given input.
        :param input_tensor: Preprocessed image tensor.
        :param target_class: Class index to compute Grad-CAM for.
        :return: Heatmap overlaid on the original image.
        """
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()  # Use highest confidence prediction
        
        # Compute gradient of the target class
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Compute weights for each activation map
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU to keep positive activations only
        
        # Normalize and resize heatmap
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize between 0-1
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))  # Resize to image size
        
        return cam

def preprocess_image(image_path):
    """
    Preprocesses an image for ResNet input.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def overlay_heatmap(image_path, heatmap):
    """
    Overlays the Grad-CAM heatmap on an image.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    
    heatmap = np.uint8(255 * heatmap)  # Scale heatmap to 0-255
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply colormap
    
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlay

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

# Example Usage
if __name__ == "__main__":

    device = "cuda"

    img_root = '/home/tomass/tomass/data/VeRi/image_test'
    df = parse_vehicle_data("/home/tomass/tomass/data/VeRi/test_label.xml")

    model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.pth", remove_classifier=True)

    # Randomly select one row
    random_row = df.sample(n=1)

    # Get the image path
    path = random_row["imageName"].values[0]

    print("Selected image:", path)

    # Use the last convolutional layer (for ResNet-50, it's layer4)
    target_layer = model.model.layer4

    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)

    # Load an image
    image_path = os.path.join(img_root, path)
    input_tensor = preprocess_image(image_path)

    # Compute Grad-CAM heatmap
    heatmap = grad_cam.generate_cam(input_tensor)

    # Overlay the heatmap on the original image
    overlayed_image = overlay_heatmap(image_path, heatmap)

    # Display the result
    plt.imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Grad-CAM Heatmap")
    plt.show()
