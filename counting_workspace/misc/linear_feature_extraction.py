import torch
import torch.nn as nn
from counting_workspace.misc.manual_feature_extraction import extract_manual_features
import numpy as np
import torch.nn.functional as F

class CombinedFeatureMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CombinedFeatureMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)   # Reduce dimensions while preserving information
        self.fc2 = nn.Linear(2048, 1024)        # Intermediate representation
        self.fc3 = nn.Linear(1024, 512)         # Further reduction
        self.fc4 = nn.Linear(512, num_classes)  # Final output layer

    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        
class HOG_MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(HOG_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)   # A larger first layer for spatial complexity
        self.fc2 = nn.Linear(4096, 2048)        # Intermediate layer to reduce dimensions
        self.fc3 = nn.Linear(2048, 1024)        # Further compression of features
        self.fc4 = nn.Linear(1024, num_classes) # Output layer (number of classes)
        
    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.fc1(x))  # ReLU activation
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class LBP_MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LBP_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)    # A smaller first layer for texture features
        self.fc2 = nn.Linear(512, 256)          # Intermediate layer
        self.fc3 = nn.Linear(256, num_classes)  # Output layer (number of classes)
        
    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.fc1(x))  # ReLU activation
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class RGB_MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(RGB_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)   # First layer to capture color features
        self.fc2 = nn.Linear(2048, 1024)        # Intermediate layer
        self.fc3 = nn.Linear(1024, 512)         # Further processing
        self.fc4 = nn.Linear(512, num_classes)  # Output layer (number of classes)
        
    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.fc1(x))  # ReLU activation
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class H10_MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(H10_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)    # First layer for hue features
        self.fc2 = nn.Linear(256, 128)          # Intermediate layer
        self.fc3 = nn.Linear(128, num_classes)  # Output layer (number of classes)
        
    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.fc1(x))  # ReLU activation
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def extract_linear_features(img, feature_type):
    """
    Pass manually extracted features through a trained linear model to get feature embeddings.

    Args:
        img (numpy.ndarray): Input image (not used for feature extraction, only as reference).
        feature_type (str): Feature type ('HOG', 'LBP', 'RGB', 'H10', 'Combined').

    Returns:
        torch.Tensor: Extracted feature embedding.
    """

    # Extract manual features (assumed function provided)
    features = extract_manual_features(img, feature_type=feature_type)

    features_array = np.array(features)


    features_size = features_array.shape[0]


    # Define model paths and corresponding architectures
    model_paths = {
        "HOG": ("/home/tomass/tomass/magistrs/Animal-Identification-from-Video-main/model_linear_HOG_VeRi_20.pth", HOG_MLP(features_size, 576)),
        "LBP": ("/home/tomass/tomass/magistrs/Animal-Identification-from-Video-main/model_linear_LBP_VeRi.pth", LBP_MLP(features_size, 576)),
        "RGB": ("/home/tomass/tomass/magistrs/Animal-Identification-from-Video-main/model_linear_RGB_VeRi.pth", RGB_MLP(features_size, 576)),
        "H10": ("/home/tomass/tomass/magistrs/Animal-Identification-from-Video-main/model_linear_H10_VeRi.pth", H10_MLP(features_size, 576)),
        "Combined": ("/home/tomass/tomass/magistrs/Animal-Identification-from-Video-main/model_linear_ALL_VeRi.pth", CombinedFeatureMLP(features_size, 576))
    }

    if feature_type not in model_paths:
        raise ValueError(f"Invalid feature type: {feature_type}. Choose from {list(model_paths.keys())}")

    # Load the model and weights
    model_path, model = model_paths[feature_type]
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    # Remove the classification layer to extract embeddings
    model.fc4 = nn.Identity()
    model.eval()

    # Convert features to tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Pass features through the model
    with torch.no_grad():
        embedding = model(features_tensor)

    return embedding.squeeze()
