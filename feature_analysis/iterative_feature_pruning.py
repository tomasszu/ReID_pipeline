import torch
import numpy as np
import os
import cv2
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
from torch import nn
from PIL import Image
import pandas as pd

# Assuming a pre-trained re-identification model is available (ResNet-50 for example)
# Example: A simplified ResNet-50 feature extractor (you can replace with any model you use)

class VehicleReIDModel(nn.Module):
    def __init__(self, pretrained=True):
        super(VehicleReIDModel, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)
        # Remove the final fully connected layers to extract features
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 512)  # 512-dimensional feature vector
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)  # Get the 512-d feature vector
        return x

# Load a model instance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VehicleReIDModel().to(device)
model.eval()

# Preprocessing function for input images
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

# Function to extract feature embeddings from an image
def extract_embedding(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        embedding = model(image)  # Get the embedding from the model
    return embedding.cpu().numpy().flatten()

# Function to load images and extract embeddings
def load_and_extract_embeddings(image_paths):
    embeddings = []
    image_ids = []
    for img_path in image_paths:
        image_id = os.path.basename(img_path).split('.')[0]  # Assuming filenames are like 'vehicleID.jpg'
        image_ids.append(image_id)
        embeddings.append(extract_embedding(img_path))
    return np.array(embeddings), image_ids

# Function to compute cosine similarity between query and gallery embeddings
def compare_embeddings(query_embedding, gallery_embeddings):
    return cosine_similarity(query_embedding.reshape(1, -1), gallery_embeddings).flatten()

# Function to evaluate the Re-ID performance: Rank-1 accuracy
def evaluate_reid(query_embeddings, query_ids, gallery_embeddings, gallery_ids):
    correct_matches = 0
    
    for i, query_embedding in enumerate(query_embeddings):
        similarities = compare_embeddings(query_embedding, gallery_embeddings)
        best_match_index = np.argmax(similarities)
        
        if gallery_ids[best_match_index] == query_ids[i]:
            correct_matches += 1
    
    rank_1_accuracy = correct_matches / len(query_embeddings)
    return rank_1_accuracy


data_dir = '/home/tomass/tomass/data'

# Load gallery and query embeddings
gallery_file_path = '/home/tomass/tomass/data/EDI_Cam_testData/cam1.csv'  # CSV file containing gallery embeddings
query_file_path = '/home/tomass/tomass/data/EDI_Cam_testData/cam2.csv'     # CSV file containing query embeddings

gallery_images = [os.path.join(gallery_folder, f) for f in os.listdir(gallery_folder) if f.endswith('.jpg')]
query_images = [os.path.join(query_folder, f) for f in os.listdir(query_folder) if f.endswith('.jpg')]

# Extract embeddings from gallery and query images
gallery_embeddings, gallery_ids = load_and_extract_embeddings(gallery_images)
query_embeddings, query_ids = load_and_extract_embeddings(query_images)

# Evaluate the re-identification performance
rank_1_accuracy = evaluate_reid(query_embeddings, query_ids, gallery_embeddings, gallery_ids)

# Output the Rank-1 accuracy (can expand to other metrics like mAP, etc.)
print(f"Rank-1 Accuracy: {rank_1_accuracy * 100:.2f}%")


gallery_embeddings, gallery_ids = load_gallery_embeddings(gallery_file_path, data_dir)
query_embeddings, query_ids = load_query_embeddings(query_file_path, data_dir)

# Perform re-identification evaluation
rank_1_accuracy = evaluate_reid(query_embeddings, query_ids, gallery_embeddings, gallery_ids)

# Output the Rank-1 accuracy (can expand to other metrics like mAP, etc.)
print(f"Rank-1 Accuracy: {rank_1_accuracy * 100:.2f}%")
