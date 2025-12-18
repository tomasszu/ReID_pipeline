import numpy as np
import torch
from torchvision import transforms
import sys
from PIL import Image

sys.path.append('/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2')

from vehicle_reid.load_model import load_model_from_opts

class FeatureExtractor:
    def __init__(self, model_opts_path="vehicle_reid_repo2/vehicle_reid/model/main_finetune_editminer_121225/opts.yaml" ,model_ckpt_path="vehicle_reid_repo2/vehicle_reid/model/main_finetune_editminer_121225/net_45.pth"):
        self.device = "cuda"
        self.model = load_model_from_opts(model_opts_path, 
                                     ckpt=model_ckpt_path, 
                                     remove_classifier=True)
        self.model.eval()
        self.model.to(self.device)

        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def fliplr(self, img):
        """flip images horizontally in a batch"""
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
        inv_idx = inv_idx.to(img.device)
        img_flip = img.index_select(3, inv_idx)
        return img_flip
    
    def extract_feature(self, model, X, device="cuda"):
        """Exract the embeddings of a single image tensor X"""
        # print("X")
        # print(X.shape)
        if len(X.shape) == 3:
            X = torch.unsqueeze(X, 0)
            # print("unsqueezed X")
            # print(X.shape)
        X = X.to(device)
        feature = model(X).reshape(-1)
        # print("extracted feature")


        X = self.fliplr(X)
        flipped_feature = model(X).reshape(-1)
        feature += flipped_feature

        fnorm = torch.norm(feature, p=2)
        return feature.div(fnorm)
    
    def extract_batch_features(self, X):
        """Extract features for a batch of images (X: B x C x H x W)"""
        X = X.to(self.device)
        features = self.model(X)  # shape: (B, D)

        X_flipped = self.fliplr(X)
        flipped_features = self.model(X_flipped)  # shape: (B, D)

        features += flipped_features

        fnorm = torch.norm(features, p=2, dim=1, keepdim=True)
        normalized = features.div(fnorm)
        return normalized  # shape: (B, D)
    
    def get_features_batch(self, images):
        # First check if image is of PIL type for the transforms functions
        if images:
            if isinstance(images[0], Image.Image.__class__):
                # If image is PIL type, proceed
                X_images = torch.stack(tuple(map(self.data_transforms, images))).to(self.device)
            else:
                # If not, try converting from numpy array
                try:
                    images = [Image.fromarray(im) for im in images]
                    X_images = torch.stack(tuple(map(self.data_transforms, images))).to(self.device)
                except:
                    print("[Feature Extraction] Error: could not convert crops to PIL Images for transformation.")
                    return None

            features = self.extract_batch_features(X_images)
            features = features.detach().cpu()
            features_array = np.array(features)

            return features_array
        else:
            return None
    
    def get_features(self, images, device="cuda"):

        # First check if image is of PIL type for the transforms functions
        if isinstance(images[0], Image.Image.__class__):
            # If image is PIL type, proceed
            X_images = torch.stack(tuple(map(self.data_transforms, images))).to(device)
        else:
            # If not, try converting from numpy array
            try:
                images = [Image.fromarray(im) for im in images]
                X_images = torch.stack(tuple(map(self.data_transforms, images))).to(device)
            except:
                print("[Feature Extraction] Error: could not convert crops to PIL Images for transformation.")
                return None

        features = [self.extract_feature(self.model, x_im) for x_im in X_images]
        features = torch.stack(features).detach().cpu()

        features_array = np.array(features)

        return features_array