import clip
import torch

from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

print(model)

# image = preprocess(Image.open("/home/tomass/tomass/ReID_pipele/cropped/corrected_fisheye/KAB_SK_1_undist_1384779302960021.bmp/im1_car.jpg")).unsqueeze(0).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image) # [1, 512] shape of embedding

# print(image_features.cpu().numpy())
