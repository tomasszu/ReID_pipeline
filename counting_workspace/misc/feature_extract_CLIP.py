import os
import sys
import numpy as np
import re
import torch
import clip
from torchvision import transforms
from scipy.special import softmax

from PIL import Image

sys.path.append("..")

#Image transforms probably adapted from vehicle Re-ID model code
data_transforms = transforms.Compose([
    transforms.Resize((224, 224), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def fliplr(img):
    """flip images horizontally in a batch"""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    inv_idx = inv_idx.to(img.device)
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def extract_feature(model, X, device="cuda"):
    """Exract the embeddings of a single image tensor X"""
    if len(X.shape) == 3:
        X = torch.unsqueeze(X, 0)
    X = X.to(device)
    feature = model(X).reshape(-1)

    X = fliplr(X)
    flipped_feature = model(X).reshape(-1)
    feature += flipped_feature

    fnorm = torch.norm(feature, p=2)
    return feature.div(fnorm)


def save_extractions_to_lance_db(folder_path, folder_name):
    import misc.lance_db_init_CLIP as create_db
    from misc.lance_db_CLIP import update_vehicle
    from vehicle_reid.load_model import load_model_from_opts


    device = "cuda" if torch.cuda.is_available() else "cpu"

    CLIPmodel, preprocess = clip.load("ViT-B/32", device=device)

    model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo/vehicle_reid/model/result/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo/vehicle_reid/model/result/net_19.pth", remove_classifier=True)
    model.eval()
    model.to(device)

    extractables_folder = folder_path
    extractable_images = os.listdir(extractables_folder)

    ReIDimages = [Image.open(extractables_folder + x) for x in extractable_images]
    ReIDX_images = torch.stack(tuple(map(data_transforms, ReIDimages))).to(device)

    ReIDfeatures = [extract_feature(model, X) for X in ReIDX_images]
    ReIDfeatures = torch.stack(ReIDfeatures).detach().cpu()

    CLIPimages = [preprocess(Image.open(extractables_folder + x)).unsqueeze(0).to(device) for x in extractable_images]

    with torch.no_grad():
        CLIPfeatures = [(CLIPmodel.encode_image(i)) for i in CLIPimages]
        CLIPfeatures = torch.stack(CLIPfeatures, 1).detach().cpu()

    CLIPfeatures_array = np.array(CLIPfeatures, dtype=np.float32)[0]

    ReIDfeatures_array = np.array(ReIDfeatures)

    features_array = np.append(CLIPfeatures_array, ReIDfeatures_array, 1)

    #features_array = CLIPfeatures_array

    db = create_db._init_(folder_name)

    for image_name, embedding in zip(extractable_images, features_array):
        image_id = re.sub(r'[^0-9]', '', image_name)
        update_vehicle(image_id, embedding, folder_name, db)
        print(f" {image_name} Embedding saved to vector_db.")
        os.remove(folder_path + image_name)
        print(f" {image_name} deleted from folder")

    #query(np.zeros(512))

    

