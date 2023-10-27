import numpy as np 
import pandas as pd
import os
import random
import torch
from torchvision import transforms
from PIL import Image
from vehicle_reid_repo.vehicle_reid.load_model import load_model_from_opts
import matplotlib.pyplot as plt

VRIC = "../data/VRIC/"
DATA_ROOT = "/home/tomass/tomass/ReID_pipele/cropped/"

QUERIED_IMAGE = 5




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


def get_scores(query_feature, gallery_features):
    """Calculate the similarity scores of the query and gallery features"""
    query = query_feature.view(-1, 1)
    score = torch.mm(gallery_features, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    return score


def visualise_main(queried_image, query_folder, gallery_folder):

    QUERIED_IMAGE = queried_image
    device = "cuda"

    model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo/vehicle_reid/model/result/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo/vehicle_reid/model/result/net_19.pth", remove_classifier=True)
    model.eval()
    model.to(device)

    query_folder = DATA_ROOT + query_folder
    query_folder_images = os.listdir(query_folder)

    gallery_folder = DATA_ROOT + gallery_folder
    gallery_folder_images = os.listdir(gallery_folder)

    #Selecting a query image

    query_image = Image.open(query_folder + query_folder_images[QUERIED_IMAGE])
    X_query = torch.unsqueeze(data_transforms(query_image), 0).to(device)

    #Ielasam gallery

    gallery_images = [Image.open(gallery_folder + x) for x in gallery_folder_images]
    X_gallery = torch.stack(tuple(map(data_transforms, gallery_images))).to(device)

    f_query = extract_feature(model, X_query).detach().cpu()
    f_gallery = [extract_feature(model, X) for X in X_gallery]
    f_gallery = torch.stack(f_gallery).detach().cpu()
    #print(f"query features: {f_query}")
    #print(f"gallery features: {f_gallery}")

    scores = get_scores(f_query, f_gallery)
    print(scores)

    reference_trans = transforms.Pad(4, (0, 0, 255)) # blue border for the query image
    gallery_trans = transforms.Pad(4, (255, 0, 0)) # red border for the gallery images

    gallery_images = [img.resize((112, 112)) for img in gallery_images]
    display_images = [(gallery_trans(img)) \
                        for img in gallery_images]
    display_images = [display_images[i] for i in np.argsort(scores)[::-1]]
    #Plot gallery images from the highest score to the lowest

    N_ROWS, N_COLS = 1, (display_images.__len__() +1) #4, 8
    score_labels = scores[np.argsort(scores)[::-1]]

    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(12, 8)) #12,8

    query_image = reference_trans(query_image.resize((112,112)))
    axes[0].imshow(query_image)

    for i in range(display_images.__len__()):
        axes[i+1].imshow(display_images[i])
    for i, ax in enumerate(axes.flat):
        if(i > 0):
            ax.set_xticks([])
            ax.set_xticks([], minor=True)
            ax.set_yticks([])
            ax.set_yticks([], minor=True)
            if(i-1 < score_labels.size):
                ax.set_xlabel(str(round(score_labels[i-1], 3)))
                for spine in [ax.spines.left, ax.spines.right, ax.spines.top, ax.spines.bottom]:
                    spine.set(visible=False)
    plt.savefig('figures/similarity_scores.png')
    # if out of memory issues arise, we come here to cleanup
    import gc
    gc.collect()
    torch.cuda.empty_cache()
