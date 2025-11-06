import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import os
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  # Get the directory above
PROJECT_DIR = os.path.dirname(PARENT_DIR)  # Get the directory above

sys.path.append(SCRIPT_DIR)  # Add current script's directory
sys.path.append(PARENT_DIR)  # Add parent directory
sys.path.append(PROJECT_DIR)  # Add parent directory


from vehicle_reid_repo2.vehicle_reid.load_model import load_model_from_opts
from counting_workspace.misc.feature_extract_AICity import extract_feature
from counting_workspace.misc.feature_extract_AICity import data_transforms

device = "cuda"
model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.pth", remove_classifier=True)
model.eval()
model.to(device)

def mean_pairwise_similarity(embs1, embs2):
    # compute all pairwise similarities
    sims = cosine_similarity(embs1, embs2)
    return sims.mean()

df = pd.read_csv("/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/fisheye_dataset.csv")  # columns: path,id
data_dir = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004"

# Example: build dictionary {id: {cam: [embeddings]}}
embeddings = {}
for _, row in df.iterrows():
    img_path = row["path"]  #data_dir + "/" + row["path"]
    track_id = row["id"]
    cam = 4

    img = Image.open(img_path)  # PIL image
    img = data_transforms(img)
    with torch.no_grad():
        emb = extract_feature(model, img)  # np.array shape (D,)

    # ensure it’s on CPU as NumPy
    emb = emb.detach().cpu().numpy()

    embeddings.setdefault(track_id, {}).setdefault(cam, []).append(emb)

df = pd.read_csv("/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c001/fisheye_dataset.csv")  # columns: path,id
data_dir = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c001"

for _, row in df.iterrows():
    img_path = row["path"]  #data_dir + "/" + row["path"]
    track_id = row["id"]
    cam = 1

    img = Image.open(img_path)  # PIL image
    img = data_transforms(img)

    with torch.no_grad():
        emb = extract_feature(model, img)  # np.array shape (D,)

    # ensure it’s on CPU as NumPy
    emb = emb.detach().cpu().numpy()

    embeddings.setdefault(track_id, {}).setdefault(cam, []).append(emb)


## Compute cross camera closeness

results = []

for track_id, cams in embeddings.items():
    if 1 in cams and 4 in cams:
        sim = mean_pairwise_similarity(
            np.vstack(cams[1]),
            np.vstack(cams[4])
        )
        results.append(sim)

mean_similarity = np.mean(results)
print("Average similarity across IDs:", mean_similarity)
