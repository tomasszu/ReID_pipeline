import numpy as np 
import pandas as pd
import os
import sys
import torch
from torchvision import transforms
from PIL import Image

sys.path.append("vehicle_reid_repo/")
sys.path.append("..")
#import vehicle_reid_repo
from vehicle_reid.load_model import load_model_from_opts
import matplotlib.pyplot as plt



DATA_ROOT = "cropped/"
#INTERSECTION_FOLDER = "intersection_1"


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

def save_extractions_to_CSV(folder):
    import numpy as np
    import csv
    import re
    
    csv_file_path = f"/home/tomass/tomass/ReID_pipele/embeddings/embeddings_data.csv"
    #csv_file_path = f"/home/tomass/tomass/ReID_pipele/embeddings/panorama_01_fisheye_day_000024.csv"


    device = "cuda"

    model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo/vehicle_reid/model/result/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo/vehicle_reid/model/result/net_19.pth", remove_classifier=True)
    model.eval()
    model.to(device)

    extractables_folder = folder
    extractable_images = os.listdir(extractables_folder)

    images = [Image.open(extractables_folder + x) for x in extractable_images]
    X_images = torch.stack(tuple(map(data_transforms, images))).to(device)

    features = [extract_feature(model, X) for X in X_images]
    features = torch.stack(features).detach().cpu()

    features_array = np.array(features)
    with open(csv_file_path, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["Vehicle_ID", "extracted_features"])
        for image_name, tensor_row in zip(extractable_images, features_array):
            image_id = re.sub(r'[^0-9]', '', image_name)
            csv_writer.writerow([image_id, tensor_row])
            # csv_writer.writerow({COUNTER : tensor_row}) ######################PROB!
            # COUNTER = COUNTER + 1
        print("Embeddings saved to CSV.")

def save_extractions_to_vector_db(folder_path, folder_name):
    import numpy as np
    import re
    #from misc.database import Vehicles
    import misc.database_init as create_db
    from misc.database import add_vehicle
    from misc.database import query

    from docarray import DocList
    import numpy as np
    from vectordb import InMemoryExactNNVectorDB, HNSWVectorDB

    device = "cuda"

    model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo/vehicle_reid/model/result/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo/vehicle_reid/model/result/net_19.pth", remove_classifier=True)
    model.eval()
    model.to(device)

    extractables_folder = folder_path
    extractable_images = os.listdir(extractables_folder)

    images = [Image.open(extractables_folder + x) for x in extractable_images]
    X_images = torch.stack(tuple(map(data_transforms, images))).to(device)

    features = [extract_feature(model, X) for X in X_images]
    features = torch.stack(features).detach().cpu()

    features_array = np.array(features)

    create_db._init_(folder_name)

    for image_name, embedding in zip(extractable_images, features_array):
        image_id = re.sub(r'[^0-9]', '', image_name)
        add_vehicle(image_id, embedding, folder_name)
        print(f" {image_name} Embedding saved to vector_db.")
        os.remove(folder_path + image_name)
        print(f" {image_name} deleted from folder")

    #query(np.zeros(512))

def save_extractions_to_lance_db(folder_path, folder_name):
    import numpy as np
    import re
    #from misc.database import Vehicles
    import misc.lance_db_init as create_db
    from misc.lance_db import add_vehicle
    from misc.lance_db import query

    from docarray import DocList
    import numpy as np
    import lancedb

    device = "cuda"

    model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo/vehicle_reid/model/result/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo/vehicle_reid/model/result/net_19.pth", remove_classifier=True)
    model.eval()
    model.to(device)

    extractables_folder = folder_path
    extractable_images = os.listdir(extractables_folder)

    images = [Image.open(extractables_folder + x) for x in extractable_images]
    X_images = torch.stack(tuple(map(data_transforms, images))).to(device)

    features = [extract_feature(model, X) for X in X_images]
    features = torch.stack(features).detach().cpu()

    features_array = np.array(features)

    db = create_db._init_(folder_name)

    for image_name, embedding in zip(extractable_images, features_array):
        image_id = re.sub(r'[^0-9]', '', image_name)
        add_vehicle(image_id, embedding, folder_name, db)
        print(f" {image_name} Embedding saved to vector_db.")
        os.remove(folder_path + image_name)
        print(f" {image_name} deleted from folder")

    #query(np.zeros(512))

    

