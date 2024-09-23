import numpy as np 
import pandas as pd
import os
import sys
import torch
from torchvision import transforms
from PIL import Image


sys.path.append("/home/toms.zinars/Anzelika/vehicle_reid_repo2/")
sys.path.append("/home/toms.zinars/Anzelika/")
sys.path.append("..")
#import vehicle_reid_repo2
#from vehicle_reid.load_model_ModelArchChange import load_model_from_opts
from vehicle_reid.load_model import load_model_from_opts
import matplotlib.pyplot as plt

import counting_workspace.misc.lance_db_CLIP_AICity as l_db



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
    # print("X")
    # print(X.shape)
    if len(X.shape) == 3:
        X = torch.unsqueeze(X, 0)
        # print("unsqueezed X")
        # print(X.shape)
    X = X.to(device)
    feature = model(X).reshape(-1)
    # print("extracted feature")
    # print(feature.shape)

    X = fliplr(X)
    flipped_feature = model(X).reshape(-1)
    feature += flipped_feature

    fnorm = torch.norm(feature, p=2)
    return feature.div(fnorm)

def save_extractions_to_CSV(folder):
    import numpy as np
    import csv
    import re
    
    csv_file_path = f"/home/toms.zinars/Anzelika/ReID_pipele/embeddings/embeddings_data.csv"
    #csv_file_path = f"/home/tomass/tomass/ReID_pipele/embeddings/panorama_01_fisheye_day_000024.csv"


    device = "cuda"

    model = load_model_from_opts("/home/toms.zinars/Anzelika/Svari_pirmais_katrs_katraa/opts.yaml", ckpt="/home/toms.zinars/Anzelika/Svari_pirmais_katrs_katraa/net_19.pth", remove_classifier=True)
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


def save_extractions_to_lance_db(folder_path, folder_name, saving_mode):
    import numpy as np
    import re
    #from misc.database import Vehicles
    sys.path.append("/home/toms.zinars/Anzelika/counting_workspace/")

    import misc.lance_db_init as create_db
    from misc.lance_db_AICity import update_vehicle
    from misc.lance_db_AICity import add_vehicle

    from docarray import DocList
    import numpy as np
    import lancedb

    device = "cuda"

    # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/result7/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/result7/net_10.pth")
    # print(model)
    #model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch_change4/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch_change4/net_22.pth", remove_classifier=True)
    model = load_model_from_opts("/home/toms.zinars/Anzelika/Svari_pirmais_katrs_katraa/opts.yaml", ckpt="/home/toms.zinars/Anzelika/Svari_pirmais_katrs_katraa/net_19.pth", remove_classifier=True)

    #print(model)
    model.eval()
    model.to(device)

    extractables_folder = folder_path
    extractable_images = os.listdir(extractables_folder)

    images = [Image.open(extractables_folder + x) for x in extractable_images]
    X_images = torch.stack(tuple(map(data_transforms, images))).to(device)

    # print("X_images shape")
    # print(X_images.shape)

    features = [extract_feature(model, X) for X in X_images]
    features = torch.stack(features).detach().cpu()

    features_array = np.array(features)

    #print(f"features_array: {features_array}")

    db = create_db._init_(folder_name)

    for image_name, embedding in zip(extractable_images, features_array):
        image_id = re.sub(r'[^0-9]', '', image_name)
        #add_vehicle(image_id, embedding, folder_name, db)
        #print(f"embedding: {embedding}")
        if (saving_mode == 0) or (saving_mode == 2):
            update_vehicle(image_id, embedding, folder_name, db)
        elif (saving_mode == 1) or (saving_mode == 3):
            add_vehicle(image_id, embedding, folder_name, db)
        #print(f" {image_name} Embedding saved to vector_db.")
        os.remove(folder_path + image_name)
        #print(f" {image_name} deleted from folder")

    #query(np.zeros(512))
        
def compare_extractions_to_lance_db(folder_path, queried_folder_name):
    import numpy as np
    import re
    #from misc.database import Vehicles
    sys.path.append("/home/toms.zinars/Anzelika/counting_workspace/")

    import misc.lance_db_init as create_db
    from misc.lance_db_AICity import update_vehicle

    from docarray import DocList
    import numpy as np
    import lancedb

    device = "cuda"

    #model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch_change4/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch_change4/net_22.pth", remove_classifier=True)

    model = load_model_from_opts("/home/toms.zinars/Anzelika/Svari_pirmais_katrs_katraa/opts.yaml", ckpt="/home/toms.zinars/Anzelika/Svari_pirmais_katrs_katraa/net_19.pth", remove_classifier=True)
    model.eval()
    model.to(device)

    extractables_folder = folder_path
    extractable_images = os.listdir(extractables_folder)

    ReIDimages = [Image.open(extractables_folder + x) for x in extractable_images]
    ReIDX_images = torch.stack(tuple(map(data_transforms, ReIDimages))).to(device)

    ReIDfeatures = [extract_feature(model, X) for X in ReIDX_images]
    ReIDfeatures = torch.stack(ReIDfeatures).detach().cpu()

    ReIDfeatures_array = np.array(ReIDfeatures)

    #print(f"features_array: {features_array}")

    db = create_db._init_(queried_folder_name)


    compare_array = []
    for image_name, embedding in zip(extractable_images, ReIDfeatures_array):
        image_id = re.sub(r'[^0-9]', '', image_name)
        compare_array.append([image_id, embedding])


    track_map = {}
    results_map = []
    print("From intersection 2. -> 1. :")
    for vehicle in compare_array:
    #print(db.query(vehicle[1],intersection))
        results = l_db.query_for_IDs(vehicle[1],queried_folder_name)
        results_map.append([vehicle[0],results[0]['vehicle_id'], results[0]['_distance']])

        print("-------------------------------")
        if(results and results != -1):
            track_map[vehicle[0]] = [results[0]['vehicle_id'], results[0]['_distance']]
            print(f"{vehicle[0]} found as ->  \n")
            for i, result in enumerate(results):
                id = result['vehicle_id']
                distance = result['_distance']
                print(f"{id} [{distance}%]")

    return results_map
        
def save_image_to_lance_db(image_path, vehicle_id, folder_name, saving_mode):
    import numpy as np
    import re
    sys.path.append("/home/toms.zinars/Anzelika/counting_workspace/")
    import misc.lance_db_init as create_db
    from misc.lance_db_AICity import update_vehicle, add_vehicle
    import lancedb

    device = "cuda"
    model = load_model_from_opts("/home/toms.zinars/Anzelika/Svari_pirmais_katrs_katraa/opts.yaml", ckpt="/home/toms.zinars/Anzelika/Svari_pirmais_katrs_katraa/net_19.pth", remove_classifier=True)
    model.eval().to(device)

    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    transformed_image = data_transforms(image).unsqueeze(0).to(device)  # Add batch dimension

    # Extract features
    features = extract_feature(model, transformed_image)
    features_array = features.detach().cpu().numpy()  # Convert to numpy array

    # Initialize the database
    db = create_db._init_(folder_name)

    # Save to the database
    if (saving_mode == 0) or (saving_mode == 2):
        update_vehicle(vehicle_id, features_array, folder_name, db)
    elif (saving_mode == 1) or (saving_mode == 3):
        add_vehicle(vehicle_id, features_array, folder_name, db)

def compare_image_to_lance_db(image_path, vehicle_id, queried_folder_name):
    import numpy as np
    import re
    sys.path.append("/home/toms.zinars/Anzelika/counting_workspace/")
    import misc.lance_db_init as create_db
    from docarray import DocList
    import lancedb

    device = "cuda"
    model = load_model_from_opts("/home/toms.zinars/Anzelika/Svari_pirmais_katrs_katraa/opts.yaml", ckpt="/home/toms.zinars/Anzelika/Svari_pirmais_katrs_katraa/net_19.pth", remove_classifier=True)
    model.eval().to(device)

    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    transformed_image = data_transforms(image).unsqueeze(0).to(device)  # Add batch dimension

    # Extract features
    features = extract_feature(model, transformed_image)
    features_array = features.detach().cpu().numpy()  # Convert to numpy array

    db = create_db._init_(queried_folder_name)
    compare_array = [[vehicle_id, features_array]]

    results_map = []
    print("From intersection 2. -> 1. :")
    for vehicle in compare_array:
        results = l_db.query_for_IDs(vehicle[1], queried_folder_name)
        if results and results != -1:
            results_map.append([vehicle[0], results[0]['vehicle_id'], results[0]['_distance']])
            print(f"{vehicle[0]} found as ->  \n")
            for result in results:
                print(f"{result['vehicle_id']} [{result['_distance']}]")

    return results_map
        


    

