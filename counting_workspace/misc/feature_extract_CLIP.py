import os
import sys
import numpy as np
import torch
from torchvision import transforms

from vehicle_reid_repo2.vehicle_reid.load_model import load_CLIP_head_from_opts

import counting_workspace.misc.lance_db_CLIP_AICity as l_db

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

def z_score_normalize_and_concat(v1, v2):
    # Calculate mean and standard deviation for each vector
    mean_v1, std_v1 = np.mean(v1), np.std(v1)
    mean_v2, std_v2 = np.mean(v2), np.std(v2)

    # Apply z-score normalization to each vector
    v1_normalized = (v1 - mean_v1) / std_v1
    v2_normalized = (v2 - mean_v2) / std_v2

    normalized_vector = np.append(v1_normalized, v2_normalized, 1)

    return normalized_vector

def concat_and_z_score_normalize(v1, v2):

    appended_vector = np.append(v1, v2, 1)
    mean_v, std_v = np.mean(appended_vector), np.std(appended_vector)
    normalized = (appended_vector - mean_v) / std_v


    return normalized

def save_image_to_lance_db(image_path, vehicle_id, folder_name, saving_mode, clip_visual):
    import numpy as np
    import re
    #from misc.database import Vehicles
    import counting_workspace.misc.lance_db_init as create_db
    from counting_workspace.misc.lance_db_AICity import update_vehicle
    from counting_workspace.misc.lance_db_AICity import add_vehicle

    from docarray import DocList
    import numpy as np
    import lancedb

    device = "cuda"

    global model
    if not 'model' in globals():
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/result7/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/result7/net_10.pth")
        # print(model)
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch+loss_change4/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch+loss_change4/net_17.pth", remove_classifier=True)
        model = load_CLIP_head_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/CLIP_head_train/opts.yaml", clip_visual=clip_visual, ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/CLIP_head_train/net_10.pth", remove_classifier=True)
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/benchmark_model/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/benchmark_model/net_19.pth", remove_classifier=True)
        #print(model)
        model.eval()
        model.to(device)
        #print(model.classifier.add_block[2])
        #model.classifier.add_block[2] = nn.Sequential()
        #print(model)


    images = [Image.open(image_path)]
    X_images = torch.stack(tuple(map(data_transforms, images))).to(device)


    # print("X_images shape")
    # print(X_images.shape)

    features = [extract_feature(model, X_images)]
    features = torch.stack(features).detach().cpu()

    features_array = np.array(features)

    features_size = features_array.shape[1]

    #print(f"features_array: {features_array}")

    db = create_db._init_(folder_name, features_size)

    if (saving_mode == 0) or (saving_mode == 2):
        update_vehicle(vehicle_id, features_array[0], folder_name, db)
    elif (saving_mode == 1) or (saving_mode == 3):
        add_vehicle(vehicle_id, features_array[0], folder_name, db)


    #query(np.zeros(512))

def compare_image_to_lance_db(image_path, vehicle_id, queried_folder_name, clip_visual):
    import numpy as np
    import re
    #from misc.database import Vehicles
    import counting_workspace.misc.lance_db_init as create_db
    from counting_workspace.misc.lance_db_AICity import update_vehicle

    import lancedb

    device = "cuda"

    global model
    if not 'model' in globals():
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch_change4/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch_change4/net_9.pth", remove_classifier=True)
        model = load_CLIP_head_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/CLIP_head_train/opts.yaml", clip_visual=clip_visual, ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/CLIP_head_train/net_39.pth", remove_classifier=True)
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/benchmark_model/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/benchmark_model/net_19.pth", remove_classifier=True)
        model.eval()
        model.to(device)
        #model.classifier.add_block[2] = nn.Sequential()

    #print(image_path)
    images = [Image.open(image_path)]
    X_images = torch.stack(tuple(map(data_transforms, images))).to(device)

    features = [extract_feature(model, X) for X in X_images]
    features = torch.stack(features).detach().cpu()

    features_array = np.array(features)

    features_size = features_array.shape[1]

    #print(f"features_array: {features_array}")

    db = create_db._init_(queried_folder_name, features_size)


    compare_array = []
    compare_array.append([vehicle_id, features_array[0]])


    track_map = {}
    results_map = []
    print("From intersection 2. -> 1. :")
    for vehicle in compare_array:
    #print(db.query(vehicle[1],intersection))
        results = l_db.query_for_IDs(vehicle[1],queried_folder_name, limit=3)
        results_map.append([vehicle[0],int(results[0]['vehicle_id']), results[0]['_distance']])

        print("-------------------------------")
        if(results and results != -1):
            track_map[vehicle[0]] = [results[0]['vehicle_id'], results[0]['_distance']]
            print(f"{vehicle[0]} found as ->  \n")
            for i, result in enumerate(results):
                id = result['vehicle_id']
                distance = result['_distance']
                print(f"{id} [{distance}%]")
    #print(results_map)

    return results_map


def save_extractions_to_lance_db(folder_path, folder_name, saving_mode, clip_visual):
    import numpy as np
    import re
    #from misc.database import Vehicles
    import counting_workspace.misc.lance_db_init as create_db
    from counting_workspace.misc.lance_db_AICity import update_vehicle
    from counting_workspace.misc.lance_db_AICity import add_vehicle

    from docarray import DocList
    import numpy as np
    import lancedb

    device = "cuda"

    # start_time = time.time()

    if not 'model' in globals():
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/result7/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/result7/net_10.pth")
        # print(model)
        #model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch_change4/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch_change4/net_22.pth", remove_classifier=True)
        global model
        model = load_CLIP_head_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/CLIP_head_train/opts.yaml", clip_visual=clip_visual, ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/CLIP_head_train/net_39.pth", remove_classifier=True)

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

    # duration = time.time() - start_time
    #print(f"[t]Load model + extract features took {duration*1000:.2f} ms.")
    # start_time = time.time()

    #print(f"features_array: {features_array}")

    db = create_db._init_(folder_name)

    for image_name, embedding in zip(extractable_images, features_array):
        image_id = re.sub(r'[^0-9]', '', image_name)
        #add_vehicle(image_id, embedding, folder_name, db)
        #print(f"embedding: {embedding}")
        if (saving_mode == 0) or (saving_mode == 2):
            update_vehicle(image_id, embedding, folder_name, db)
            # duration = time.time() - start_time
            # print(f"[t]Update vehicle in db took {duration*1000:.2f} ms.")
        elif (saving_mode == 1) or (saving_mode == 3):
            add_vehicle(image_id, embedding, folder_name, db)
            # duration = time.time() - start_time
            # print(f"[t]Add vehicle in db took {duration*1000:.2f} ms.")
        #print(f" {image_name} Embedding saved to vector_db.")
        os.remove(folder_path + image_name)
        #print(f" {image_name} deleted from folder")

    #query(np.zeros(512))

def compare_extractions_to_lance_db_For_Rank(folder_path, queried_folder_name, clip_visual):
    import numpy as np
    import re
    import counting_workspace.misc.lance_db_init as create_db

    import numpy as np
    import lancedb

    device = "cuda"

    # start_time = time.time()

    #model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch_change4/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch_change4/net_22.pth", remove_classifier=True)

    global model
    if not 'model' in globals():
        model = load_CLIP_head_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/CLIP_head_train/opts.yaml", clip_visual=clip_visual, ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/CLIP_head_train/net_19.pth", remove_classifier=True)
        model.eval()
        model.to(device)


    extractable_images = os.listdir(folder_path)
    ReIDimages = [Image.open(folder_path + x) for x in extractable_images]
    ReIDX_images = torch.stack(tuple(map(data_transforms, ReIDimages))).to(device)

    ReIDfeatures = torch.stack([extract_feature(model, X) for X in ReIDX_images]).detach().cpu().numpy()

    print("From intersection 2. -> 1. :")
    results_map = []
    for image_name, embedding in zip(extractable_images, ReIDfeatures):
        image_id = re.sub(r'[^0-9]', '', image_name)  # Extract numerical ID
        print("-------------------------------")
        print(f"{image_id} found as ->  \n")

        # Query top-n results
        results = l_db.query_for_IDs(embedding, queried_folder_name, limit=5)
        #results = [r for r in results if r['_distance'] <= 0.6]
        
        #0.8 similarity te ir randomaa kkada, bet baasically 0.8 ir bare minimum. Kkadiem modeliem vnk tie similarity scores ir atskirigi
        if results and results != -1:
            retrieved_ids = [int(result['vehicle_id']) if result['_distance'] <= 0.8 else None for result in results] # Filter by distance threshold
            distances = [result['_distance'] if result['_distance'] <= 0.8 else None for result in results] # Filter by distance threshold

            results_map.append([image_id, retrieved_ids, distances])
            retrieved_ids = retrieved_ids[:5] # Take top-5
            distances = distances[:5]
            print(f"{retrieved_ids} [{distances}%]")

    return results_map  # Now contains Top-5 results per query

    

