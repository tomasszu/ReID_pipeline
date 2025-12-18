import numpy as np 
import os
import sys
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_pybind11_state import OrtValue

import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)  # Get the directory above
PROJECT_DIR = os.path.dirname(PARENT_DIR)  # Get the directory above

sys.path.append(SCRIPT_DIR)  # Add current script's directory
sys.path.append(PARENT_DIR)  # Add parent directory
sys.path.append(PROJECT_DIR)  # Add parent directory

#import vehicle_reid_repo2
#from vehicle_reid.load_model_ModelArchChange import load_model_from_opts
# from vehicle_reid.load_model_ModelArchChange_ForInfer_partial import load_model_from_opts
from vehicle_reid_repo2.vehicle_reid.load_model import load_model_from_opts
import matplotlib.pyplot as plt

import counting_workspace.misc.lance_db_CLIP_AICity as l_db

from counting_workspace.misc.linear_feature_extraction import extract_linear_features

from counting_workspace.misc.manual_feature_extraction import extract_manual_features



DATA_ROOT = "cropped/"
#INTERSECTION_FOLDER = "intersection_1"

# global ONNX session to reuse
session = None


#Image transforms probably adapted from vehicle Re-ID model code
data_transforms = transforms.Compose([
    transforms.Resize((224, 224), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to print memory usage on the GPU
def print_gpu_memory():
    if torch.cuda.is_available():
        print("VRAM USAGE -------------------------------------------------------------")
        # Report GPU memory allocated by tensors
        print(f"Memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")

        # Report GPU memory reserved (includes fragmentation and cached memory)
        print(f"Memory reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

        # Report the maximum memory allocated (useful for peak usage during inference)
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")


def fliplr(img):
    """flip images horizontally in a batch"""
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    inv_idx = inv_idx.to(img.device)
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def fliplr_numpy(img_numpy):
    # Assume img_numpy shape: (1, C, H, W)
    flipped = np.flip(img_numpy, axis=3).copy()  # horizontal flip along width
    return flipped


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


    X = fliplr(X)
    flipped_feature = model(X).reshape(-1)
    feature += flipped_feature

    fnorm = torch.norm(feature, p=2)
    return feature.div(fnorm)

def extract_feature_onnx(session, X_numpy):
    """Extract embeddings using ONNX Runtime"""

    X_numpy = [X_numpy]  # ONNX expects a batch shape of (1, C, H, W)

    input_names=["input"]
    output_names=["output"]

    # Forward
    ort_inputs = {input_names[0]: X_numpy}
    feature1 = session.run(None, ort_inputs)[0].reshape(-1)

    # Flip
    flipped = fliplr_numpy(X_numpy)
    ort_inputs = {input_names[0]: flipped}
    feature2 = session.run(None, ort_inputs)[0].reshape(-1)

    # Sum and normalize
    feature = feature1 + feature2
    norm = np.linalg.norm(feature, ord=2)
    return feature / norm

def extract_batch_features_onnx(session: ort.InferenceSession, X_numpys: np.ndarray):
    """
    Efficient ONNX inference using GPU OrtValues and batch processing.
    """

    input_name = session.get_inputs()[0].name

    def run_inference(input_np):
        ort_input = ort.OrtValue.ortvalue_from_numpy(input_np, device_type="cuda", device_id=0)
        ort_inputs = {input_name: ort_input}
        ort_output = session.run_with_ort_values(None, ort_inputs)[0]
        return ort_output.numpy()  # <-- extract NumPy array from OrtValue

    features1 = run_inference(X_numpys)
    flipped = np.flip(X_numpys, axis=3).copy()
    features2 = run_inference(flipped)

    features = features1 + features2

    # Normalize
    features_torch = torch.from_numpy(features).cuda()
    normalized = torch.nn.functional.normalize(features_torch, dim=1)
    return normalized.cpu().numpy()



def extract_batch_features(model, X, device="cuda"):
    """Extract features for a batch of images (X: B x C x H x W)"""
    X = X.to(device)
    features = model(X)  # shape: (B, D)

    X_flipped = fliplr(X)
    flipped_features = model(X_flipped)  # shape: (B, D)

    features += flipped_features

    fnorm = torch.norm(features, p=2, dim=1, keepdim=True)
    normalized = features.div(fnorm)
    return normalized  # shape: (B, D)


def save_extractions_to_CSV(folder):
    import numpy as np
    import csv
    import re
    
    csv_file_path = f"/home/tomass/tomass/ReID_pipele/embeddings/embeddings_data.csv"
    #csv_file_path = f"/home/tomass/tomass/ReID_pipele/embeddings/panorama_01_fisheye_day_000024.csv"


    device = "cuda"

    model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/result/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/result/net_10.pth", remove_classifier=True)
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


def save_onnx_batch_to_opensearch_db(batch_image_paths, batch_vehicle_ids, db, saving_mode):

    onnx_device = "CUDAExecutionProvider" # or CPUExecutionProvider

    global session
    if session is None:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            "/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39_opt.onnx",
            sess_options=so,
            providers=[onnx_device],
        )

    images = [Image.open(path).convert("RGB") for path in batch_image_paths]
    X_images = [data_transforms(img) for img in images]  # shape (C, H, W)

    # Convert to numpy and stack into batch (B,C,H,W)
    imgs_numpy = np.stack([img.cpu().numpy().astype(np.float32) for img in X_images], axis=0)

    # Extract features batch from ONNX
    features = extract_batch_features_onnx(session, imgs_numpy)  # shape (B, D)


    # VEKTORA IZMERS DATUBAZEE TIEK NOMAINITS VEIDOJOT SHEEMU, NE KKUR KODAA

    # Save each vector with its vehicle_id
    if saving_mode in [1, 3]:
        for vehicle_id, feature_vector in zip(batch_vehicle_ids, features):
            db.insert(vehicle_id=vehicle_id, feature_vector=feature_vector, times_summed=0)
    else:
        print("Error! Not provisioned vector summing operation!")



def save_onnx_image_to_opensearch_db(image_path, vehicle_id, db, saving_mode):

    onnx_device = "CUDAExecutionProvider" # or CPUExecutionProvider


    global session
    if session is None:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            "/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.onnx",
            sess_options=so,
            providers=[onnx_device]
        )

    image = Image.open(image_path)
    X_image = data_transforms(image)  # shape (C, H, W)

    img_numpy = X_image.cpu().numpy().astype(np.float32)

    features_array = [extract_feature_onnx(session, img_numpy)]


    # VEKTORA IZMERS DATUBAZEE TIEK NOMAINITS VEIDOJOT SHEEMU, NE KKUR KODAA

    if (saving_mode == 0) or (saving_mode == 2):
        print("Error! Not provisioned vector summing operation!><><><><><<><<><><><><><><><><><><><><><")
    elif (saving_mode == 1) or (saving_mode == 3):
        db.insert(vehicle_id = vehicle_id, feature_vector = features_array[0], times_summed = 0)

def compare_onnx_batch_to_opensearch_db(batch_image_paths, batch_vehicle_ids, db):

    onnx_device = "CUDAExecutionProvider" # or CPUExecutionProvider

    global session
    if session is None:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            "/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39_opt.onnx",
            sess_options=so,
            providers=[onnx_device]
        )

    images = [Image.open(path).convert("RGB") for path in batch_image_paths]
    X_images = [data_transforms(img) for img in images]  # shape (C, H, W)

    # Convert to numpy and stack into batch (B,C,H,W)
    imgs_numpy = np.stack([img.cpu().numpy().astype(np.float32) for img in X_images], axis=0)

    # Extract features batch from ONNX
    features = extract_batch_features_onnx(session, imgs_numpy)  # shape (B, D)

    compare_array = []
    for vehicle_id, feature_vector in zip(batch_vehicle_ids, features):
        compare_array.append([vehicle_id, feature_vector])
    
    track_map = {}
    results_map = []


    print("From intersection 2. -> 1. :")
    for vehicle in compare_array:
        results = db.query_vector(vehicle[1], k=3)
        results_map.append([vehicle[0], int(results[0][0]), results[0][1]])

        print("-------------------------------")
        if results and results != -1:
            track_map[vehicle[0]] = [results[0][0], results[0][1]]
            print(f"{vehicle[0]} found as ->  \n")
            for i, result in enumerate(results):
                id = result[0]
                distance = result[1]
                print(f"{id} [{distance}%]")

    return results_map



def compare_onnx_image_to_opensearch_db(image_path, vehicle_id, db):

    onnx_device = "CUDAExecutionProvider" # or CPUExecutionProvider


    global session
    if session is None:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            "/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.onnx",
            sess_options=so,
            providers=[onnx_device]
        )
    
    image = Image.open(image_path)
    X_image = data_transforms(image)  # shape (C, H, W)

    img_numpy = X_image.cpu().numpy().astype(np.float32)

    features_array = [extract_feature_onnx(session, img_numpy)]

    compare_array = []
    compare_array.append([vehicle_id, features_array[0]])


    track_map = {}
    results_map = []
    print("From intersection 2. -> 1. :")
    for vehicle in compare_array:
        results = db.query_vector(vehicle[1], k=3)
        results_map.append([vehicle[0],int(results[0][0]), results[0][1]])

        print("-------------------------------")
        if(results and results != -1):
            track_map[vehicle[0]] = [results[0][0], results[0][1]]
            print(f"{vehicle[0]} found as ->  \n")
            for i, result in enumerate(results):
                id = result[0]
                distance = result[1]
                print(f"{id} [{distance}%]")

    return results_map

def save_batch_to_opensearch_db(batch_image_paths, batch_vehicle_ids, db, saving_mode):
    import numpy as np

    device = "cuda"

    print("Initial Memory Usage:")
    print_gpu_memory()

    global model
    if not 'model' in globals():
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/result7/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/result7/net_10.pth")
        # print(model)
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch+loss_change4/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch+loss_change4/net_17.pth", remove_classifier=True)
        model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.pth", remove_classifier=True)
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/benchmark_model/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/benchmark_model/net_19.pth", remove_classifier=True)
        #print(model)
        model.eval()
        model.to(device)
        #print(model.classifier.add_block[2])
        model.classifier.add_block[2] = nn.Sequential()
        #print(model)

    images = [Image.open(path).convert("RGB") for path in batch_image_paths]
    X_images = torch.stack([data_transforms(img) for img in images]).to(device)

    features = extract_batch_features(model, X_images)  # (B, D)
    features = features.detach().cpu().numpy()

    print("Features array Memory Usage:")
    print_gpu_memory()

    if saving_mode in [1, 3]:
        for vehicle_id, feature_vector in zip(batch_vehicle_ids, features):
            db.insert(vehicle_id=vehicle_id, feature_vector=feature_vector, times_summed=0)
    else:
        print("Error! Not provisioned vector summing operation!")

    print("Save Memory Usage:")
    print_gpu_memory()

def compare_batch_to_opensearch_db(batch_image_paths, batch_vehicle_ids, db):
    import numpy as np

    device = "cuda"

    # print("Initial Memory Usage:")
    # print_gpu_memory()

    global model
    if not 'model' in globals():
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/result7/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/result7/net_10.pth")
        # print(model)
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch+loss_change4/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch+loss_change4/net_17.pth", remove_classifier=True)
        model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.pth", remove_classifier=True)
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/benchmark_model/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/benchmark_model/net_19.pth", remove_classifier=True)
        #print(model)
        model.eval()
        model.to(device)
        #print(model.classifier.add_block[2])
        model.classifier.add_block[2] = nn.Sequential()
        #print(model)

    images = [Image.open(path).convert("RGB") for path in batch_image_paths]
    X_images = torch.stack([data_transforms(img) for img in images]).to(device)

    features = extract_batch_features(model, X_images)  # (B, D)
    features = features.detach().cpu().numpy()

    compare_array = []
    for vehicle_id, feature_vector in zip(batch_vehicle_ids, features):
        compare_array.append([vehicle_id, feature_vector])
    
    track_map = {}
    results_map = []

    print("From intersection 2. -> 1. :")
    for vehicle in compare_array:
        results = db.query_vector(vehicle[1], k=3)
        results_map.append([vehicle[0], int(results[0][0]), results[0][1]])

        print("-------------------------------")
        if results and results != -1:
            track_map[vehicle[0]] = [results[0][0], results[0][1]]
            print(f"{vehicle[0]} found as ->  \n")
            for i, result in enumerate(results):
                id = result[0]
                distance = result[1]
                print(f"{id} [{distance}%]")

    return results_map

def compare_image_to_opensearch_db(image_path, vehicle_id, db):
    import numpy as np


    device = "cuda"

    global model
    if not 'model' in globals():
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch_change4/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch_change4/net_9.pth", remove_classifier=True)
        model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.pth", remove_classifier=True)
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/benchmark_model/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/benchmark_model/net_19.pth", remove_classifier=True)
        model.eval()
        model.to(device)
        model.classifier.add_block[2] = nn.Sequential()

    #print(image_path)
    images = [Image.open(image_path)]
    X_images = torch.stack(tuple(map(data_transforms, images))).to(device)

    features = [extract_feature(model, X) for X in X_images]
    features = torch.stack(features).detach().cpu()

    features_array = np.array(features)

    compare_array = []
    compare_array.append([vehicle_id, features_array[0]])


    track_map = {}
    results_map = []
    print("From intersection 2. -> 1. :")
    for vehicle in compare_array:
        results = db.query_vector(vehicle[1], k=3)
        results_map.append([vehicle[0],int(results[0][0]), results[0][1]])

        print("-------------------------------")
        if(results and results != -1):
            track_map[vehicle[0]] = [results[0][0], results[0][1]]
            print(f"{vehicle[0]} found as ->  \n")
            for i, result in enumerate(results):
                id = result[0]
                distance = result[1]
                print(f"{id} [{distance}%]")

    return results_map

def save_image_to_opensearch_db(image_path, vehicle_id, db, saving_mode):
    device = "cuda"

    print("Initial Memory Usage:")
    print_gpu_memory()

    global model
    if not 'model' in globals():
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/result7/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/result7/net_10.pth")
        # print(model)
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch+loss_change4/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch+loss_change4/net_17.pth", remove_classifier=True)
        model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.pth", remove_classifier=True)
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/benchmark_model/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/benchmark_model/net_19.pth", remove_classifier=True)
        #print(model)
        model.eval()
        model.to(device)
        #print(model.classifier.add_block[2])
        model.classifier.add_block[2] = nn.Sequential()
        #print(model)

    images = [Image.open(image_path)]
    X_images = torch.stack(tuple(map(data_transforms, images))).to(device)

    features = [extract_feature(model, X_images)]
    features = torch.stack(features).detach().cpu()

    features_array = np.array(features)

    # VEKTORA IZMERS DATUBAZEE TIEK NOMAINITS VEIDOJOT SHEEMU, NE KKUR KODAA

    print("Features array Memory Usage:")
    print_gpu_memory()

    if (saving_mode == 0) or (saving_mode == 2):
        print("Error! Not provisioned vector summing operation!><><><><><<><<><><><><><><><><><><><><><")
    elif (saving_mode == 1) or (saving_mode == 3):
        db.insert(vehicle_id = vehicle_id, feature_vector = features_array[0], times_summed = 0)

    print("Save Memory Usage:")
    print_gpu_memory()



def save_extractions_to_lance_db(folder_path, folder_name, saving_mode):
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
        model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/main_samples_pc_4_181225/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/main_samples_pc_4_181225/net_4.pth", remove_classifier=True)

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
        
def compare_extractions_to_lance_db(folder_path, queried_folder_name):
    import numpy as np
    import re
    #from misc.database import Vehicles
    import counting_workspace.misc.lance_db_init as create_db
    from counting_workspace.misc.lance_db_AICity import update_vehicle

    from docarray import DocList
    import numpy as np
    import lancedb

    device = "cuda"

    # start_time = time.time()

    #model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch_change4/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch_change4/net_22.pth", remove_classifier=True)

    if not 'model' in globals():
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/result7/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/result7/net_10.pth")
        # print(model)
        #model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch_change4/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch_change4/net_22.pth", remove_classifier=True)
        global model
        model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/main_samples_pc_4_181225/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/main_samples_pc_4_181225/net_4.pth", remove_classifier=True)

        #print(model)
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

    # duration = time.time() - start_time
    # print(f"[t]Load model + extract features took {duration*1000:.2f} ms.")
    # start_time = time.time()

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
    
    # duration = time.time() - start_time
    # print(f"[t]query vehicle in db took {duration*1000:.2f} ms.")

    #print("results map")
    #print(results_map)

    return results_map

def compare_extractions_to_lance_db_For_Rank(folder_path, queried_folder_name):
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
        model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.pth", remove_classifier=True)
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

        # Query top-n results that have cosinus similarity distance <= 0.50
        results = l_db.query_for_IDs(embedding, queried_folder_name, limit=100)
        #results = [r for r in results if r['_distance'] <= 0.6]
        
        if results and results != -1:
            retrieved_ids = [int(result['vehicle_id']) if result['_distance'] <= 0.6 else None for result in results] # Filter by distance threshold
            distances = [result['_distance'] if result['_distance'] <= 0.6 else None for result in results] # Filter by distance threshold

            results_map.append([image_id, retrieved_ids, distances])
            retrieved_ids = retrieved_ids[:5] # Take top-5
            distances = distances[:5]
            print(f"{retrieved_ids} [{distances}%]")

    return results_map  # Now contains Top-5 results per query
        
def save_image_to_lance_db(image_path, vehicle_id, folder_name, saving_mode):
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
        model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/Pidgeon_model_3_split_ids/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/Pidgeon_model_3_split_ids/net_12.pth", remove_classifier=True)
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

def compare_image_to_lance_db(image_path, vehicle_id, queried_folder_name):
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
        model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/Pidgeon_model_2_no_CE/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/Pidgeon_model_2_no_CE/net_5.pth", remove_classifier=True)
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

def save_image_to_lance_db_prune(image_path, vehicle_id, folder_name, saving_mode, idx_to_remove):
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

    print("Initial Memory Usage:")
    print_gpu_memory()

    global model
    if not 'model' in globals():
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/result7/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/result7/net_10.pth")
        # print(model)
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch+loss_change4/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/model_arch+loss_change4/net_17.pth", remove_classifier=True)
        model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.pth", remove_classifier=True)
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/benchmark_model/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/benchmark_model/net_19.pth", remove_classifier=True)
        #print(model)
        model.eval()
        model.to(device)
        #print(model.classifier.add_block[2])
        model.classifier.add_block[2] = nn.Sequential()
        #print(model)

    print("Load Memory Usage:")
    print_gpu_memory()

    images = [Image.open(image_path)]
    X_images = torch.stack(tuple(map(data_transforms, images))).to(device)

    print("Image stack Memory Usage:")
    print_gpu_memory()

    # print("X_images shape")
    # print(X_images.shape)

    features = [extract_feature(model, X_images)]
    features = torch.stack(features).detach().cpu()

    features_array = np.array(features)
    features_array = np.delete(features_array, idx_to_remove)

    features_size = features_array.shape[0]

    print("Features array Memory Usage:")
    print_gpu_memory()

    #print(f"features_array: {features_array}")

    db = create_db._init_(folder_name, features_size)

    if (saving_mode == 0) or (saving_mode == 2):
        update_vehicle(vehicle_id, features_array[0], folder_name, db)
    elif (saving_mode == 1) or (saving_mode == 3):
        add_vehicle(vehicle_id, features_array, folder_name, db)

    print("Save Memory Usage:")
    print_gpu_memory()

    #query(np.zeros(512))

def compare_image_to_lance_db_prune(image_path, vehicle_id, queried_folder_name, idx_to_remove):
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
        model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/veri+vehixlex_editTrainPar1/net_39.pth", remove_classifier=True)
        # model = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/benchmark_model/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo2/vehicle_reid/model/benchmark_model/net_19.pth", remove_classifier=True)
        model.eval()
        model.to(device)
        model.classifier.add_block[2] = nn.Sequential()

    #print(image_path)
    images = [Image.open(image_path)]
    X_images = torch.stack(tuple(map(data_transforms, images))).to(device)

    features = [extract_feature(model, X) for X in X_images]
    features = torch.stack(features).detach().cpu()

    features_array = np.array(features)
    features_array = np.delete(features_array, idx_to_remove)

    features_size = features_array.shape[0]

    #print(f"features_array: {features_array}")

    db = create_db._init_(queried_folder_name, features_size)


    compare_array = []
    compare_array.append([vehicle_id, features_array])


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


    print("Compared Memory Usage:")
    print_gpu_memory()
    return results_map

def save_image_to_lance_db_manual_features(image_path, vehicle_id, folder_name, saving_mode, feature_type=None):

    import counting_workspace.misc.lance_db_init as create_db
    from counting_workspace.misc.lance_db_AICity import update_vehicle
    from counting_workspace.misc.lance_db_AICity import add_vehicle

    # Load and process image
    img = np.array(Image.open(image_path))
    features = extract_manual_features(img, feature_type=feature_type)
    

    features_array = np.array(features)

    features_size = features_array.shape[0]
    #print(features_size)
    
    
    db = create_db._init_(folder_name, features_size)
    
    if (saving_mode == 0) or (saving_mode == 2):
        update_vehicle(vehicle_id, features_array, folder_name, db)
    elif (saving_mode == 1) or (saving_mode == 3):
        add_vehicle(vehicle_id, features_array, folder_name, db)
        

def compare_image_to_lance_db_manual_features(image_path, vehicle_id, queried_folder_name, feature_type=None):
    import counting_workspace.misc.lance_db_init as create_db
    from counting_workspace.misc.lance_db_AICity import update_vehicle
    """
    Compares an image's extracted features to a LanceDB database.
    
    :param image_path: Path to the input image.
    :param vehicle_id: ID of the vehicle in the image.
    :param queried_folder_name: Folder name for querying the database.
    :param feature_type: Feature type to use for comparison (HOG, LBP, RGB, H10).
    :return: List of matched vehicle IDs with distances.
    """
    img = np.array(Image.open(image_path))
    features = extract_manual_features(img, feature_type=feature_type)
    

    features_array = np.array(features)

    features_size = features_array.shape[0]
    #print(features_size)
    
    # Initialize database
    db = create_db._init_(queried_folder_name, features_size)
    
    # Query database
    results = l_db.query_for_IDs(features_array, queried_folder_name, limit=3)
    
    # Process results
    results_map = []
    if results and results != -1:
        print("-------------------------------")
        results_map.append([vehicle_id, int(results[0]['vehicle_id']), results[0]['_distance']])
        print(f"{vehicle_id} found as ->")
        for result in results:
            print(f"{result['vehicle_id']} [{result['_distance']}%]")
    
    return results_map

def save_image_to_lance_db_linear_features(image_path, vehicle_id, folder_name, saving_mode, feature_type=None):

    import counting_workspace.misc.lance_db_init as create_db
    from counting_workspace.misc.lance_db_AICity import update_vehicle
    from counting_workspace.misc.lance_db_AICity import add_vehicle

    # Load and process image
    img = np.array(Image.open(image_path))
    features = extract_linear_features(img, feature_type=feature_type)
    

    features_array = np.array(features)

    features_size = features_array.shape[0]
    #print(features_size)
    
    
    db = create_db._init_(folder_name, features_size)
    
    if (saving_mode == 0) or (saving_mode == 2):
        update_vehicle(vehicle_id, features_array, folder_name, db)
    elif (saving_mode == 1) or (saving_mode == 3):
        add_vehicle(vehicle_id, features_array, folder_name, db)
    
def compare_image_to_lance_db_linear_features(image_path, vehicle_id, queried_folder_name, feature_type=None):
    import counting_workspace.misc.lance_db_init as create_db
    from counting_workspace.misc.lance_db_AICity import update_vehicle
    """
    Compares an image's extracted features to a LanceDB database.
    
    :param image_path: Path to the input image.
    :param vehicle_id: ID of the vehicle in the image.
    :param queried_folder_name: Folder name for querying the database.
    :param feature_type: Feature type to use for comparison (HOG, LBP, RGB, H10).
    :return: List of matched vehicle IDs with distances.
    """
    img = np.array(Image.open(image_path))
    features = extract_linear_features(img, feature_type=feature_type)
    

    features_array = np.array(features)

    features_size = features_array.shape[0]
    #print(features_size)
    
    # Initialize database
    db = create_db._init_(queried_folder_name, features_size)
    
    # Query database
    results = l_db.query_for_IDs(features_array, queried_folder_name, limit=3)
    
    # Process results
    results_map = []
    if results and results != -1:
        print("-------------------------------")
        results_map.append([vehicle_id, int(results[0]['vehicle_id']), results[0]['_distance']])
        print(f"{vehicle_id} found as ->")
        for result in results:
            print(f"{result['vehicle_id']} [{result['_distance']}%]")
    
    return results_map
