# Object Detection 
import cv2

# basics
import pandas as pd
import numpy as np
import os
import shutil

import counting_workspace.misc.crop_AICity as detection_crop
# No CLIP
import counting_workspace.misc.feature_extract_AICity as fExtract
# With CLIP
# import counting_workspace.misc.feature_extract_AICity_CLIP as fExtract
# For ModelArchChange - removing all classification head
# import counting_workspace.misc.feature_extract_AICity_ModelArchChange_ForInfer as fExtract


# SAVING MODE OPTIONS: 0 - complete summing of all vectors of one vehicle in one
# SAVING MODE OPTIONS: 1 - complete saving of all vectors of one vehicle independently
# SAVING MODE OPTIONS: 2 - summing vectors of vehicle in different zones only
# SAVING MODE OPTIONS: 3 - saving all vectors of vehicle in different zones only
saving_mode = 3

total_iters = 0
accumulative_accuracy = 0

# Initialize global variables for macro and micro averaging
class_precisions = []
class_recalls = []
class_counts = {}


def results(results_map):
    frame_findings = len(results_map)
    if frame_findings:

        # Increment total queries
        global total_iters, accumulative_accuracy
        total_iters += 1

        for result in results_map:
            id1, id2, distance = result
            
            # Update accuracy as before
            if id1 == id2:
                accumulative_accuracy += 1

        # Output the Rank-1 accuracy directly
        return accumulative_accuracy / total_iters


data_dir = '/home/tomass/tomass/data'

ground_truths_path_1 = "/home/tomass/tomass/data/EDI_Cam_testData/cam1.csv"

ground_truths_path_2 = "/home/tomass/tomass/data/EDI_Cam_testData/cam2.csv"


ground_truths_path_3 = "/home/tomass/tomass/data/EDI_Cam_testData/cam3.csv"

vector_db_folder = '/home/tomass/tomass/ReID_pipele/lancedb'


file1 = pd.read_csv(ground_truths_path_1)
file2 = pd.read_csv(ground_truths_path_2)
file3 = pd.read_csv(ground_truths_path_3)

seen_vehicle_ids = [20]

# Try pruning one feature at a time
for i in range(255):

    #_________________________________________________________________________________________#
    # Turn vehicles from camera y, z, ... (gallery cameras) into embeddings and save in DB:

    for index, row in file1.iterrows():
        image_path = row['path']  # Get the image path
        vehicle_id = row['ID']     # Get the vehicle ID

        image_path = os.path.join(data_dir, image_path)
        
        if vehicle_id not in seen_vehicle_ids:
            seen_vehicle_ids.append(vehicle_id)
        
        fExtract.save_image_to_lance_db_prune(image_path, vehicle_id, 1, saving_mode, i)

    for index, row in file3.iterrows():
        image_path = row['path']  # Get the image path
        vehicle_id = row['ID']     # Get the vehicle ID

        image_path = os.path.join(data_dir, image_path)
        
        if vehicle_id not in seen_vehicle_ids:
            seen_vehicle_ids.append(vehicle_id)
        
        fExtract.save_image_to_lance_db_prune(image_path, vehicle_id, 1, saving_mode, i)

    # _____________________________________________________________________________________#
    # Turn vehicles from camera x (query camera) into embeddings and search in DB:

    for index, row in file2.iterrows():
        image_path = row['path']  # Get the image path
        vehicle_id = row['ID']     # Get the vehicle ID

        image_path = os.path.join(data_dir, image_path)

        if vehicle_id in seen_vehicle_ids:
            results_map = fExtract.compare_image_to_lance_db_prune(image_path, vehicle_id, 1, i)
            results(results_map)

    # Check if the folder exists
    if os.path.exists(vector_db_folder):
        # Use shutil.rmtree to delete the folder and all its contents
        shutil.rmtree(vector_db_folder)
        print(f"Folder {vector_db_folder} has been deleted.")
    else:
        print(f"Folder {vector_db_folder} does not exist.")

    print(i, " : ", results(results_map))
    # Open the file in append mode ('a') to add the output to the file without overwriting it
    with open("prune_results.txt", "a") as file:
        file.write(f"{i} : {results(results_map)}\n")

    total_iters = 0
    accumulative_accuracy = 0