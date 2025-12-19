
#basics
import pandas as pd
import numpy as np
import os
import shutil

from sklearn.metrics import roc_auc_score

import counting_workspace.misc.crop_AICity as detection_crop
#Basic one model extraction, arch unchanged
import counting_workspace.misc.feature_extract_AICity as fExtract
#With CLIP
# import counting_workspace.misc.feature_extract_AICity_CLIP as fExtract
#For ModelArchChange - removing all classification head
# import counting_workspace.misc.feature_extract_AICity_ModelArchChange_ForInfer as fExtract


#SAVING MODE OPTIONS: 0 - complete summing of all vectors of one vehicle in one
#SAVING MODE OPTIONS: 1 - complete saving of all vectors of one vehicle independently
#SAVING MODE OPTIONS: 2 - summing vectors of vehicle in different zones only
#SAVING MODE OPTIONS: 3 - saving all vectors of vehicle in different zones only
saving_mode = 3

total_queries = 0
rank1_correct = 0

all_labels = []     # 1 = correct match, 0 = incorrect
all_scores = []     # similarity scores (higher = better)


def update_reid_metrics(results_map):
    global total_queries, rank1_correct
    global all_labels, all_scores

    if not results_map:
        return

    for gt_id, nn_id, dist in results_map:
        total_queries += 1

        is_correct = int(gt_id == nn_id)
        rank1_correct += is_correct

        # Rank-1
        rank1_acc = rank1_correct / total_queries

        # For ROC/AUC: higher score = more likely same-ID
        similarity = -dist

        all_labels.append(is_correct)
        all_scores.append(similarity)

    # ROC/AUC only makes sense if both classes exist
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_scores)
    else:
        auc = float("nan")

    print(f"Rank-1 Accuracy: {rank1_acc:.4f}")
    print(f"ROC AUC:        {auc:.4f}")

data_dir = '/home/tomass/tomass/data'

ground_truths_path_1 = "/home/tomass/tomass/data/EDI_Cam_testData/cam1.csv"

ground_truths_path_2 = "/home/tomass/tomass/data/EDI_Cam_testData/cam2.csv"


ground_truths_path_3 = "/home/tomass/tomass/data/EDI_Cam_testData/cam3.csv"


file1 = pd.read_csv(ground_truths_path_3)
file2 = pd.read_csv(ground_truths_path_2)
file3 = pd.read_csv(ground_truths_path_1)

seen_vehicle_ids = [20]

#_________________________________________________________________________________________#
# Turn vehicles from camera y, z, ... (gallery cameras) into embeddings and save in DB:

for index, row in file1.iterrows():
    image_path = row['path']  # Get the image path
    vehicle_id = row['ID']     # Get the vehicle ID

    image_path = os.path.join(data_dir, image_path)
    
    if vehicle_id not in seen_vehicle_ids:
        seen_vehicle_ids.append(vehicle_id)
    
    fExtract.save_image_to_lance_db(image_path, vehicle_id, 1, saving_mode)

for index, row in file2.iterrows():
    image_path = row['path']  # Get the image path
    vehicle_id = row['ID']     # Get the vehicle ID

    image_path = os.path.join(data_dir, image_path)
    
    if vehicle_id not in seen_vehicle_ids:
        seen_vehicle_ids.append(vehicle_id)
    
    fExtract.save_image_to_lance_db(image_path, vehicle_id, 1, saving_mode)

# _____________________________________________________________________________________#
# Turn vehicles from camera x (query camera) into embeddings and search in DB:

for index, row in file3.iterrows():
    image_path = row['path']  # Get the image path
    vehicle_id = row['ID']     # Get the vehicle ID

    image_path = os.path.join(data_dir, image_path)

    if vehicle_id in seen_vehicle_ids:
        results_map = fExtract.compare_image_to_lance_db(image_path, vehicle_id, 1)
        update_reid_metrics(results_map)

# _______________________________________________________________________________________#

# DELETING the lancedb folder automatically after use (at: /home/tomass/tomass/ReID_pipele/lancedb)

shutil.rmtree("/home/tomass/tomass/ReID_pipele/lancedb")



    #print(seen_vehicle_ids)

    # resized = cv2.resize(labeled_frame1, (1280, 720))
    # cv2.imshow("frame1", resized)

    # resized2 = cv2.resize(labeled_frame2, (1280, 720))
    # cv2.imshow("frame2", resized2)
    # cv2.waitKey(0)