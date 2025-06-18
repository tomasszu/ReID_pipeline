# Object Detecion 
import cv2

#basics
import pandas as pd
import os

import time

import counting_workspace.misc.crop_AICity as detection_crop
#No CLIP
import counting_workspace.misc.feature_extract_AICity as fExtract
#With CLIP
# import counting_workspace.misc.feature_extract_AICity_CLIP as fExtract
#For ModelArchChange - removing all classification head
# import counting_workspace.misc.feature_extract_AICity_ModelArchChange_ForInfer as fExtract

from counting_workspace.misc.opensearch_db import Opensearch_db

db = Opensearch_db("localhost", 9200, ("admin", "admin"))


#SAVING MODE OPTIONS: 0 - complete summing of all vectors of one vehicle in one
#SAVING MODE OPTIONS: 1 - complete saving of all vectors of one vehicle independently
#SAVING MODE OPTIONS: 2 - summing vectors of vehicle in different zones only
#SAVING MODE OPTIONS: 3 - saving all vectors of vehicle in different zones only
saving_mode = 3

total_iters = 0
accumulative_accuracy = 0

# Initialize global variables for macro and micro averaging
total_true_positives = 0
total_false_positives = 0
total_false_negatives = 0
# Initialize lists to store class precisions and recalls for macro averaging
# Using lists to store precision and recall values for each class
# This will allow us to calculate macro-averaged precision and recall later
class_precisions = {}
class_recalls = {}
# Initialize class counts for macro averaging
# Using a dictionary to store counts for each class
# This will allow us to track true positives, false positives, and false negatives for each class
class_counts = {}


def results(results_map):
    global total_iters, accumulative_accuracy, total_true_positives, total_false_positives, total_false_negatives
    global class_recalls, class_counts, class_precisions

    frame_findings = len(results_map)

    # Increment total queries
    
    total_iters += frame_findings  # instead of += 1 (since we are processing batch results at once)


    if frame_findings:
        frame_true_positives = 0
        frame_false_positives = 0
        frame_false_negatives = 0


        for result in results_map:
            id1, id2, distance = result
            
            # Update accuracy as before
            if id1 == id2:
                accumulative_accuracy += 1
                frame_true_positives += 1
            else:
                frame_false_positives += 1
                frame_false_negatives += 1
            
            # Track total number of instances per class for macro averaging
            if id1 not in class_counts:
                class_counts[id1] = {"TP": 0, "FP": 0, "FN": 0}
            if id2 not in class_counts:
                class_counts[id2] = {"TP": 0, "FP": 0, "FN": 0}

            # Update true positives, false positives, and false negatives for each class
            if id1 == id2:
                class_counts[id1]["TP"] += 1
            else:
                class_counts[id1]["FN"] += 1
                class_counts[id2]["FP"] += 1

        # Update micro-averaged precision and recall
        total_true_positives += frame_true_positives
        total_false_positives += frame_false_positives
        total_false_negatives += frame_false_negatives

        # Calculate micro-averaged precision and recall
        if total_true_positives + total_false_positives > 0:
            micro_precision = total_true_positives / (total_true_positives + total_false_positives)
        else:
            micro_precision = 0

        if total_true_positives + total_false_negatives > 0:
            micro_recall = total_true_positives / (total_true_positives + total_false_negatives)
        else:
            micro_recall = 0

        # Update macro-averaged precision and recall
        for class_id, counts in class_counts.items():
            tp = counts["TP"]
            fp = counts["FP"]
            fn = counts["FN"]

            if tp + fp > 0:
                class_precision = tp / (tp + fp)
            else:
                class_precision = 0

            if tp + fn > 0:
                class_recall = tp / (tp + fn)
            else:
                class_recall = 0

            class_precisions[class_id] = class_precision
            class_recalls[class_id] = class_recall

            # macro computation
            if class_precisions:
                macro_precision = sum(class_precisions.values()) / len(class_precisions)
                macro_recall = sum(class_recalls.values()) / len(class_recalls)
            else:
                macro_precision = 0
                macro_recall = 0


        # Print results
        print("Rank-1 accuracy:", accumulative_accuracy / total_iters)
        print("Micro Precision | Micro Recall | Macro Precision | Macro Recall")
        print(f"{micro_precision:.4f}         {micro_recall:.4f}        {macro_precision:.4f}         {macro_recall:.4f}")


data_dir = '/home/tomass/tomass/data'

ground_truths_path_1 = "/home/tomass/tomass/data/EDI_Cam_testData/cam1.csv"

ground_truths_path_2 = "/home/tomass/tomass/data/EDI_Cam_testData/cam2.csv"


ground_truths_path_3 = "/home/tomass/tomass/data/EDI_Cam_testData/cam3.csv"


file1 = pd.read_csv(ground_truths_path_1)
file2 = pd.read_csv(ground_truths_path_2)
file3 = pd.read_csv(ground_truths_path_3)

seen_vehicle_ids = [20]


BATCH_SIZE = 1  # Choose based on memory/performance
batch_image_paths = []
batch_vehicle_ids = []

#_________________________________________________________________________________________#
# Turn vehicles from camera y, z, ... (gallery cameras) into embeddings and save in DB:

start_time = time.time()

for index, row in file1.iterrows():
    image_path = os.path.join(data_dir, row['path'])
    vehicle_id = row['ID']
    
    if vehicle_id not in seen_vehicle_ids:
        seen_vehicle_ids.append(vehicle_id)
    
    batch_image_paths.append(image_path)
    batch_vehicle_ids.append(vehicle_id)

    if len(batch_image_paths) >= BATCH_SIZE:
        fExtract.save_onnx_batch_to_opensearch_db(batch_image_paths, batch_vehicle_ids, db, saving_mode)
        batch_image_paths.clear()
        batch_vehicle_ids.clear()

# Process any remaining images
if batch_image_paths:
    fExtract.save_onnx_batch_to_opensearch_db(batch_image_paths, batch_vehicle_ids, db, saving_mode)
    batch_image_paths.clear()
    batch_vehicle_ids.clear()

end_time = time.time()
print(f"Time taken to save images from camera 1: {end_time - start_time} seconds")

start_time = time.time()

for index, row in file2.iterrows():
    image_path = os.path.join(data_dir, row['path'])
    vehicle_id = row['ID']
    
    if vehicle_id not in seen_vehicle_ids:
        seen_vehicle_ids.append(vehicle_id)
    
    batch_image_paths.append(image_path)
    batch_vehicle_ids.append(vehicle_id)

    if len(batch_image_paths) >= BATCH_SIZE:
        fExtract.save_onnx_batch_to_opensearch_db(batch_image_paths, batch_vehicle_ids, db, saving_mode)
        batch_image_paths.clear()
        batch_vehicle_ids.clear()

# Process any remaining images
if batch_image_paths:
    fExtract.save_onnx_batch_to_opensearch_db(batch_image_paths, batch_vehicle_ids, db, saving_mode)
    batch_image_paths.clear()
    batch_vehicle_ids.clear()

end_time = time.time()
print(f"Time taken to save images from camera 1: {end_time - start_time} seconds")

start_time = time.time()

for index, row in file3.iterrows():
    image_path = row['path']  # Get the image path
    vehicle_id = row['ID']     # Get the vehicle ID

    image_path = os.path.join(data_dir, image_path)

    if vehicle_id in seen_vehicle_ids:
        batch_image_paths.append(image_path)
        batch_vehicle_ids.append(vehicle_id)

        if len(batch_image_paths) >= BATCH_SIZE:
            results_map = fExtract.compare_onnx_batch_to_opensearch_db(batch_image_paths, batch_vehicle_ids, db)
            batch_image_paths.clear()
            batch_vehicle_ids.clear()
            results(results_map)

# Process any remaining images
if batch_image_paths:
    results_map = fExtract.compare_onnx_batch_to_opensearch_db(batch_image_paths, batch_vehicle_ids, db)
    batch_image_paths.clear()
    batch_vehicle_ids.clear()
    results(results_map)

end_time = time.time()
print(f"Time taken to compare images from camera 3: {end_time - start_time} seconds")

# _______________________________________________________________________________________#

    #print(seen_vehicle_ids)

    # resized = cv2.resize(labeled_frame1, (1280, 720))
    # cv2.imshow("frame1", resized)

    # resized2 = cv2.resize(labeled_frame2, (1280, 720))
    # cv2.imshow("frame2", resized2)
    # cv2.waitKey(0)