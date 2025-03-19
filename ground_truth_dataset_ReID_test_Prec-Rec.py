# Object Detecion 
import cv2

#basics
import pandas as pd
import numpy as np
import os

import counting_workspace.misc.crop_AICity as detection_crop
#No CLIP
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

total_iters = 0
accumulative_accuracy = 0

# Initialize global variables for macro and micro averaging
total_true_positives = 0
total_false_positives = 0
total_false_negatives = 0
class_precisions = []
class_recalls = []
class_counts = {}


def results(results_map):
    frame_findings = len(results_map)
    if frame_findings:
        frame_accuracy = 0
        frame_true_positives = 0
        frame_false_positives = 0
        frame_false_negatives = 0

        # Increment total queries
        global total_iters, accumulative_accuracy
        total_iters += 1

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

        if frame_accuracy > 0:
            frame_accuracy /= frame_findings


        # Update micro-averaged precision and recall
        global total_true_positives, total_false_positives, total_false_negatives
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

            class_precisions.append(class_precision)
            class_recalls.append(class_recall)

        if len(class_precisions) > 0:
            macro_precision = sum(class_precisions) / len(class_precisions)
            macro_recall = sum(class_recalls) / len(class_recalls)
        else:
            macro_precision = 0
            macro_recall = 0

        # Output results
        print("Rank-1 accuracy: ",accumulative_accuracy/total_iters )
        
        print("Micro Precision|Micro Recall|Macro Precision|Macro Recall")
        print(micro_precision,"         ",micro_recall,"        ",macro_precision,"         ",macro_recall)
        # print("Micro Precision: ", micro_precision)
        # print("Micro Recall: ", micro_recall)
        # print("Macro Precision: ", macro_precision)
        # print("Macro Recall: ", macro_recall)

data_dir = '/home/tomass/tomass/data'

ground_truths_path_1 = "/home/tomass/tomass/data/EDI_Cam_testData/cam1.csv"

ground_truths_path_2 = "/home/tomass/tomass/data/EDI_Cam_testData/cam2.csv"


ground_truths_path_3 = "/home/tomass/tomass/data/EDI_Cam_testData/cam3.csv"


file1 = pd.read_csv(ground_truths_path_1)
file2 = pd.read_csv(ground_truths_path_2)
file3 = pd.read_csv(ground_truths_path_3)

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

for index, row in file3.iterrows():
    image_path = row['path']  # Get the image path
    vehicle_id = row['ID']     # Get the vehicle ID

    image_path = os.path.join(data_dir, image_path)
    
    if vehicle_id not in seen_vehicle_ids:
        seen_vehicle_ids.append(vehicle_id)
    
    fExtract.save_image_to_lance_db(image_path, vehicle_id, 1, saving_mode)

# _____________________________________________________________________________________#
# Turn vehicles from camera x (query camera) into embeddings and search in DB:

for index, row in file2.iterrows():
    image_path = row['path']  # Get the image path
    vehicle_id = row['ID']     # Get the vehicle ID

    image_path = os.path.join(data_dir, image_path)

    if vehicle_id in seen_vehicle_ids:
        results_map = fExtract.compare_image_to_lance_db(image_path, vehicle_id, 1)
        results(results_map)

# _______________________________________________________________________________________#

    #print(seen_vehicle_ids)

    # resized = cv2.resize(labeled_frame1, (1280, 720))
    # cv2.imshow("frame1", resized)

    # resized2 = cv2.resize(labeled_frame2, (1280, 720))
    # cv2.imshow("frame2", resized2)
    # cv2.waitKey(0)