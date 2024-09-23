# Object Detecion 
import cv2
import sys
sys.path.append("/home/toms.zinars/Anzelika/counting_workspace/")

# basics
import pandas as pd
import numpy as np
import os

import counting_workspace.misc.crop_AICity as detection_crop
import counting_workspace.misc.feature_extract_AICity as fExtract

# Saving mode options: 0, 1, 2, 3 are predefined strategies for saving
saving_mode = 3

# Initialize global variables for accumulative accuracy and iterations
total_iters = 0
accumulative_accuracy = 0
accumulative_top1 = 0

# Global variables for micro and macro averaging
total_true_positives = 0
total_false_positives = 0
total_false_negatives = 0
class_precisions = []
class_recalls = []
class_counts = {}
id_counts_total = 0



def results(results_map):
    frame_findings = len(results_map)
    if frame_findings:
        frame_accuracy = 0
        top1_acc = 0
        frame_true_positives = 0
        frame_false_positives = 0
        frame_false_negatives = 0
        id_counts = 0

        # Processing the results
        for idx, result in enumerate(results_map):
            id1, id2, distance = result
            

            # We care only about the first (top1) result
            if results_map[0]:
                print(str(results_map[0]))
                print(str(id1))
                print(str(id2))
                id_counts += 1
                if str(id1) == str(id2):
                    # Top-1 correct match: add 1 to the accuracy
                    top1_acc += 1
                    frame_true_positives += 1
                    print(str(frame_true_positives))
                else:
                    frame_false_positives += 1
                    frame_false_negatives += 1

            # Track total number of instances per class for macro averaging
            if id1 not in class_counts:
                class_counts[id1] = {"TP": 0, "FP": 0, "FN": 0}
            if id2 not in class_counts:
                class_counts[id2] = {"TP": 0, "FP": 0, "FN": 0}

            # Update true positives, false positives, and false negatives for each class
            if str(id1) == str(id2):
                class_counts[id1]["TP"] += 1
            else:
                class_counts[id1]["FN"] += 1
                class_counts[id2]["FP"] += 1

        # Update global accumulative metrics
        global total_iters, accumulative_accuracy, accumulative_top1, id_counts_total
        total_iters += 1
        accumulative_top1 += top1_acc
        id_counts_total += id_counts

        # Update micro-averaged precision and recall
        global total_true_positives, total_false_positives, total_false_negatives
        total_true_positives += frame_true_positives
        total_false_positives += frame_false_positives
        total_false_negatives += frame_false_negatives

        print('Total True' + str(total_true_positives))
        print('Total False Positives' + str(total_false_positives))
        print('Total False Negatives' + str(total_false_negatives))

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
            # print(str(tp))
            # print(str(fp))
            # print(str(fn))

            if tp + fp > 0:
                print(str(tp+fp))
                print(tp)
                class_precision = tp / (tp + fp)
                print(class_precision)
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
            print(str(sum(class_precisions)))
            print(str(len(class_precisions)))
            macro_recall = sum(class_recalls) / len(class_recalls)
        else:
            macro_precision = 0
            macro_recall = 0

        # Output results
        print("Frame precision: ", frame_accuracy, "Out of: ", frame_findings, " frame findings")

        # Calculate total accuracy based on top-1 results
        total_top1_acc = accumulative_top1 / id_counts_total

        print("Accuracy (Top-1): ", total_top1_acc, "Id_counts ", id_counts_total)
        print("Micro Precision|Micro Recall|Macro Precision|Macro Recall")
        print(micro_precision, "         ", micro_recall, "        ", macro_precision, "         ", macro_recall)


data_dir1 = '/home/toms.zinars/Anzelika/Pirmais_datasets_tris/test_vaicajums/'
data_dir2 = '/home/toms.zinars/Anzelika/Pirmais_datasets_tris/test_search/'

ground_truths_path_1 = "/home/toms.zinars/Anzelika/Pirmais_datasets_tris/test_vaicajums.csv"
ground_truths_path_2 = "/home/toms.zinars/Anzelika/Pirmais_datasets_tris/test_search.csv"


file1 = pd.read_csv(ground_truths_path_1)
file2 = pd.read_csv(ground_truths_path_2)


seen_vehicle_ids = [20]

# _________________________________________________________________________________________#
# Turn vehicles from gallery cameras (cam1, cam3) into embeddings and save in DB:

for index, row in file1.iterrows():
    image_path = row['path']  # Get the image path
    vehicle_id = row['ID']    # Get the vehicle ID

    image_path = os.path.join(data_dir1, image_path)

    if vehicle_id not in seen_vehicle_ids:
        seen_vehicle_ids.append(vehicle_id)

    fExtract.save_image_to_lance_db(image_path, vehicle_id, 3, saving_mode)


# _________________________________________________________________________________________#
# Turn vehicles from query camera (cam2) into embeddings and search in DB:

for index, row in file2.iterrows():
    image_path = row['path']  # Get the image path
    vehicle_id = row['ID']    # Get the vehicle ID

    image_path = os.path.join(data_dir2, image_path)

    if vehicle_id in seen_vehicle_ids:
        results_map = fExtract.compare_image_to_lance_db(image_path, vehicle_id, 3)
        results(results_map)

