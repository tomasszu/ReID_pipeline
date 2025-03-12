# Object Detecion 
import cv2

#basics
import pandas as pd
import numpy as np
import os

import counting_workspace.misc.crop_AICity as detection_crop
#Basic one model extraction, arch unchanged
import counting_workspace.misc.feature_extract as fExtract
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
accumulative_top1 = 0


def results(results_map):
    frame_findings = len(results_map)
    if(frame_findings):
        frame_accuracy = 0
        top1_acc = 0
        for result in results_map:
            id1, id2, distance = result
            if(id1 != id2):
                frame_accuracy += 0
            else:
                frame_accuracy += (1 - distance)
                top1_acc += 1
        if(frame_accuracy > 0):
            frame_accuracy = frame_accuracy / frame_findings
            top1_acc = top1_acc / frame_findings
        print("Frame precision: ", frame_accuracy, "Out of: ", frame_findings, " frame findings" )
        global total_iters
        total_iters += 1
        global accumulative_accuracy
        global accumulative_top1
        accumulative_accuracy += frame_accuracy
        accumulative_top1 += top1_acc
        if(accumulative_accuracy != 0 and total_iters != 0):
            total_accuracy = accumulative_accuracy / total_iters
            total_top1_acc = accumulative_top1 / total_iters
        else:
            total_accuracy = 0
            total_top1_acc = 0

        print("Accuracy: ", total_top1_acc, "Out of: ", total_iters, " frames" )
        print("(Accuracy*Confidence: ", total_accuracy, "Out of: ", total_iters, " frames)" )

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
        results(results_map)

# _______________________________________________________________________________________#

    #print(seen_vehicle_ids)

    # resized = cv2.resize(labeled_frame1, (1280, 720))
    # cv2.imshow("frame1", resized)

    # resized2 = cv2.resize(labeled_frame2, (1280, 720))
    # cv2.imshow("frame2", resized2)
    # cv2.waitKey(0)