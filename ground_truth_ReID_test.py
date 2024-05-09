# Object Detecion 
import glob
import cv2
import supervision as sv
from ultralytics import YOLO

#basics
import pandas as pd
import numpy as np
import os
import sys
import re
import copy

import counting_workspace.misc.crop_AICity as detection_crop
#No CLIP
# import counting_workspace.misc.feature_extract_AICity as fExtract
#With CLIP
#import counting_workspace.misc.feature_extract_AICity_CLIP as fExtract
#For ModelArchChange
import counting_workspace.misc.feature_extract_AICity_ModelArchChange_ForInfer as fExtract


#SAVING MODE OPTIONS: 0 - complete summing of all vectors of one vehicle in one
#SAVING MODE OPTIONS: 1 - complete saving of all vectors of one vehicle independently
#SAVING MODE OPTIONS: 2 - summing vectors of vehicle in different zones only
#SAVING MODE OPTIONS: 3 - saving all vectors of vehicle in different zones only
saving_mode = 3

total_iters = 0
accumulative_accuracy = 0

def results(results_map):
    frame_findings = len(results_map)
    if(frame_findings):
        frame_accuracy = 0
        for result in results_map:
            id1, id2, distance = result
            if(id1 != id2):
                frame_accuracy += 0
            else:
                frame_accuracy += (1 - distance)
        if(frame_accuracy > 0):
            frame_accuracy = frame_accuracy / frame_findings
        print("Frame accuracy: ", frame_accuracy, "Out of: ", frame_findings, " frame findings" )
        global total_iters
        total_iters += 1
        global accumulative_accuracy
        accumulative_accuracy += frame_accuracy
        if(accumulative_accuracy != 0 and total_iters != 0):
            total_accuracy = accumulative_accuracy / total_iters
        else:
            total_accuracy = 0
        print("Total accuracy: ", total_accuracy, "Out of: ", total_iters, " frames" )


def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    return x1 ,y1, x2, y2



def draw_bbox(frame, bbox, id):

    x1,y1,x2,y2 = xywh_to_xyxy(bbox)

    color = (0, 255, 0)  # BGR color for the bounding box (green in this case)
    thickness = 2  # Thickness of the bounding box lines

    # Draw the bounding box on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 2
    text_color = (255, 255, 255)  # BGR color for the text (white in this case)

    # Put the object ID next to the bounding box
    text = f'ID: {id}'
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = x1
    text_y = y1 - 5  # Adjust this value for the vertical position of the text

    cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

def draw_zone(frame, bbox):

    x1,y1,x2,y2 = bbox

    color = (255, 0, 0)  # BGR color for the bounding box (red in this case)
    thickness = 2  # Thickness of the bounding box lines

    # Draw the bounding box on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

def draw_point(frame, point):
    cv2.circle(frame, point, 5, (0, 0, 255), -1)  # Red circles with a radius of 5 pixels


def ground_truth_for_frame(frame_id, last_read, frame_nr, curr_line, un_labeled_frame, lines, seen_ids=None):
    croppable_detections = [] #kur saglabāt izgriežamos bbokšus
    frame = un_labeled_frame.copy()
    if(seen_ids == None or len(seen_ids)):
        if(last_read != 0 and frame_id == frame_nr):
            #print("output:",frame_nr, curr_line)
            if((seen_ids == None) or (int(curr_line[1]) in seen_ids)):
                draw_bbox(frame, [int(curr_line[2]),int(curr_line[3]),int(curr_line[4]),int(curr_line[5])], int(curr_line[1]))
                croppable_detections.append([frame_nr, int(curr_line[1]), xywh_to_xyxy([int(curr_line[2]),int(curr_line[3]),int(curr_line[4]),int(curr_line[5])])])
        if(last_read == 0 or frame_id == frame_nr):
            while frame_id == frame_nr:
                line = lines[last_read]
                curr_line = line.split(",", maxsplit=6)
                frame_id = int(curr_line[0])
                vehicle_id = int(curr_line[1])
                xywh = [int(curr_line[2]),int(curr_line[3]),int(curr_line[4]),int(curr_line[5])]
                if(frame_id == frame_nr):
                    #print("output:",frame_nr, curr_line)
                    last_read = last_read+1 # !!!!!!! Moš kkas tiek izlaists cauri apakšējā pārbaudē, bet kopumā izskatās, ka ir ok
                    #Ja ReID daļā (2. krust) tad paarbaudam vai tads id vispār ir pirmstam piefiksēts               
                    if((seen_ids == None) or (vehicle_id in seen_ids)):
                        draw_bbox(frame, xywh, vehicle_id)
                        croppable_detections.append([frame_nr, vehicle_id, xywh_to_xyxy(xywh)])
                else:
                    last_read = last_read+1
                    break
    return frame_id, last_read, curr_line, frame, croppable_detections

def zone_of_point(zones, point):
    """
    Determine the zone index in which a given point lies.
    
    Args:
    - point (tuple): (x, y) coordinates of the point
    - zones (list): List of zone coordinates, each zone represented as ((x1, y1), (x2, y2))
    
    Returns:
    - zone_index (int): Index of the zone in which the point lies, or -1 if it's not in any zone
    """
    x, y = point
    
    # Iterate through each zone and check if the point lies within it
    for zone_index, zone in enumerate(zones):
        x1, y1, x2, y2 = zone
        if x1 <= x <= x2 and y1 <= y <= y2:
            return zone_index
    
    # If the point is not in any zone, return -1
    return -1


def filter_for_crop_zones(frame, croppable_detections, zone_of_detections):
    
    center_points = [] #center points of detections
    croppable_detections_filtered = []

    # Extract center points from bounding boxes and store in the center_points list
    for detection in croppable_detections:
        bbox = detection[2]
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        center_points.append((center_x, center_y))

    for center_point in center_points:
        draw_point(frame, center_point)

    #Crop Zone definitions for intersection 1.
    zones = []
    rows, cols = 8, 4

    # Calculate the width and height of each zone based on the frame dimensions
    zone_width = frame.shape[1] // cols
    zone_height = frame.shape[0] // rows

    # Create the zones (rectangles) and store their coordinates
    for i in range(rows):
        for j in range(cols):
            x1 = j * zone_width
            y1 = i * zone_height
            x2 = (j + 1) * zone_width
            y2 = (i + 1) * zone_height
            if(y1 > 325):
                zones.append((x1, y1, x2, y2))

    # Draw the zones on the frame
    for zone in zones:
        draw_zone(frame, zone)

    for detection, center in zip(croppable_detections, center_points):
        zone = zone_of_point(zones, center)
        if(zone != -1):
            if((detection[1] not in zone_of_detections) or (zone_of_detections[detection[1]] != zone)):
                croppable_detections_filtered.append(detection)
                zone_of_detections.update({detection[1] : zone})


    return labeled_frame1, croppable_detections_filtered


video_path_1 = '/home/tomass/tomass/ReID_pipele/source_videos/AI_City_01_Itersection/vdo4.avi'
ground_truths_path_1 = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/gt/gt.txt"
intersection1_folder = os.path.join(sys.path[0], f'cropped/AIC22_Track1_MTMC_train_S01/1/')


video_path_2 = '/home/tomass/tomass/ReID_pipele/source_videos/AI_City_01_Itersection/vdo1.avi'
ground_truths_path_2 = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c001/gt/gt.txt"
intersection2_folder = os.path.join(sys.path[0], f'cropped/AIC22_Track1_MTMC_train_S01/2/')


video1 = cv2.VideoCapture(video_path_1)
file1 = open(ground_truths_path_1, 'r')
lines1 = file1.readlines()

video2 = cv2.VideoCapture(video_path_2)
file2 = open(ground_truths_path_2, 'r')
lines2 = file2.readlines()

curr_line1 = None
last_read1 = 0
frame_id1 = 1

curr_line2 = None
last_read2 = 0
frame_id2 = 1

seen_vehicle_ids = []

zone_of_detections = {}


for frame_nr in range(int(video1.get(cv2.CAP_PROP_FRAME_COUNT))):
    frame_nr +=1
    # -------------------- INTERSECTION 1 -------------------------
    # reading frame from video
    _, frame1 = video1.read()
    frame_id1, last_read1, curr_line1, labeled_frame1, croppable_detections1 = ground_truth_for_frame(frame_id1, last_read1, frame_nr, curr_line1, frame1, lines1)
    #print(croppable_detections1)

    #Ja izmantojam crop zones
    if(saving_mode == 2 or saving_mode == 3):
        labeled_frame1, croppable_detections1 = filter_for_crop_zones(labeled_frame1, croppable_detections1, zone_of_detections)
    
    for detection in croppable_detections1: #croppable detections satur detections zonai, iteree cauri zonaam
        detection_crop.crop_from_bbox(frame1, detection[1], detection[2], 1) # (frame, vehID, bbox, intersectionNr)
        if detection[1] not in seen_vehicle_ids:
            seen_vehicle_ids.append(detection[1])

    if(os.path.exists(intersection1_folder) and (not len(os.listdir(intersection1_folder)) == 0)):
        #fExtract.save_extractions_to_CSV(intersection_folder)
        #fExtract.save_extractions_to_vector_db(intersection_folder, intersection)
        fExtract.save_extractions_to_lance_db(intersection1_folder, 1, saving_mode)
        #fExtract.save_extractions_to_lance_db(intersection1_folder, 1, saving_mode)






    # -------------------- INTERSECTION 2 -------------------------
    # reading frame from video
    _, frame2 = video2.read()
    frame_id2, last_read2, curr_line2, labeled_frame2, croppable_detections2 = ground_truth_for_frame(frame_id2, last_read2, frame_nr, curr_line2, frame2, lines2, seen_vehicle_ids)

    for detection in croppable_detections2: #croppable detections satur detections zonai, iteree cauri zonaam
        detection_crop.crop_from_bbox(frame2, detection[1], detection[2], 2) # (frame, vehID, bbox, intersectionNr)

    if(os.path.exists(intersection2_folder) and (not len(os.listdir(intersection2_folder)) == 0)):
        #fExtract.save_extractions_to_CSV(intersection_folder)
        #fExtract.save_extractions_to_vector_db(intersection_folder, intersection)
        #fExtractCLIP.save_extractions_to_lance_db(intersection_folder, intersection)
        results_map = fExtract.compare_extractions_to_lance_db(intersection2_folder, 1)
        results(results_map)

    


    #refresh 2. intersection detections
    images = glob.glob(intersection2_folder + '/*')
    for i in images:
        os.remove(i)

    #print(seen_vehicle_ids)

    # resized = cv2.resize(labeled_frame1, (1280, 720))
    # cv2.imshow("frame1", resized)

    # resized2 = cv2.resize(labeled_frame2, (1280, 720))
    # cv2.imshow("frame2", resized2)
    # cv2.waitKey(0)

video1.release()
file1.close()

video2.release()
file2.close()