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

def ground_truth_for_frame(frame_id, last_read, frame_nr, curr_line, un_labeled_frame, lines):
    croppable_detections = [] #kur saglabāt izgriežamos bbokšus
    frame = un_labeled_frame.copy()

    if(last_read != 0 and frame_id == frame_nr):
        #print("output:",frame_nr, curr_line)
        draw_bbox(frame, [int(curr_line[2]),int(curr_line[3]),int(curr_line[4]),int(curr_line[5])], int(curr_line[1]))
        croppable_detections.append([frame_nr, int(curr_line[1]), xywh_to_xyxy([int(curr_line[2]),int(curr_line[3]),int(curr_line[4]),int(curr_line[5])]), 1])
    if(last_read == 0 or frame_id == frame_nr):
        while frame_id == frame_nr:
            line = lines[last_read]
            curr_line = line.split(",", maxsplit=6)
            frame_id = int(curr_line[0])
            vehicle_id = int(curr_line[1])
            xywh = [int(curr_line[2]),int(curr_line[3]),int(curr_line[4]),int(curr_line[5])]
            if(frame_id == frame_nr):
                #print("output:",frame_nr, curr_line)
                draw_bbox(frame, xywh, vehicle_id)
                croppable_detections.append([frame_nr, vehicle_id, xywh_to_xyxy(xywh), 1 ])
                last_read = last_read+1
            else:
                last_read = last_read+1
                break
    return frame_id, last_read, curr_line, frame, croppable_detections

video_path_1 = '/home/tomass/tomass/ReID_pipele/source_videos/AI_City_01_Itersection/vdo4.avi'
ground_truths_path_1 = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/gt/gt.txt"

video_path_2 = '/home/tomass/tomass/ReID_pipele/source_videos/AI_City_01_Itersection/vdo1.avi'
ground_truths_path_2 = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c001/gt/gt.txt"

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

for frame_nr in range(int(video1.get(cv2.CAP_PROP_FRAME_COUNT))):
    frame_nr +=1
    # -------------------- INTERSECTION 1 -------------------------
    # reading frame from video
    _, frame1 = video1.read()
    frame_id1, last_read1, curr_line1, labeled_frame1, croppable_detections1 = ground_truth_for_frame(frame_id1, last_read1, frame_nr, curr_line1, frame1, lines1)
    print(croppable_detections1)

    #------------------------------------------------------------------
    #print(croppable_detections)
    for detection in croppable_detections1: #croppable detections satur detections zonai, iteree cauri zonaam
        detection_crop.crop_from_bbox(frame1, detection[1], detection[2], detection[3]) # (frame, vehID, bbox, intersectionNr)




    # -------------------- INTERSECTION 2 -------------------------
    # reading frame from video
    _, frame2 = video2.read()
    frame_id2, last_read2, curr_line2, labeled_frame2, croppable_detections2 = ground_truth_for_frame(frame_id2, last_read2, frame_nr, curr_line2, frame2, lines2)
 
 
 
    # if(last_read1 != 0):
    #     print("output:",frame_nr, curr_line1)
    #     draw_bbox(frame1, [int(curr_line1[2]),int(curr_line1[3]),int(curr_line1[4]),int(curr_line1[5])], int(curr_line1[1]))
    # while frame_id1 == frame_nr:
    #     line = lines1[last_read1]
    #     curr_line1 = line.split(",", maxsplit=6)
    #     frame_id1 = int(curr_line1[0])
    #     vehicle_id = int(curr_line1[1])
    #     xywh = [int(curr_line1[2]),int(curr_line1[3]),int(curr_line1[4]),int(curr_line1[5])]
    #     if(frame_id1 == frame_nr):
    #         print("output:",frame_nr, curr_line1)
    #         draw_bbox(frame1, xywh, vehicle_id)
    #         last_read1 = last_read1+1
    #     else:
    #         last_read1 = last_read1+1
    #         break




    resized = cv2.resize(labeled_frame1, (1280, 720))
    cv2.imshow("frame1", resized)

    resized2 = cv2.resize(labeled_frame2, (1280, 720))
    cv2.imshow("frame2", resized2)
    cv2.waitKey(0)

video1.release()
file1.close()

video2.release()
file2.close()