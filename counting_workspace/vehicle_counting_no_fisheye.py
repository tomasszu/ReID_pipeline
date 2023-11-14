# Object Detecion 
import glob
import cv2
import supervision as sv
from ultralytics import YOLO

#plots
import matplotlib.pyplot as plt

#basics
import pandas as pd
import numpy as np
import os
import sys
import re

#Vehivle ReID model
sys.path.append("vehicle_reid_repo/")
sys.path.append("..")
from vehicle_reid.load_model import load_model_from_opts
import torch

import misc.fisheye_vid_to_pano as toPano

import misc.crop as detection_crop

import misc.counting_package.counting_and_crop_list as counting

import misc.feature_extract as fExtract

import misc.database as db

from tqdm import tqdm

import os

def detections_process(model, frame, tracker):
    confidence_threshold = 0.7

    #results = model()
    results = model(frame)[0]

    detections = sv.Detections.from_ultralytics(results)

    #mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
    detections = detections[np.isin(detections.class_id, CLASS_ID)]
    detections = detections[np.greater(detections.confidence, confidence_threshold)]

    detections = tracker.update_with_detections(detections)

    return detections

def frame_annotations(detections, frame):

    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    # format custom labels
    labels = [
        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id 
        in detections
    ]


    annotated_frame = box_annotator.annotate(
        scene=frame.copy(), detections=detections
    )


    annotated_labeled_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )


    annotated_labeled_frame = trace_annotator.annotate(
        scene=annotated_labeled_frame,
        detections=detections
    )

    return annotated_labeled_frame


# settings
MODEL = "yolov8x.pt"

model = YOLO(MODEL)

model.fuse()

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [2, 3, 5, 7]

zone_annotator = counting.ZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)

#------------ INTERSECTION 1 ------------------------------------------------------------

video_path = '/home/tomass/tomass/ReID_pipele/source_videos/Sequence1a/Intersection_1.mp4'

video = cv2.VideoCapture(video_path)

intersection = "intersection_1"
intersection_folder = os.path.join(sys.path[0], f'../cropped/Sequence1a/{intersection}/')

if not os.path.exists(intersection_folder):
    os.makedirs(intersection_folder)

# tracker = sv.ByteTrack(track_thresh = 0.25, track_buffer = 30, match_thresh = 0.8, frame_rate = 4 )#BYTETrackerArgs())
tracker = sv.ByteTrack(track_thresh = 0.40, track_buffer = 30, match_thresh = 0.7, frame_rate = 20 )#BYTETrackerArgs())


ZONE1 = counting.countZone(362, 127, 122, -70)
ZONE2 = counting.countZone(1, 164, 155, -70)
ZONE3 = counting.countZone(0, 493, 242, -140)
ZONE4 = counting.countZone(557, 328, 655, -149)

#------------ INTERSECTION 2 ------------------------------------------------------------

video_path2 = '/home/tomass/tomass/ReID_pipele/source_videos/Sequence1a/Intersection_2.mp4'

video2 = cv2.VideoCapture(video_path2)

intersection2 = "intersection_2"
intersection_folder2 = os.path.join(sys.path[0], f'../cropped/Sequence1a/{intersection2}/')

if not os.path.exists(intersection_folder2):
    os.makedirs(intersection_folder2)

# tracker = sv.ByteTrack(track_thresh = 0.25, track_buffer = 30, match_thresh = 0.8, frame_rate = 4 )#BYTETrackerArgs())
tracker2 = sv.ByteTrack(track_thresh = 0.40, track_buffer = 30, match_thresh = 0.7, frame_rate = 20 )#BYTETrackerArgs())


for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):

    # -------------------- INTERSECTION 1 -------------------------

    # reading frame from video
    _, frame = video.read()


    detections = detections_process(model, frame, tracker)

    croppable_detections = []
    croppable_detections.append(ZONE1.trigger(detections=detections))
    croppable_detections.append(ZONE2.trigger(detections=detections))
    croppable_detections.append(ZONE3.trigger(detections=detections))
    croppable_detections.append(ZONE4.trigger(detections=detections))

    annotated_frame = frame_annotations(detections, frame)

    annotated_frame = zone_annotator.annotate(
        frame=annotated_frame, zone_counter=ZONE1
    )
    annotated_frame = zone_annotator.annotate(
        frame=annotated_frame, zone_counter=ZONE2
    )
    annotated_frame = zone_annotator.annotate(
        frame=annotated_frame, zone_counter=ZONE3
    )
    annotated_frame = zone_annotator.annotate(
        frame=annotated_frame, zone_counter=ZONE4
    )
    #print(croppable_detections)
    for zone_detections in croppable_detections: #croppable detections satur detections zonai, iteree cauri zonaam
        if(zone_detections): # ja zonaa ir detection
            #print(detection[])
            for detection in zone_detections:
                detection_crop.crop_from_bbox(frame, detection[0], detection[1], intersection) # (frame, vehID, bbox, intersectionNr)
    

    if(not len(os.listdir(intersection_folder)) == 0):
        #fExtract.save_extractions_to_CSV(intersection_folder)
        fExtract.save_extractions_to_vector_db(intersection_folder, intersection)

    cv2.imshow("frame", annotated_frame)

    # -------------------- INTERSECTION 2 -------------------------

    _, frame2 = video2.read()

    detections2 = detections_process(model, frame2, tracker2)
    #print(detections2.xyxy)
    #print(detections2.tracker_id)


    if(len(detections2.xyxy) != 0):
        for bbox, id in zip(detections2.xyxy, detections2.tracker_id):
            detection_crop.crop_from_bbox(frame2, id, bbox, intersection2)
    

    # EXTRACTIONS TEST FUNKCIJA ----------------------------------------
    if(not len(os.listdir(intersection_folder2)) == 0):
    #     #fExtract.save_extractions_to_CSV(intersection_folder)
    #     #fExtract.save_extractions_to_vector_db(intersection2)
        from PIL import Image
        device = "cuda"

        reIdModel = load_model_from_opts("/home/tomass/tomass/ReID_pipele/vehicle_reid_repo/vehicle_reid/model/result/opts.yaml", ckpt="/home/tomass/tomass/ReID_pipele/vehicle_reid_repo/vehicle_reid/model/result/net_19.pth", remove_classifier=True)
        reIdModel.eval()
        reIdModel.to(device)

        extractables_folder = intersection_folder2
        extractable_images = os.listdir(extractables_folder)

        images = [Image.open(extractables_folder + x) for x in extractable_images]
        X_images = torch.stack(tuple(map(fExtract.data_transforms, images))).to(device)

        features = [fExtract.extract_feature(reIdModel, X) for X in X_images]
        features = torch.stack(features).detach().cpu()

        features_array = np.array(features)
        compare_array = []
        for image_name, embedding in zip(extractable_images, features_array):
            image_id = re.sub(r'[^0-9]', '', image_name)
            compare_array.append([image_id, embedding])
            #print(f"{image_id}: {embedding} \n")
        print("From intersection 2. -> 1. :")
        track_map = {}
        for vehicle in compare_array:
            #print(db.query(vehicle[1],intersection))
            result = db.query_for_ID(vehicle[1],intersection)
            if(result != -1):
                track_map[vehicle[0]] = result[0].vehicle_id
                print(f"{vehicle[0]} found as -> {result[0].vehicle_id}")

        print(track_map)
        #convert 2. frame track id's to 1.st frame detected tracks
        if(len(track_map) != 0):
            for i, track in enumerate(detections2.tracker_id):
                detections2.tracker_id[i] = track_map[str(track)]
        else:
            for i, track in enumerate(detections2.tracker_id):
                detections2.tracker_id[i] = i * (-1)

        # ŠITO VISU VAJAG PATESTEET TAD !!! --------------------------------->

        #refresh 2. intersection detections
        images = glob.glob(extractables_folder + '/*')
        for i in images:
            os.remove(i)
        
    # -------------------------------------------------------------------

    annotated_frame2 = frame_annotations(detections2, frame2)
    cv2.imshow("frame2", annotated_frame2)




#     db.query(embedding)
    cv2.waitKey(0)