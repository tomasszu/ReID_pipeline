# Object Detecion 
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
    
    for detection in croppable_detections:
        if(detection):
            #print(detection[])
            detection_crop.crop_from_bbox(frame, detection[0][0], detection[0][1], intersection)
    

    if(not len(os.listdir(intersection_folder)) == 0):
        #fExtract.save_extractions_to_CSV(intersection_folder)
        fExtract.save_extractions_to_vector_db(intersection_folder)

    cv2.imshow("frame", annotated_frame)

    # -------------------- INTERSECTION 2 -------------------------

    _, frame2 = video2.read()

    detections2 = detections_process(model, frame2, tracker2)

    if(len(detections2) is not 0):

        annotated_frame2 = frame_annotations(detections2, frame2)

        # <--------------------- ERROR, jasaprot ar ko detections2 / detection atskiras no croppable detections
        for detection in detections2:
            if(detection):
                #print(detection[])
                detection_crop.crop_from_bbox(frame2, detection[0][0], detection[0][1], intersection2)

    cv2.imshow("frame2", annotated_frame2)




#     db.query(embedding)
    cv2.waitKey(0)