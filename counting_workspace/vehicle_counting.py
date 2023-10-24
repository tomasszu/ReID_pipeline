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

import misc.fisheye_vid_to_pano as toPano

import misc.crop as detection_crop

import misc.counting_package.counting_and_crop_list as counting

from tqdm import tqdm

import os
HOME = os.getcwd()
print(HOME)

from dataclasses import dataclass

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

# settings
MODEL = "yolov8x.pt"

model = YOLO(MODEL)

model.fuse()

confidence_threshold: float = 0.5,
iou_threshold: float = 0.7,

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [2, 3, 5, 7]

video_path = '/home/tomass/tomass/ReID_pipele/source_videos/fisheye_vid_1.mp4'
TARGET_VIDEO_PATH = f"{HOME}/fisheye-counting-result.mp4"

video = cv2.VideoCapture(video_path)

intersection = "intersection_1"

save_dir = f'../cropped/{intersection}/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
os.chdir(save_dir)

# tracker = sv.ByteTrack(track_thresh = 0.25, track_buffer = 30, match_thresh = 0.8, frame_rate = 4 )#BYTETrackerArgs())
tracker = sv.ByteTrack(track_thresh = 0.40, track_buffer = 30, match_thresh = 0.7, frame_rate = 2 )#BYTETrackerArgs())
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

#line_counter_1 = sv.LineZone(start=LINE_START_1, end=LINE_END_1)
line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)

ZONE1 = counting.countZone(1244, 256, 324, -150)
ZONE2 = counting.countZone(582, 191, 150, -60)
ZONE3 = counting.countZone(205, 164, 130, -60)
ZONE4 = counting.countZone(2, 284, 90, -130)
zone_annotator = counting.ZoneAnnotator(thickness=2, text_thickness=2, text_scale=1)


for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
    # reading frame from video
    _, frame = video.read()

    frame = toPano.fisheye_video_to_pano(frame)

    croppable_detections = []

    #results = model()
    results = model(frame)[0]

    detections = sv.Detections.from_ultralytics(results)

    mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
    detections = detections[np.isin(detections.class_id, CLASS_ID)]
    detections = detections[np.greater(detections.confidence, confidence_threshold)]

    detections = tracker.update_with_detections(detections)

    # format custom labels
    labels = [
        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id 
        in detections
    ]


    annotated_frame = box_annotator.annotate(
        scene=frame.copy(), detections=detections
    )

    croppable_detections.append(ZONE1.trigger(detections=detections))
    croppable_detections.append(ZONE2.trigger(detections=detections))
    croppable_detections.append(ZONE3.trigger(detections=detections))
    croppable_detections.append(ZONE4.trigger(detections=detections))

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


    annotated_labeled_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels
    )


    annotated_labeled_frame = trace_annotator.annotate(
         scene=annotated_labeled_frame,
         detections=detections
    )
    for detection in croppable_detections:
        if(detection):
            #print(detection[])
            detection_crop.crop_from_bbox(frame, detection[0][0], detection[0][1])
    
    
    cv2.imshow("frame", annotated_labeled_frame)
    cv2.waitKey(0)