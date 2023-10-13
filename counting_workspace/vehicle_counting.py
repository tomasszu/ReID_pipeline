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

confidence_threshold: float = 0.3,
iou_threshold: float = 0.7,

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [2, 3, 5, 7]

 #line settings
LINE_START = sv.Point(1250, 290)
LINE_END = sv.Point(1600, 240)

# create instance of BoxAnnotator
# box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=1)

video_path = '/home/tomass/tomass/ReID_pipele/source_videos/fisheye_vid_1.mp4'
TARGET_VIDEO_PATH = f"{HOME}/fisheye-counting-result.mp4"

video = cv2.VideoCapture(video_path)

tracker = sv.ByteTrack()#BYTETrackerArgs())
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

line_counter = sv.LineZone(start=LINE_START, end=LINE_END)
line_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2)

for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
    # reading frame from video
    _, frame = video.read()

    frame = toPano.fisheye_video_to_pano(frame)

    #results = model()
    results = model(frame)[0]

    detections = sv.Detections.from_ultralytics(results)

    mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
    detections = detections[np.isin(detections.class_id, CLASS_ID)]

    detections = tracker.update_with_detections(detections)

    #for detection in detections: print(detection)

    # filtering out detections without trackers
    #detections = [detection for detection in detections if detection[4] is not None]


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


    # updating line counter
    line_counter.trigger(detections=detections)

    annotated_labeled_frame = line_annotator.annotate(frame=annotated_labeled_frame, line_counter=line_counter)

    print(line_counter.in_count)

    # box_annotator = sv.BoxAnnotator()
    # annotated_frame = box_annotator.annotate(
    #     scene=frame,
    #     detections=detections
    # )

    # # annotate and display frame
    # frame = sv.BoxAnnotator.annotate(frame=frame, detections=detections, labels=labels)
    # #print(detections.xyxy, detections.confidence, detections.class_id)


    cv2.imshow("frame", annotated_labeled_frame)
    cv2.waitKey(0)