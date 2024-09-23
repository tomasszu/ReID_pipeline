import supervision as sv
from ultralytics import YOLO

import cv2

#basics
import pandas as pd
import numpy as np
import os

model = YOLO('yolov8s.pt')

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [2, 3, 5, 7]

result = model("/home/tomass/tomass/ReID_pipele/source_images/panoramas/panorama_01_fisheye_day_000000.jpg")[0]
detections = sv.Detections.from_ultralytics(result)

frame = cv2.imread("/home/tomass/tomass/ReID_pipele/source_images/panoramas/panorama_01_fisheye_day_000000.jpg")


# format custom labels
labels = [
    f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
    for _, _, confidence, class_id, tracker_id
    in detections
]

box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(
    scene=frame,
    detections=detections
)

# # annotate and display frame
# frame = sv.BoxAnnotator.annotate(frame=frame, detections=detections, labels=labels)
# #print(detections.xyxy, detections.confidence, detections.class_id)


cv2.imshow("frame", annotated_frame)
cv2.waitKey(0)