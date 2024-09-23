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
sys.path.append("../../Documents")

from cam_credentials import CAMERA_ADDRESS1
from cam_credentials import CAMERA_ADDRESS2

#Vehivle ReID model
sys.path.append("vehicle_reid_repo/")
sys.path.append("..")
from vehicle_reid.load_model import load_model_from_opts
import torch
import clip

import misc.crop as detection_crop

#import misc.counting_package.counting_and_crop_list as counting
import misc.counting_package.counting_and_crop_list_v2 as counting

import supervision.detection.line_counter as counter
# import supervision.detection.line_counter_copy as counter
from supervision.geometry.core import Point

import misc.feature_extract as fExtract
import misc.feature_extract_CLIP as fExtractCLIP

import misc.lance_db_CLIP as l_db

from scipy.special import softmax


def detections_process(model, frame, tracker):
    confidence_threshold = 0.6

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
line_zone_annotator = counter.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=1, color=sv.Color.red())

#------------ INTERSECTION 1 ------------------------------------------------------------

video_path = '/home/tomass/tomass/ReID_pipele/source_videos/NetCam/cam2_cuts.avi'

video = cv2.VideoCapture(video_path)

video.set(3, 1920)
video.set(4, 1080)


# tracker = sv.ByteTrack(track_thresh = 0.25, track_buffer = 30, match_thresh = 0.8, frame_rate = 4 )#BYTETrackerArgs())
tracker = sv.ByteTrack(track_thresh = 0.01, track_buffer = 40, match_thresh = 0.99, frame_rate = 2 )#BYTETrackerArgs())

#PAGASREIZ BEIDZJU KKUR TE


#AICITY(4)--------------------------------------------
ZONE1 = counting.countZone(294, 475, 698, -229)

#CAMERA1
# point1 = Point(150, 400)
# point2 = Point(850, 500)
# line1 = counter.LineZone(point1, point2)
#CAMERA2
point1 = Point(500, 1600)
point2 = Point(1800, 700)
line1 = counter.LineZone(point1, point2)
#_, frame = video.read()
for i in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
    i = i +1
    if i%1 == 0:
        i = 0
        # reading frame from video
        #last_frame = frame
        _, frame = video.read()
        
        detections = detections_process(model, frame, tracker)

            # croppable_detections = []
            # # croppable_detections.append(ZONE1.trigger(detections=detections))
            # # croppable_detections.append(ZONE2.trigger(detections=detections))
            # # croppable_detections.append(ZONE3.trigger(detections=detections))
            # # croppable_detections.append(ZONE4.trigger(detections=detections))

        annotated_frame = frame_annotations(detections, frame)

        line1.trigger(detections=detections)

        # annotated_frame = zone_annotator.annotate(
        #     frame=annotated_frame, zone_counter=ZONE1
        # )

        annotated_frame = line_zone_annotator.annotate(frame=annotated_frame, line_counter=line1)
            # # annotated_frame = zone_annotator.annotate(
            # #     frame=annotated_frame, zone_counter=ZONE2
            # # )
            # # annotated_frame = zone_annotator.annotate(
            # #     frame=annotated_frame, zone_counter=ZONE3
            # # )
            # # annotated_frame = zone_annotator.annotate(
            # #     frame=annotated_frame, zone_counter=ZONE4
            # # )

            # #--------AICITY(4) Zones -----------------------------------------
            # croppable_detections.append(ZONE1.trigger(detections=detections))

            # annotated_frame = frame_annotations(detections, frame)

            # annotated_frame = zone_annotator.annotate(
            #     frame=annotated_frame, zone_counter=ZONE1
            # )
            # #------------------------------------------------------------------
            # #print(croppable_detections)
            # for zone_detections in croppable_detections: #croppable detections satur detections zonai, iteree cauri zonaam
            #     if(zone_detections): # ja zonaa ir detection
            #         #print(detection[])
            #         for detection in zone_detections:
            #             detection_crop.crop_from_bbox(frame, detection[0], detection[1], intersection) # (frame, vehID, bbox, intersectionNr)
            

            # if(not len(os.listdir(intersection_folder)) == 0):
            #     #fExtract.save_extractions_to_CSV(intersection_folder)
            #     #fExtract.save_extractions_to_vector_db(intersection_folder, intersection)
            #     fExtractCLIP.save_extractions_to_lance_db(intersection_folder, intersection)
            #     #fExtract.save_extractions_to_lance_db(intersection_folder, intersection)
            
        #annotated_frame2 = frame_annotations(detections2, frame2)
        resized =  cv2.resize(annotated_frame, (1440, 960))
        cv2.imshow(f"{CAMERA_ADDRESS1}", resized)


        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        cv2.waitKey(0)

video.release()
cv2.destroyAllWindows()