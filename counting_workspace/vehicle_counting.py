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
intersection_folder = os.path.join(sys.path[0], f'../cropped/{intersection}/')

if not os.path.exists(intersection_folder):
    os.makedirs(intersection_folder)

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
    

    if(not len(os.listdir(intersection_folder)) == 0):
        #fExtract.save_extractions_to_CSV(intersection_folder)
        fExtract.save_extractions_to_vector_db(intersection_folder)

    embedding = np.array([ 5.23397923e-02, 6.03124350e-02, 3.25482823e-02, -5.58898179e-03
, -3.16619454e-03, 2.61551924e-02, 1.14738885e-02, 4.04893272e-02
, -3.56019475e-02, -3.73675264e-02, -2.56420724e-04, -4.47622761e-02
, -7.99676403e-02, 3.17001343e-02, 9.74051058e-02, 2.96873264e-02
, -7.59678520e-03, 2.07117312e-02, 2.06376184e-02, 3.27378251e-02
, 9.36533734e-02, 2.83522159e-02, -4.96786796e-02, 2.73423456e-02
, 2.29427009e-03, 6.00968711e-02, 1.48897870e-02, -5.24051348e-03
, 1.43639240e-02, 3.16389017e-02, -1.84501987e-02, -2.67454702e-02
, -1.87312197e-02, 1.63553588e-04, 4.05064933e-02, -4.15326245e-02
, -3.32346931e-02, 1.81193575e-02, -5.26402937e-03, -1.49473231e-02
, -1.02765664e-01, 3.45297270e-02, -5.08931058e-04, 8.22347105e-02
, -6.34537935e-02, 3.46768908e-02, -2.90241819e-02, -3.03269606e-02
, 1.14481999e-02, 3.51094306e-02, 1.53724058e-02, -4.98009613e-03
, 8.98971129e-03, 2.62154192e-02, -1.24695161e-02, -2.73312796e-02
, -4.97592203e-02, -4.86832391e-03, -7.77373090e-02, -2.49203034e-02
, 5.79760000e-02, -7.36415535e-02, -2.51181051e-02, -4.47955653e-02
, -6.28358126e-03, -3.36908996e-02, 2.28409730e-02, 3.33951786e-02
, -1.84340589e-02, -5.27780782e-03, 3.76922563e-02, -1.30618913e-02
, -2.70509049e-02, -9.45652369e-03, 1.16618094e-03, -2.11310647e-02
, 3.16098519e-02, 8.67504999e-02, 3.50778401e-02, -3.00518069e-02
, 5.33289462e-02, 8.20333138e-02, -2.60640569e-02, -1.82081927e-02
, 2.79850122e-02, -4.95479666e-02, -1.56610776e-02, -1.89843238e-03
, -5.29926084e-02, 5.53026469e-03, -1.98586192e-02, -2.92165540e-02
, -8.21288750e-02, -1.13198152e-02, -5.48755489e-02, -9.52467881e-03
, 7.36127868e-02, 7.57377073e-02, 2.53249966e-02, -1.33601516e-01
, 7.12664872e-02, 2.61400128e-03, 2.08348338e-03, -2.06381008e-02
, 3.65312956e-02, -8.31078216e-02, -6.12504221e-02, 6.42409399e-02
, -8.88608620e-02, -1.60824619e-02, 3.51093058e-03, 4.71761450e-02
, -2.43071243e-02, -6.01305533e-03, 4.80648652e-02, 3.29258367e-02
, -4.81134057e-02, -3.71802337e-02, 1.03647858e-01, 6.34987578e-02
, 8.60196501e-02, -3.67617644e-02, -6.05894513e-02, 3.39566567e-03
, 1.29186437e-02, 6.93701878e-02, -1.90261919e-02, -7.38546252e-02
, 1.54030398e-02, -1.69554129e-02, 1.16883349e-02, 2.75010318e-02
, 1.10223768e-02, 4.05848809e-02, 3.73883024e-02, -3.67355789e-03
, -1.99415274e-02, 1.12643398e-01, 5.62691726e-02, 3.43566239e-02
, -3.99152525e-02, -2.91418415e-02, -8.54269043e-03, -4.35872339e-02
, -1.12578180e-02, 2.03155186e-02, 1.87028944e-02, -1.29472939e-02
, -7.28632044e-03, -6.64256066e-02, 1.52227040e-02, -1.92214623e-02
, -5.91503046e-02, 1.01473918e-02, 9.35487896e-02, -3.18821073e-02
, 3.87250483e-02, 2.09556594e-02, 2.20752116e-02, 5.29465713e-02
, -3.72251682e-02, -2.80424859e-02, 4.74353060e-02, 1.18548833e-01
, -4.59261015e-02, -4.14065868e-02, -1.17011908e-02, 1.80921666e-02
, 9.06200856e-02, -4.21602242e-02, 5.12907282e-02, 3.18101645e-02
, 1.75172705e-02, 1.43246865e-02, 4.66242842e-02, -5.95022440e-02
, -5.53128757e-02, -5.25135882e-02, -5.90760335e-02, -2.74347924e-02
, 3.85314338e-02, -4.93511604e-03, 4.94286651e-04, -1.38069876e-02
, 5.72097860e-02, -1.74386241e-02, -9.45243388e-02, 7.41735026e-02
, 1.39837684e-02, 9.55234170e-02, 1.58923995e-02, -3.12901055e-03
, 1.48156229e-02, -5.29695209e-03, 2.46847086e-02, -1.21766329e-02
, -1.61769204e-02, 2.87927277e-02, 7.65750483e-02, -4.38482687e-02
, -4.68451977e-02, 1.10170646e-02, 1.58169139e-02, -5.52345552e-02
, -6.24407269e-03, -5.14715649e-02, -3.18749063e-02, 2.37536002e-02
, -6.10630065e-02, 6.24355935e-02, -9.71984956e-03, -5.11387102e-02
, -6.50513405e-03, -8.83717462e-02, 4.44263071e-02, -5.05973003e-04
, 2.34758835e-02, -3.46365795e-02, 1.24120964e-02, -7.38238022e-02
, -2.17709318e-02, 2.26472709e-02, -3.01727559e-02, -1.21678151e-01
, -3.59846861e-04, -1.52190803e-02, -6.37329444e-02, -1.14307351e-01
, 2.64561214e-02, -2.68157255e-02, 4.52375971e-02, 4.58565876e-02
, -1.41201625e-02, 1.95541140e-02, 1.57055426e-02, -6.82587773e-02
, 4.09084978e-03, -2.43593976e-02, 2.00622212e-02, -6.95421472e-02
, -4.96041439e-02, -2.59301607e-02, 2.10224688e-02, 1.80572681e-02
, 7.17378175e-03, 3.88453901e-03, 1.95548385e-02, -1.01285707e-03
, -2.09039524e-02, -2.42226310e-02, -3.21090259e-02, 3.57332565e-02
, 1.11230277e-02, 9.07008070e-03, 3.98348495e-02, 6.17483109e-02
, -4.18918729e-02, -5.89665882e-02, 1.63226556e-02, 1.92377642e-02
, 2.03715581e-02, -1.01778889e-02, 8.75562243e-03, 5.60152382e-02
, 1.19549870e-01, -3.71623114e-02, -1.60924532e-02, -5.08439168e-03
, 5.83569519e-02, -3.93855356e-04, 8.81984923e-03, -7.11737201e-02
, -2.75807735e-02, -4.64844741e-02, 1.24567449e-02, -8.43477696e-02
, 5.53994104e-02, 9.48827993e-03, -2.90654395e-02, -1.60923861e-02
, 9.95023027e-02, 5.67610785e-02, -2.58897524e-02, 3.22502740e-02
, -3.50949392e-02, -5.40912338e-02, -4.83774617e-02, -3.63904461e-02
, -2.14614626e-02, 6.22867309e-02, 2.89276876e-02, -3.04686166e-02
, -5.01246229e-02, -7.31403828e-02, -6.90370128e-02, 4.82145995e-02
, 1.80608127e-03, 7.37320781e-02, -3.97550985e-02, 3.01923305e-02
, -2.93995142e-02, 7.17635639e-03, 1.37752473e-01, 4.95246835e-02
, -6.25236556e-02, -3.27766815e-04, 1.52377440e-02, 3.47450189e-03
, 7.68085010e-03, -7.35008856e-03, 3.32886465e-02, 3.65670361e-02
, 1.03034908e-02, 5.49350828e-02, 7.37021416e-02, -5.23116663e-02
, 3.48454225e-03, -3.42503712e-02, -5.48350178e-02, 2.11103610e-03
, 2.42606062e-03, -5.69424219e-02, -7.96771888e-03, -2.50941161e-02
, 3.24984603e-02, 3.27391066e-02, -2.85368077e-02, 1.23898769e-02
, 2.86448784e-02, 6.00809455e-02, -2.89183967e-02, -6.76065236e-02
, 4.44912240e-02, -4.40795487e-03, 1.26093645e-02, -5.93200587e-02
, -2.60740463e-02, 8.87213834e-03, -4.81629325e-03, 4.07681707e-03
, 3.23710851e-02, -8.02573469e-03, -6.00213856e-02, -1.05629899e-02
, 6.15890138e-03, 8.43426958e-02, -6.54539391e-02, 2.30924110e-03
, 7.86484964e-03, -1.36198496e-05, 2.55830288e-02, 7.03513846e-02
, 6.09770301e-04, -1.52386734e-02, -1.11290831e-02, -5.93713997e-03
, 3.71410162e-03, -4.56072874e-02, 1.20410081e-02, 3.89726087e-02
, 1.38120316e-02, 2.63563101e-03, -2.50903144e-02, 5.95729984e-02
, 6.85466453e-02, 1.72917899e-02, 1.20781520e-02, 6.15742207e-02
, 3.84033062e-02, 8.64709765e-02, 2.88419295e-02, 5.71667850e-02
, 1.22700952e-01, -3.83676365e-02, -1.06226113e-02, -5.42576835e-02
, 3.21412086e-02, -7.57364854e-02, -2.91859061e-02, -3.88474874e-02
, 3.25831249e-02, 4.34765741e-02, -1.08339816e-01, 1.61349680e-02
, -3.99116576e-02, -2.79334579e-02, 5.92592582e-02, 3.43397595e-02
, 2.36124522e-03, -1.60213802e-02, 4.02836502e-02, 6.20944649e-02
, -3.03374212e-02, 5.45657314e-02, -3.07366885e-02, -3.67422588e-02
, 2.94173099e-02, 9.15941375e-04, 4.21903171e-02, -2.79659480e-02
, 3.22813541e-02, 7.61713982e-02, 1.77347776e-03, 9.42658558e-02
, 3.93364169e-02, 1.15945134e-02, -1.30844330e-02, -2.24842876e-02
, -1.97915733e-03, 5.82406719e-05, -5.00794016e-02, -6.54418021e-02
, -4.21686620e-02, 6.07803911e-02, 7.18065426e-02, 3.98695730e-02
, -1.01568997e-01, 6.66929409e-02, 1.20927393e-02, -4.70637195e-02
, -8.92301649e-03, -2.64564082e-02, -2.64551546e-02, -5.15790796e-03
, -2.62586810e-02, -3.44835222e-02, 1.98681466e-02, 7.21910922e-03
, -6.60277903e-02, 3.92290112e-03, -4.49204110e-02, -6.67458121e-03
, 7.43905222e-03, -1.40978638e-02, -8.68223011e-02, 3.96782197e-02
, 2.77070235e-02, 1.58085041e-02, 8.65534917e-02, 8.74950737e-03
, 9.06923339e-02, 7.37696094e-03, -4.79923896e-02, -2.59363428e-02
, 1.33881466e-02, 5.32274097e-02, -5.60533851e-02, 1.79481059e-02
, -4.72821780e-02, 2.01433199e-03, -4.56737587e-03, -5.38394339e-02
, -1.39117017e-02, -2.08739229e-02, -6.20598532e-02, 7.09064305e-02
, -1.38203967e-02, 1.25455679e-02, 1.20391259e-02, -2.06372142e-02
, -2.64690742e-02, 3.96311618e-02, 4.89323102e-02, 4.82703261e-02
, -5.01656299e-03, -6.51119575e-02, 1.22947758e-03, -4.25326973e-02
, -1.85614154e-02, -6.39273599e-02, 5.22675663e-02, 5.63537003e-03
, 2.70223692e-02, -9.98988748e-02, 4.68365801e-03, -5.30666672e-02
, 1.89510116e-03, -4.46808785e-02, 6.21310174e-02, -3.55625525e-02
, -2.57955473e-02, 6.60646881e-04, -2.05264874e-02, -9.55965221e-02
, 9.57948528e-03, -2.13143621e-02, -5.99243352e-03, 2.12000161e-02
, 2.53390446e-02, -1.40606826e-02, 2.49212086e-02, 3.76769528e-02
, -2.42918581e-02, -8.99358019e-02, 2.43428648e-02, 7.65778273e-02
, 2.69414708e-02, 1.75295286e-02, 5.62985279e-02, 1.88213903e-02
, -4.02790820e-03, 1.21517209e-02, 2.59349048e-02, -3.30337174e-02
, 9.45208296e-02, -8.94706231e-03, 3.13097872e-02, 9.06526111e-03
, -1.41582051e-02, 4.84469533e-02, -5.31190522e-02, -2.17318144e-02])


    db.query(embedding)
    
    cv2.imshow("frame", annotated_labeled_frame)
    cv2.waitKey(0)