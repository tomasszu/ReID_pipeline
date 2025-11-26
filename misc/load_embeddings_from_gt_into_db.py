import cv2
import os
import numpy as np
import sys

sys.path.append('/home/tomass/tomass/ReID_pipele/pipeline_precision_testing')

from FeatureExtract import FeatureExtractor
from opensearchDatabaseOperations import Database

class EmbeddingsFromGT:
    def __init__(self, vdo_path, gt_path, cam_id, split):

        self.extractor = FeatureExtractor()

        self.cam_id = cam_id
        self.split = split

        #load video file
        self.vdo = cv2.VideoCapture(vdo_path)
        if not self.vdo.isOpened():
            raise IOError(f"Could not open video: {vdo_path}")
        
        #load gt
        gt_lines = None
        with open(gt_path, 'r') as f:
            gt_lines = [line.strip() for line in f.readlines() if line.strip()]
        if gt_lines is None:
            raise Exception(f"gt file {gt_path} could not be read")
        
        self.turn_gt_into_dict(gt_lines)

        self.add_embedding_to_dict()


    def turn_gt_into_dict(self, gt_lines):

        # Parse gt into a dict
        # {frame_id: [(veh_id, (x1, y1, x2, y2))]}

        self.gt_dict = {}

        for line in gt_lines:
            parts = line.split(',')
            frame_id = int(parts[0])
            veh_id = int(parts[1])
            x, y, w, h = map(int, parts[2:6])
            x1, y1, x2, y2 = x, y, x + w, y + h
            if frame_id not in self.gt_dict:
                self.gt_dict[frame_id] = []
            self.gt_dict[frame_id].append((veh_id, (x1, y1, x2, y2)))
        
    def get_crops(self,frame, detections):
        crops = []

        for veh_id, (x1, y1, x2, y2) in detections:
            crop = frame[y1:y2, x1:x2]
            if crop.size != 0:
                crops.append(crop)
            else:
                print("[CROP] Warning, rejected crop! Crop size = 0.")

        return crops

    def add_embedding_to_dict(self):

        total_frames = int(self.vdo.get(cv2.CAP_PROP_FRAME_COUNT))
        curr_frame = 1  ### <<<<<<<<<<<<<<<<<<<<<<< Moš es tgd salaboju ar to ka continue clause nebija tas curr_frame +=1, tapec vins ieciklejas ???? >>>>>>>>>>>>>>>>>>>>>>>>>>

        while curr_frame <= total_frames:
            ret, frame = self.vdo.read()
            if not ret:
                return
            detections = self.gt_dict.get(curr_frame, [])
            #print(f"Frame: {curr_frame}; detections: {detections}")
            crops = self.get_crops(frame, detections)
            # cv2.imshow(str(crops[0][0]), crops[0][1])
            # cv2.waitKey(0)
            features_batch = self.extractor.get_features_batch(crops)
            if features_batch is None:
                print(f"[Warning] frame #{curr_frame} features batch was none. Skipping iteration")
                curr_frame += 1
                continue
            #print(features_batch.shape) # Līdz šejienei viss pareizi (5, 256)...

            # Update the detections with added embedding, camera_id and split
            updated_detections = []
            for i, (veh_id, bbox) in enumerate(detections):
                embedding = features_batch[i]
                updated_detections.append({
                    'vehicle_id': veh_id,
                    'bbox': bbox,
                    'feature_vector': embedding,
                    'cam_id': self.cam_id,
                    'split': self.split
                })
                        

            # Finally, update the dict with updated detections
            self.gt_dict[curr_frame] = updated_detections

            
            #print(f"Frame: {curr_frame}; dict: \n{self.gt_dict[curr_frame]}")
            curr_frame += 1

    def save_dict_to_db(self, db):

        db.insert_whole_dict(self.gt_dict)

def main():

    db = Database()

    # try:
    embedder = EmbeddingsFromGT(vdo_path = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/vdo.avi",
                                    gt_path = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/gt/gt.txt",
                                    cam_id = "S01c004",
                                    split = "train")
    embedder.save_dict_to_db(db)
    # except:
    #     print("[Main] Embeddings not properly loaded.")
    #     return

    # cv2.destroyAllWindows()5


if __name__ == "__main__":
    main()