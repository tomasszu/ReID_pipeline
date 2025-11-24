import cv2
import os
import numpy as np
import sys

sys.path.append('/home/tomass/tomass/ReID_pipele/pipeline_precision_testing')

from FeatureExtract import FeatureExtractor

class EmbeddingsFromGT:
    def __init__(self, vdo_path, gt_path, cam_id, db, split):

        self.extractor = FeatureExtractor()

        #load video file
        self.vdo = cv2.VideoCapture(vdo_path)
        if not self.vdo.isOpened():
            raise IOError(f"Could not open video: {vdo_path}")
        
        #load gt
        with open(gt_path, 'r') as f:
            gt_lines = [line.strip() for line in f.readlines() if line.strip()]
        
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

        curr_frame = 1

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
            print(features_batch.shape) # Līdz šejienei viss pareizi (5, 256)...



            curr_frame += 1


    


def main():

    db = None

    # try:
    embedder = EmbeddingsFromGT(vdo_path = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/vdo.avi",
                                    gt_path = "/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/gt/gt.txt",
                                    cam_id = "S01c004",
                                    db = db,
                                    split = "train")
    # except:
    #     print("[Main] Embeddings not properly loaded.")
    #     return

    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()