## Code to test the model Re-Identification capability in real life circumstances
## aka no guardrails (except Object detection part is simulated with bboxes from GT)
## Re-Identification must be done even to establish uniqueness in the saved identities pool of the same camera aka to see if sighting of the vehicle is a unique instance or has already been spotted.

## The plan
## Load in Frames with GT
## Pass detections through zone saving
import os
import argparse
import yaml
import torch
import numpy as np


from VideoStreaming import VideoStreaming
from Visualizer import Visualizer
from FeatureExtract import FeatureExtractor
from database import Database
from CheckDetection import CheckDetection
from ResultsTally import ResultsTally

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_source', type=str, default='/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/vdo.avi', help='Path to the first video file. (Re-Identification)')
    parser.add_argument('--gt_source', type=str, default='/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/gt/gt.txt', help='Path to the first video file. (Re-Identification)')
    parser.add_argument('--play_mode', type=int, default=200, help='Delay between frames in milliseconds. Set to 0 for manual frame stepping (Pressing Enter for new frame).')
    parser.add_argument('--camera_id', type=int, default=4, help='Camera ID for the current video source.')
    parser.add_argument('--input_conf', type=str, default='pipeline_precision_testing/inputs_conf.yaml',
                        help='Path to the input configuration file, that contains crop zones and mqtt topics for receiving info from each camera')

    return parser.parse_args()

def load_embedding_from_disc(path):
    if os.path.exists(path):
        embedding = torch.load(path).cpu().numpy()
        return np.array(embedding)
    else:
        return None

def main(args):

    video_src = args.video_source
    gt_src = args.gt_source
    cam_id = args.camera_id

    streamer = VideoStreaming(video_src,gt_src)
    extractor = FeatureExtractor()
    db = Database()
    results = ResultsTally(db)

    # For crop zone filtering
    with open(args.input_conf, "r") as f:
        config = yaml.safe_load(f)
    
    cam_params = config['streams'].get(f'camera_{cam_id}', None)
    if cam_params is None:
        print(f"[Error] No configuration found for camera ID {cam_id} in {args.input_conf}. Exiting Test.")
        return
    
    zone_check = CheckDetection(
        cam_params["crop_zone_rows"],
        cam_params["crop_zone_cols"],
        tuple(cam_params["crop_zone_area_bottom_left"]),
        tuple(cam_params["crop_zone_area_top_right"])
    )

    # Optional. Tampering with embeddings
    avg_features = load_embedding_from_disc("/home/tomass/tomass/ReID_pipele/embeddings/AI_City_Images/averaged/average_cam4.pt")
    if avg_features is None:
        print("[Fatal Error] Avg. embedding load failed.")
        return

    # -------- Optional:  crop display or full frame visualization----------
    visualizer = Visualizer()
    if not visualizer.get_zones(zone_check.zones):
        return
    # ----------------------------------------------------------------------

    while True:
        # 1. Gathering Detections (Video stream + gt.txt)
        # -----------------------------------------------
        frame, detections = streamer.next_frame()
        if frame is None:
            break

        #print(f"Frame {streamer.current_frame -1}: {len(detections)} detections")

        # -------- Optional: full frame visualization---------------------------
        # visualizer.crop_visualizer(crops)
        # if not visualizer.visualize(frame, detections):
        #     break
        # ----------------------------------------------------------------------
        # 2. Crop zone filtering
        # -----------------------------------------------

        nr_filtered = 0 
        for d in detections:
            track_id = d[0]
            bbox = d[1]  # (x_min, y_min, x_max, y_max)

            if zone_check.perform_checks(track_id, bbox):
                nr_filtered += 1

                crop = streamer.get_single_crop(bbox)
                if crop is not None:
                    #--------Optional crop display ------
                    #visualizer.crop_visualizer([[track_id, crop]])
                    #------------------------------------

                    # 3. Feature Extraction for each crop
                    # -----------------------------------------------
                    features = extractor.get_features([crop])
                    if features is None:
                        break
                    # <<<<<<<<<<<<<<<<<<<<<<<<< Testing changes to embeddings >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

                    #features = features - avg_features
                    # <<<<<<<<<<<<<<<<<<<<<<<<< !!!!!!!!!!!!!!!!!!!!!!!!!!!!! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                    #print(features.shape()) #shape: (1, 256)

                    # 4. Saving features to database (Vehicle ID, Camera ID, Feature Vector)
                    # -----------------------------------------------
                    db.query(track_id, features.squeeze(), cam_id)

                else:
                    print("[Feature Extraction] Error: no crop returned in this frame - nothing to extract.")
                    break

        # 5. Update test results
        #------------------------------------------------
        if nr_filtered:
            print(f"\n[Frame #{streamer.current_frame}]")
            results.display_results()

    results.complete_results()


                
        #print(f"Frame {streamer.current_frame -1}: {nr_filtered} filtered detections")
            




    streamer.release()

if __name__ == "__main__":
    args = parse_args()
    main(args)