## Code to test the model Re-Identification capability in real life circumstances
## aka no guardrails (except Object detection part is simulated with bboxes from GT)
## Re-Identification must be done even to establish uniqueness in the saved identities pool of the same camera aka to see if sighting of the vehicle is a unique instance or has already been spotted.

## The plan
## Load in Frames with GT
## Pass detections through zone saving

import argparse
from VideoStreaming import VideoStreaming
from Visualizer import Visualizer
from FeatureExtract import FeatureExtractor
from database import Database

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_source', type=str, default='/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/vdo.avi', help='Path to the first video file. (Re-Identification)')
    parser.add_argument('--gt_source', type=str, default='/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/gt/gt.txt', help='Path to the first video file. (Re-Identification)')
    parser.add_argument('--play_mode', type=int, default=200, help='Delay between frames in milliseconds. Set to 0 for manual frame stepping (Pressing Enter for new frame).')
    parser.add_argument('--camera_id', type=int, default=4, help='Camera ID for the current video source.')

    return parser.parse_args()


def main(args):

    video_src = args.video_source
    gt_src = args.gt_source
    cam_id = args.camera_id

    streamer = VideoStreaming(video_src,gt_src)
    extractor = FeatureExtractor()
    db = Database()

    # -------------------- Optional: crop visualization---------------------
    # visualizer = Visualizer()
    # ----------------------------------------------------------------------

    while True:
        # 1. Gathering Detections (Video stream + gt.txt)
        # -----------------------------------------------
        frame, detections, crops = streamer.next_frame()
        if frame is None:
            break

        print(f"Frame {streamer.current_frame -1}: {len(detections)} detections")

        # -------------------- Optional: crop visualization---------------------
        # visualizer.crop_visualizer(crops)
        # ----------------------------------------------------------------------

        # 2. Feature Extraction for each crop
        # -----------------------------------------------
        if len(crops) > 0:
            crops_images = [crop[1] for crop in crops]
            features = extractor.get_features(crops_images)
            #print(features.shape)
            if features is None:
                break
        else:
            print("[Feature Extraction] Warning: no crops detected in this frame - nothing to extract.")

        # 3. Saving features to database (Vehicle ID, Camera ID, Feature Vector)
        # -----------------------------------------------






    streamer.release()

if __name__ == "__main__":
    args = parse_args()
    main(args)