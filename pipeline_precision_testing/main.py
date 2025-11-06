## Code to test the model Re-Identification capability in real life circumstances
## aka no guardrails (except Object detection part is simulated with bboxes from GT)
## Re-Identification must be done even to establish uniqueness in the saved identities pool of the same camera aka to see if sighting of the vehicle is a unique instance or has already been spotted.

## The plan
## Load in Frames with GT
## Pass detections through zone saving

import argparse
from VideoStreaming import VideoStreaming

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_source', type=str, default='/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/vdo.avi', help='Path to the first video file. (Re-Identification)')
    parser.add_argument('--gt_source', type=str, default='/home/tomass/tomass/data/AIC22_Track1_MTMC_Tracking(1)/train/S01/c004/gt/gt.txt', help='Path to the first video file. (Re-Identification)')
    parser.add_argument('--play_mode', type=int, default=200, help='Delay between frames in milliseconds. Set to 0 for manual frame stepping (Pressing Enter for new frame).')

    return parser.parse_args()


def main(args):

    video_src = args.video_source
    gt_src = args.gt_source

    streamer = VideoStreaming(video_src,gt_src)







if __name__ == "__main__":
    args = parse_args()
    main(args)