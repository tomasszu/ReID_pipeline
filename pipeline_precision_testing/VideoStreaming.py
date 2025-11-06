import cv2


class VideoStreaming:
    
    def __init__(self, video_src, gt_src):
    
        self.video = cv2.VideoCapture(video_src)
        gt_file = open(gt_src, 'r')
        self.gt_lines = gt_file.readlines()

    def next_frame():
        pass