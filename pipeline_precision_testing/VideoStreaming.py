import cv2


class VideoStreaming:
    
    def __init__(self, video_src, gt_src):
        # Function opens video stream and reads in gt file
        self.video = cv2.VideoCapture(video_src)
        if not self.video.isOpened():
            raise IOError(f"Could not open video: {video_src}")
        
        with open(gt_src, 'r') as f:
            self.gt_lines = [line.strip() for line in f.readlines() if line.strip()]

        # Parse gt into a dict
        # {frame_id: [(veh_id, (x1, y1, x2, y2)), ...]}

        self.gt_dict = {}

        for line in self.gt_lines:
            parts = line.split(',')
            frame_id = int(parts[0])
            veh_id = int(parts[1])
            x, y, w, h = map(int, parts[2:6])
            x1, y1, x2, y2 = x, y, x + w, y + h
            if frame_id not in self.gt_dict:
                self.gt_dict[frame_id] = []
            self.gt_dict[frame_id].append((veh_id, (x1, y1, x2, y2)))

        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        # the current frame variable helps align with the gt file, where frames start from #1
        self.current_frame = 1
        self.current_frame_img = None

    def next_frame(self):
        # Function Reads video frame one by one

        ret, frame = self.video.read()
        if not ret:
            return None, None
        self.current_frame_img = frame.copy()
        
        detections = self.gt_dict.get(self.current_frame, [])
    

        self.current_frame += 1
        return frame, detections
    
    def get_crops(self,detections):
        crops = []

        for veh_id, (x1, y1, x2, y2) in detections:
            crop = self.current_frame_img[y1:y2, x1:x2]
            if crop.size != 0:
                crops.append((veh_id, crop))
            else:
                print("[CROP] Warning, rejected crop! Crop size = 0.")

        return crops
    
    def get_single_crop(self, bbox):
        x1, y1, x2, y2 = bbox
        crop = self.current_frame_img[y1:y2, x1:x2]
        if crop.size != 0:
            return crop
        else:
            print("[CROP] Warning, rejected single crop! Crop size = 0.")
            return None

    
    def release(self):
        # Release video capture
        self.video.release()