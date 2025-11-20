import cv2
import supervision as sv

class Visualizer:

    def __init__(self):
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()

        self.zones = None

    def crop_visualizer(self, crops):
        
        for v_id, crop in crops:
            cv2.imshow(str(v_id),crop)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

    def get_zones(self, zones):
        if self.zones is None:
            self.zones = zones
            print("[Visualizer] Zones assigned.")
            return 1
        else:
            print("[Visualizer] Re-assigning zones is not allowed.")
            return 0


    def frame_annotator(self, frame, detections):
        if self.zones is None:
            print("[Visualizer] Zones not assigned. Cannot visualize frame.")
            return frame

        # Draw zones

        for zone in self.zones:
            x1, y1, x2, y2 = zone
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Draw bboxes, label, centerpoint

        for d in detections:
            track_id = d[0]
            bboxes = d[1]

            #bbox
            cv2.rectangle(frame, (bboxes[0], bboxes[1]), (bboxes[2], bboxes[3]), (0,255,0),1)
            #centerpoint
            c1 = bboxes[0] + (bboxes[2] - bboxes[0])/2
            c2 = bboxes[1] + (bboxes[3] - bboxes[1])/2
            frame = cv2.circle(frame, (int(c1),int(c2)), 1, (0,0,255), 5)
            #label
            font = cv2.FONT_HERSHEY_SIMPLEX
            anchor_coords = (bboxes[0], bboxes[1])
            text = str(track_id)
            cv2.putText(
                img=frame,
                text=text,
                org=anchor_coords,
                fontFace=font,
                fontScale=1.25,
                color=(0,255,255),
                thickness=4,
                lineType=cv2.LINE_AA,
            )

        return frame
    
    def visualize(self, frame, detections):
        frame = self.frame_annotator(frame, detections)

        frame = cv2.resize(frame, (1280,720))

        cv2.imshow("debug frame visualization", frame)
        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()
            return 0
        return 1

