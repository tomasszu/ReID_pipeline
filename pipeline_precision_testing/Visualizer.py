import cv2

class Visualizer:

    def __init__(self):
        pass

    def crop_visualizer(self, crops):
        
        for v_id, crop in crops:
            cv2.imshow(str(v_id),crop)
            cv2.waitKey(0)

        cv2.destroyAllWindows()