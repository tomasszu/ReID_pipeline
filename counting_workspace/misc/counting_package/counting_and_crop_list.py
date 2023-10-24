from typing import Dict, Optional

import cv2
import numpy as np

import supervision as sv
#from supervision.detection.core import Detections
# from supervision.draw.color import Color
# from supervision.geometry.core import Point, Rect, Vector


class countZone:
    """
    Count the number of objects that have entered and later exit the zone.
    """

    def __init__(self, x: sv.Point, y: sv.Point, w: sv.Point, h: sv.Point):
        """
        Initialize a countZone object.

        Attributes:
            The four edges of the zone

        """
        self.zone = sv.Rect(x,y,w,h)
        self.tracker_state: Dict[str, bool] = {}
        self.triggers  = []
        #self.in_count: int = 0
        self.out_count: int = 0

    def trigger(self, detections: sv.Detections):
        """
        Update the out_count for the detections that cross the into the zone and later leave.

        Attributes:
            detections (Detections): The detections for which to update the counts.

        """
        #print(detections) 

        #bboxes of detections that have been counted (left count zone) and need to be saved for future ReID
        crop_detections = []

        for xyxy, _, confidence, class_id, tracker_id in detections:
            # handle detections with no tracker_id
            if tracker_id is None:
                continue
            
            # anchor point for the detection bounding box
            bbox_anchor = np.array(
                [
                    (xyxy[0] + xyxy[2]) / 2,
                    (xyxy[1] + xyxy[3]) / 2,
                ]
            ).transpose()

            # Either add detection to triggers if it entered the zone or remove it and add +1 to out_count if one that entered has left
            if(self.pointInZone(bbox_anchor) is True):
                if(not self.triggers.__contains__(tracker_id)): self.triggers.append(tracker_id)
            else:
                if(self.triggers.__contains__(tracker_id)):
                    self.out_count = self.out_count+1
                    self.triggers.remove(tracker_id)
                    crop_detections.append([tracker_id,xyxy])
        return crop_detections


    def pointInZone(self, point):
        x1, y1, w, h = self.zone.x, self.zone.y, self.zone.width, self.zone.height
        x2, y2 = x1+w, y1+h
        x, y = point
        if (x1 < x and x < x2):
            #print(f"{x1} < {x} and {x} < {x2}")
            if (y1 > y and y > y2):
                #print(f"{y1} > {y} and {y} > {y2}")
                return True
        return False


            

class ZoneAnnotator:
    def __init__(
        self,
        thickness: float = 2,
        color: sv.Color = sv.Color.white(),
        text_thickness: float = 2,
        text_color: sv.Color = sv.Color.black(),
        text_scale: float = 0.5,
        text_offset: float = 1.5,
        text_padding: int = 10,
        custom_in_text: Optional[str] = None,
        custom_out_text: Optional[str] = None,
    ):
        """
        Initialize the LineCounterAnnotator object with default values.

        Attributes:
            thickness (float): The thickness of the line that will be drawn.
            color (Color): The color of the line that will be drawn.
            text_thickness (float): The thickness of the text that will be drawn.
            text_color (Color): The color of the text that will be drawn.
            text_scale (float): The scale of the text that will be drawn.
            text_offset (float): The offset of the text that will be drawn.
            text_padding (int): The padding of the text that will be drawn.

        """
        self.thickness: float = thickness
        self.color: sv.Color = color
        self.text_thickness: float = text_thickness
        self.text_color: sv.Color = text_color
        self.text_scale: float = text_scale
        self.text_offset: float = text_offset
        self.text_padding: int = text_padding
        self.custom_in_text: str = custom_in_text
        self.custom_out_text: str = custom_out_text

    def annotate(self, frame: np.ndarray, zone_counter: countZone) -> np.ndarray:

        out_text = (
            f"{self.custom_out_text}: {countZone.out_count}"
            if self.custom_out_text is not None
            else f"Out: {zone_counter.out_count}"
        )

        start_point: sv.Point
        end_point: sv.Point
        start_point = zone_counter.zone.top_left.as_xy_int_tuple()
        end_point = zone_counter.zone.bottom_right.as_xy_int_tuple()
        frame = cv2.rectangle(frame, start_point, end_point, [0,0,255], self.thickness)

        out_text_x = int(
            (zone_counter.zone.x)
        )
        out_text_y = int(
            (zone_counter.zone.y + 30)
        )


        cv2.putText(frame, out_text, (out_text_x, out_text_y), cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, [255,255,255], self.text_thickness, cv2.LINE_AA)

        return frame