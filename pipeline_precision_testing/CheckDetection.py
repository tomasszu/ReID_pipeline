import numpy as np

class CheckDetection:
    def __init__(self, rows=None, cols=None,
                 area_bottom_left = None, area_top_right=None):

        self.zones = []

        # Crop zone parameters
        self.rows = rows
        self.cols = cols

        # Set the cropping area
        self.area_bottom_left = area_bottom_left  # (x_min, y_max)
        self.area_top_right = area_top_right      # (x_max, y_min)

        self.zone_of_detections = {}

        # asserts that the points are of valid structure
        def _assert_point(name, point):
            assert isinstance(point, (tuple, list)), f"{name} must be a tuple or list"
            assert len(point) == 2, f"{name} must have exactly two elements (x, y)"
            assert all(isinstance(v, (int, float)) for v in point), f"{name} values must be int or float"

        if self.rows is not None and self.cols is not None \
        and self.area_bottom_left is not None and self.area_top_right is not None:

            _assert_point("area_bottom_left", self.area_bottom_left)
            _assert_point("area_top_right", self.area_top_right)

            self.zones = self._generate_zones()
            self.use_zones = True
        else:
            self.zones = []
            self.use_zones = False  # zone filtering disabled

    def _generate_zones(self):
        """Generates the zones based on the specified rows and columns within the defined area.
        Returns:
            list: A list of tuples representing the zones, each defined by its top-left and bottom-right coordinates.
        """
        x_min, y_max = self.area_bottom_left
        x_max, y_min = self.area_top_right

        zone_width = (x_max - x_min) / self.cols
        zone_height = (y_max - y_min) / self.rows
        zones = []

        for i in range(self.rows):
            for j in range(self.cols):
                x1 = x_min + j * zone_width
                y1 = y_min + i * zone_height
                x2 = x_min + (j + 1) * zone_width
                y2 = y_min + (i + 1) * zone_height
                zones.append((int(x1), int(y1), int(x2), int(y2)))

        return zones



    def perform_checks(self, track_id, bbox):
        # Perform checks on the bounding box

        if self.use_zones:
            if self.check_crop_zones(track_id, bbox):
                # If the bounding box is in a crop zone, return True
                return True
        else: 
            # If there arent crop zones, we don't check crop zones
            return True
    # If the bounding box is not in the attention area or crop zone, return False
        return False

    def verify_attention(self, bbox):
        
        center_point = self.get_center(bbox)

        attention = self.is_point_in_attention(center_point)

        return attention
    
    def check_crop_zones(self, track_id, bbox):

        center_point = self.get_center(bbox)

        zone = self.zone_of_point(center_point)
        if zone != -1:
            # If the point is in a zone
            if((track_id not in self.zone_of_detections) or (self.zone_of_detections[track_id] != zone)):
                # If the track_id is not in the dictionary or the zone has changed
                # Update the zone of detections for the track_id
                self.zone_of_detections.update({track_id: zone})
                return True
        else:
            # If the point is not in any zone, remove the track_id from the dictionary
            if track_id in self.zone_of_detections:
                del self.zone_of_detections[track_id]
                return False
        # If the point is not in any zone, return False
            return False


    def zone_of_point(self, point):
        """
        Determine the zone index in which a given point lies.
        
        Args:
        - point (tuple): (x, y) coordinates of the point
        - zones (list): List of zone coordinates, each zone represented as ((x1, y1), (x2, y2))
        
        Returns:
        - zone_index (int): Index of the zone in which the point lies, or -1 if it's not in any zone
        """
        x, y = point
        
        # Iterate through each zone and check if the point lies within it
        for zone_index, zone in enumerate(self.zones):
            x1, y1, x2, y2 = zone
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone_index
        
        # If the point is not in any zone, return -1
        return -1




    def get_center(self,bbox):
        
        bbox_center = np.array([(bbox[0] + bbox[2]) / 2,
                                (bbox[1] + bbox[3]) / 2])
        
        return bbox_center
    
    # def is_point_in_attention(self,point):

    #     vector1 = self.attention_vector1
    #     vector2 = self.attention_vector2
        
    #     if vector1 is not None and vector2 is not None:
    #         # Assuming vector1 and vector2 are represented by [x, y] points
    #         vector1, sign1 = vector1
    #         vector2, sign2 = vector2
    #         v1p1, v1p2 = vector1
    #         v2p1, v2p2 = vector2
    #         # Calculate the cross product to determine if the point is on the same side of the line
    #         if(sign1 == ">"):
    #             cross_product = (v1p2[0]-v1p1[0])*(point[1] - v1p1[1]) - (v1p2[1] - v1p1[1]) *(point[0] - v1p1[0]) > 0
    #         else:
    #             cross_product = (v1p2[0]-v1p1[0])*(point[1] - v1p1[1]) - (v1p2[1] - v1p1[1]) *(point[0] - v1p1[0]) < 0
    #         if(sign2 == ">"):
    #             cross_product2 = (v2p2[0]-v2p1[0])*(point[1] - v2p1[1]) - (v2p2[1] - v2p1[1]) *(point[0] - v2p1[0]) > 0
    #         else:
    #             cross_product2 = (v2p2[0]-v2p1[0])*(point[1] - v2p1[1]) - (v2p2[1] - v2p1[1]) *(point[0] - v2p1[0]) < 0
    #     elif(vector1 is not None):
    #         vector1, sign1 = vector1
    #         v1p1, v1p2 = vector1
    #         if(sign1 == ">"):
    #             cross_product = (v1p2[0]-v1p1[0])*(point[1] - v1p1[1]) - (v1p2[1] - v1p1[1]) *(point[0] - v1p1[0]) > 0
    #         else:
    #             cross_product = (v1p2[0]-v1p1[0])*(point[1] - v1p1[1]) - (v1p2[1] - v1p1[1]) *(point[0] - v1p1[0]) < 0
    #         cross_product2 = True
    #     else:
    #         cross_product = True
    #         cross_product2 = True

    #     cross_product = cross_product and cross_product2

    #     # If the cross product is positive, the point is on the same side as the frame
    #     return cross_product
        


        

