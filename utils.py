from sensor_msgs.point_cloud2 import read_points
import numpy as np


class SegmentationResult():

    def __init__(self, msg):
        """
        Convert a Segmentation msg to a Python class
        Segmented object point clouds are converted to N by 3 numpy arrays

        Parameters
        ----------
        msg: Segmentation
        """
        self.class_ids = np.array(msg.class_ids)
        self.class_names = np.array(msg.class_names)
        self.object_points = []
        for pc_msg in msg.object_points:
            pointgen = read_points(pc_msg)
            pc = np.array([p for p in pointgen])
            self.object_points.append(pc)
