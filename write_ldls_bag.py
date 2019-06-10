import rospy
import rosbag
import argparse
import os
import threading
import numpy as np

import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image, RegionOfInterest
from sensor_msgs.point_cloud2 import read_points, create_cloud_xyz32

from src.segmentation import LidarSegmentationResult, LidarSegmentation
from src.detections import MaskRCNNDetections
from src.utils import Projection

from mask_rcnn_ros.msg import Result
from ldls_ros.msg import Segmentation

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: CLASS_NAMES.index('teddy bear')
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def msg_to_detections(result_msg):
    """

    Parameters
    ----------
    result_msg: Result

    Returns
    -------

    """
    class_ids = np.array(result_msg.class_ids)
    scores = np.array(result_msg.scores)
    rois = None # not needed
    bridge = CvBridge()
    if len(result_msg.masks) == 0:
        shape = (1208, 1920)
        masks = np.empty((1208, 1920, 0))
    else:
        masks_list = [bridge.imgmsg_to_cv2(m, 'mono8') for m in result_msg.masks]
        shape = masks_list[0].shape
        masks = np.stack(masks_list, axis=2)
        masks[masks==255] = 1
    return MaskRCNNDetections(shape, rois, masks, class_ids, scores)


def write_bag(input_path, output_path, mrcnn_results_topic, lidar_topic):
    """
    Reads an input rosbag, and writes an output bag including all input bag
    messages as well as Mask-RCNN results, written to the following topics:
    mask_rcnn/result: mask_rcnn_ros.Result
    mask_rcnn/visualization: Image

    Parameters
    ----------
    input_path: str
    output_path: str
    image_topic: str

    Returns
    -------

    """
    inbag = rosbag.Bag(input_path, 'r')
    outbag = rosbag.Bag(output_path, 'w')
    lidar_list = []
    lidar_headers = []
    lidar_t = []
    start = inbag.get_start_time()
    mrcnn_directory = 'mrcnn/'
    if not os.path.exists(mrcnn_directory):
        os.mkdir(mrcnn_directory)

    # Lidar-to-camera rotation and translation
    R = np.array([[0.6979, -0.7161, -0.0112],
                [-0.0184, -0.0023, -0.9998],
                [0.7160, 0.6980, -0.0148]])
    T = np.array([-0.0876, -0.1172, -0.0710])

    Tr = np.zeros((4,4))
    Tr[0:3,0:3] = R
    Tr[0:3, 3] = T
    Tr[3,3] = 1

    # Camera intrinsics matrix
    intrins = 1.0e3 * np.array([[1.8239, 0, 0],
                                [-0.0042, 1.8228, 0],
                                [0.9743, 0.6661, 0.0010]])
    P = np.zeros((3,4))
    P[0:3,0:3] = intrins.T
    projection = Projection(Tr, P)

    # Write all input messages to the output
    print("Reading messages...")
    for topic, msg, t in inbag.read_messages():
        outbag.write(topic, msg, t)

    # start_time = rospy.Time.from_sec(inbag.get_start_time() + 110)

    # Generate LDLS results
    for topic, msg, t in inbag.read_messages(topics=[lidar_topic]):
        point_gen = read_points(msg)
        points = np.array([p for p in point_gen])
        lidar_list.append(points[:,0:3])
        lidar_headers.append(msg.header)
        lidar_t.append(t)
    print("Running LDLS...")
    lidarseg = LidarSegmentation(projection)
    i=0
    for topic, msg, t in inbag.read_messages(topics=[mrcnn_results_topic]):
        if i % 50 == 0:
            print("Message %d..." % i)
        detections = msg_to_detections(msg)
        # Get the class IDs, names, header from the MRCNN message
        class_ids = msg.class_ids
        class_names = list(msg.class_names)
        lidar = lidar_list[i]
        header = lidar_headers[i]
        ldls_res = lidarseg.run(lidar, detections, save_all=False)
        lidar = lidar[ldls_res.in_camera_view,:]

        ldls_msg = Segmentation()
        ldls_msg.header = header
        ldls_msg.class_ids = class_ids
        ldls_msg.class_names = class_names
        instance_ids = []
        pc_msgs = []
        # Get segmented point cloud for each object instance
        labels = ldls_res.instance_labels()
        class_labels = ldls_res.class_labels()
        for inst in range(1, len(class_names)+1):
            in_instance = labels == inst
            if np.any(in_instance):
                instance_ids.append(inst-1)
                inst_points = lidar[in_instance,:]
                pc_msg = create_cloud_xyz32(header, inst_points)
                pc_msgs.append(pc_msg)
        ldls_msg.instance_ids = instance_ids
        ldls_msg.object_points = pc_msgs
        foreground = lidar[class_labels != 0, :]
        foreground_msg = create_cloud_xyz32(header, foreground)

        outbag.write('/ldls/segmentation', ldls_msg, t)
        outbag.write('/ldls/foreground', foreground_msg, t)
        i += 1
    inbag.close()
    outbag.close()


if __name__ == '__main__':
    mrcnn_results_topic = 'mask_rcnn/result'
    lidar_topic = '/velo3/pointcloud'
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile",
                        help="path to the bagfile to process")
    args = parser.parse_args()
    bag_path = args.bagfile
    if not os.path.exists(bag_path):
        raise IOError("Bag file '%s' not found" % bag_path)
    out_name = bag_path.split('.bag')[0] + '_ldls.bag'
    write_bag(bag_path, out_name, mrcnn_results_topic, lidar_topic)

