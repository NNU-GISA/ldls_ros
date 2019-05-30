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

from mask_rcnn_ros import coco
from mask_rcnn_ros import utils
from mask_rcnn_ros import model as modellib
from mask_rcnn_ros import visualize
from mask_rcnn_ros.msg import Result


# Mask-RCNN stuff. Referenced from akio mask_rcnn_ros implementation

# Local path to trained weights file
ROS_HOME = os.environ.get('ROS_HOME', os.path.join(os.environ['HOME'], '.ros'))
COCO_MODEL_PATH = os.path.join(ROS_HOME, 'mask_rcnn_coco.h5')

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


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class MaskRCNNDetector(object):
    def __init__(self):
        self._cv_bridge = CvBridge()

        config = InferenceConfig()
        config.display()

        self._visualization = True

        # Create model object in inference mode.
        self._model = modellib.MaskRCNN(mode="inference", model_dir="",
                                        config=config)
        # Load weights trained on MS-COCO
        model_path = rospy.get_param('~model_path', COCO_MODEL_PATH)
        # Download COCO trained weights from Releases if needed
        if model_path == COCO_MODEL_PATH and not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)

        self._model.load_weights(model_path, by_name=True)

        self._class_names = rospy.get_param('~class_names', CLASS_NAMES)

        self._last_msg = None
        self._msg_lock = threading.Lock()

        self._class_colors = visualize.random_colors(len(CLASS_NAMES))

        self._publish_rate = rospy.get_param('~publish_rate', 100)

    """
    # ORIGINAL RUN METHOD FOR MASK-RCNN NODE
    def run(self):
        self._result_pub = rospy.Publisher('~result', Result, queue_size=1)
        vis_pub = rospy.Publisher('~visualization', Image, queue_size=1)
        sub = rospy.Subscriber('~input', Image,
                               self._image_callback, queue_size=1)

        rate = rospy.Rate(self._publish_rate)
        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if msg is not None:
                np_image = self._cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

                # Run detection
                results = self._model.detect([np_image], verbose=0)
                result = results[0]
                result_msg = self._build_result_msg(msg, result)
                self._result_pub.publish(result_msg)

                # Visualize results
                if self._visualization:
                    vis_image = self._visualize(result, np_image)
                    cv_result = np.zeros(shape=vis_image.shape, dtype=np.uint8)
                    cv2.convertScaleAbs(vis_image, cv_result)
                    image_msg = self._cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
                    vis_pub.publish(image_msg)

            rate.sleep()
    """
    def run(self, msg):
        """

        Parameters
        ----------
        image: ndarray
            Cv2 format bgr8 image

        Returns
        -------

        """
        np_image = self._cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self._model.detect([np_image], verbose=0)
        result = results[0]
        result_msg = self._build_result_msg(msg, result)

        vis_image = self._visualize(result, np_image)
        cv_result = np.zeros(shape=vis_image.shape, dtype=np.uint8)
        cv2.convertScaleAbs(vis_image, cv_result)
        vis_msg = self._cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')

        return result_msg, vis_msg



    def _build_result_msg(self, msg, result):
        result_msg = Result()
        result_msg.header = msg.header
        for i, (y1, x1, y2, x2) in enumerate(result['rois']):
            box = RegionOfInterest()
            box.x_offset = np.asscalar(x1)
            box.y_offset = np.asscalar(y1)
            box.height = np.asscalar(y2 - y1)
            box.width = np.asscalar(x2 - x1)
            result_msg.boxes.append(box)

            class_id = result['class_ids'][i]
            result_msg.class_ids.append(class_id)

            class_name = self._class_names[class_id]
            result_msg.class_names.append(class_name)

            score = result['scores'][i]
            result_msg.scores.append(score)

            mask = Image()
            mask.header = msg.header
            mask.height = result['masks'].shape[0]
            mask.width = result['masks'].shape[1]
            mask.encoding = "mono8"
            mask.is_bigendian = False
            mask.step = mask.width
            mask.data = (result['masks'][:, :, i] * 255).tobytes()
            result_msg.masks.append(mask)
        return result_msg

    def _visualize(self, result, image):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        axes = fig.gca()
        visualize.display_instances(image, result['rois'], result['masks'],
                                    result['class_ids'], CLASS_NAMES,
                                    result['scores'], ax=axes,
                                    class_colors=self._class_colors)
        fig.tight_layout()
        canvas.draw()
        result = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

        _, _, w, h = fig.bbox.bounds
        result = result.reshape((int(h), int(w), 3))
        return result

    def _image_callback(self, msg):
        rospy.logdebug("Get an image")
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()

# rosbag writing

def write_bag(input_path, output_path, image_topic):
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
    print("Reading messages...")
    mask_rcnn = MaskRCNNDetector()
    image_count = 0
    for topic, msg, t in inbag.read_messages():
        # Copy all messages in the input into the output bag
        # Also store all images and lidar pointclouds, for M-RCNN/LDLS
        if topic == image_topic:
            result_msg, vis_msg = mask_rcnn.run(msg)
            outbag.write('mask_rcnn/result', result_msg, t)
            outbag.write('mask_rcnn/visualization', vis_msg, t)
            image_count += 1
        outbag.write(topic, msg, t)
    print("Read %d images" % (image_count))


    inbag.close()
    outbag.close()


if __name__ == '__main__':
    image_topic = "/raw_image"
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile",
                        help="path to the bagfile to process")
    args = parser.parse_args()
    bag_path = args.bagfile
    if not os.path.exists(bag_path):
        raise IOError("Bag file '%s' not found" % bag_path)
    out_name = bag_path.split('.bag')[0] + '_detections.bag'
    write_bag(bag_path, out_name, image_topic)

