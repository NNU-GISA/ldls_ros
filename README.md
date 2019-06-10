# LDLS ROS
ROS package for LDLS (Label Diffusion Lidar Segmentation)

Currently, this code only supports offline processing by reading lidar point clouds and camera images from an input rosbag, and then writing Mask-RCNN and LDLS results to an output rosbag.
It is assumed that the input rosbag includes one lidar point cloud for each image, and vice versa, and the two are synchronized in time.

Requires ROS kinetic, and an Nvidia GPU. Tested on an Ubuntu 16.04 computer with an Nvidia GTX 1060.

# Dependencies

The Mask-RCNN ROS package from https://github.com/akio/mask_rcnn_ros must be installed

Also required are [numba](http://numba.pydata.org/numba-doc/latest/user/installing.html) (with CUDA support), and [cupy](https://cupy.chainer.org/).


# Usage

To generate Mask-RCNN results:
```python write_mrcnn_results.py LIDAR_AND_IMAGES.bag```

Rosbag with Mask-RCNN results will be written to `mrcnn.bag`

To generate LDLS results:
```python write_ldls_results.py mrcnn.bag```

Rosbag with LDLS results will be written to `ldls.bag`

Please see the msg/Segmentation.msg file for specification of the output Segmentation message type, which includes labeled points for detected object instances.
