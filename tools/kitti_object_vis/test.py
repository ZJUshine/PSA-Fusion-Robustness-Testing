#%%
import matplotlib.pyplot as plt
import cv2
from kitti_object import kitti_object, show_lidar_with_depth, show_lidar_on_image, \
                         show_image_with_boxes, show_lidar_topview_with_boxes

dataset = kitti_object('/home/usslab/SensorFusion/kitti', 'training')
data_idx_list = [1,2,4,5,6,8]
for data_idx in data_idx_list:
    objects = dataset.get_label_objects(data_idx)
    pc_velo = dataset.get_lidar(data_idx)
    calib = dataset.get_calibration(data_idx)
    img = dataset.get_image(data_idx)
    img_height, img_width, _ = img.shape

    img_bbox2d, img_bbox3d = show_image_with_boxes(img, objects, calib)
    img_bbox2d = cv2.cvtColor(img_bbox2d, cv2.COLOR_BGR2RGB)

    fig_bbox2d = plt.figure(figsize=(14, 7))
    ax_bbox2d = fig_bbox2d.subplots()
    ax_bbox2d.imshow(img_bbox2d)
    plt.show()
# %%
