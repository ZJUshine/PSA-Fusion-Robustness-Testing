import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import os
def line_discard(img_input_path,img_output_path):
    img = cv2.imread(img_input_path)
    start_line_num = line_num = int(np.shape(img)[0]/3)
    end_line_num  = start_line_num + line_num
    img1 = np.delete(img, slice(start_line_num,end_line_num), axis=0)
    img2 = np.append(img1, img[-line_num:], axis=0)
    cv2.imwrite(img_output_path,img2)

if __name__ == '__main__':
    image_paths = glob('/home/usslab/SensorFusion/kitti/training/image_3/*.png')
    for image_path in tqdm(image_paths):
        file_path, file_name = os.path.split(image_path)
        image_output_path = "/home/usslab/SensorFusion/kitti_attack/image_3_attack/camera_truncation/"+file_name
        line_discard(image_path, image_output_path)