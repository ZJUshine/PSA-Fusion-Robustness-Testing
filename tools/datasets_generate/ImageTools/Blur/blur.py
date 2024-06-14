import math
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import os
from numba import jit

# cal_blur
"""
img : Image.open(image_path)
theta : 旋转角度
dx : x方向位移
dy : y方向位移
"""
@jit
def cal_blur(img, theta, dx, dy, S=0):
    imgheight = img.size[1]
    imgwidth = img.size[0]
    imgarray = np.asarray(img)
    c0 = int(imgheight / 2)
    c1 = int(imgwidth / 2)
    delta = np.arctan(dy/dx)
    L = np.sqrt(dx*dx+dy*dy)
    theta = theta / 180 * math.pi
    blurred_imgarray = np.copy(imgarray)
    for x in range(0, imgheight):
        for y in range(0, imgwidth):
            R = math.sqrt((x - c0) ** 2 + (y - c1) ** 2)
            alpha = math.atan2(y - c1, x - c0)
            X_cos = L * math.cos(delta) - S * R * math.cos(alpha)
            Y_sin = L * math.sin(delta) - S * R * math.sin(alpha)
            N = int(max(abs(R * math.cos(alpha + theta) + X_cos + c0 - x),
                        abs(R * math.sin(alpha + theta) + Y_sin + c1 - y)))
            if N <= 0:
                continue
            count = 0
            sum_r, sum_g, sum_b = 0, 0, 0
            for i in range(0, N + 1):
                n = i / N
                xt = int(R * math.cos(alpha + n * theta) + n * X_cos + c0)
                yt = int(R * math.sin(alpha + n * theta) + n * Y_sin + c1)
                if xt < 0 or xt >= imgheight:
                    continue
                elif yt < 0 or yt >= imgwidth:
                    continue
                else:
                    sum_r += imgarray[xt, yt][0]
                    sum_g += imgarray[xt, yt][1]
                    sum_b += imgarray[xt, yt][2]
                    count += 1
            blurred_imgarray[x, y] = np.array([sum_r / count, sum_g / count, sum_b / count])
    return blurred_imgarray


# if __name__ == "__main__":
# image_paths = glob('/home/usslab/SensorFusion/Dataset/KITTI/object/training/image_2/*.png')
# for image_path in tqdm(image_paths):
#     file_path, file_name = os.path.split(image_path)
#     image_output_path = "/home/usslab/SensorFusion/kitti_attack/camera_acoustic_blur_linear/"+file_name
#     img = Image.open(image_path)
#     blurred_imgarray = cal_blur(img, 0, 15, 5)
#     blurred_img = Image.fromarray(blurred_imgarray)
#     blurred_img.save(image_output_path)

for i in tqdm(range(0,7481)):
    img_path = '/home/usslab/SensorFusion/kitti/training/image_3/'+str(i).zfill(6)+'.png'
    img = Image.open(img_path)
    blurred_imgarray = cal_blur(img, 0, 30, 30)
    blurred_img = Image.fromarray(blurred_imgarray)
    blurred_img.save("/home/usslab/SensorFusion/kitti_attack/image_3_attack/camera_acoustic_blur_linear/"+str(i).zfill(6)+'.png')