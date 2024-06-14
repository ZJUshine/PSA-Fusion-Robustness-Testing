#scripts for camera_hiding 
#先exposure 再blooming

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from tqdm import tqdm


def blooming(img, strength):
    #读取原始图像

    #img = cv.imread('.//SSIM//123 (1).png')

    #获取图像行和列
    rows, cols = img.shape[:2]

    #设置中心点和光照半径
    centerX = rows / 2 - 0
    centerY = cols / 2 + 0
    radius = min(centerX, centerY)

    #新建目标图像
    dst = np.zeros((rows, cols, 3), dtype="uint8")
    #图像光照特效
    for i in range(rows):

        for j in range(cols):

    #计算当前点到光照中心距离(平面坐标系中两点之间的距离)

            distance = math.pow((centerY-j), 2) + math.pow((centerX-i), 2)
            #获取原始图像
            B = img[i,j][0]
            G = img[i,j][1]
            R = img[i,j][2]

            if (distance < radius * radius):

            #按照距离大小计算增强的光照值

                result = (int)(strength*( 1.0 - math.sqrt(distance) / radius ))
                B = img[i,j][0] + result
                G = img[i,j][1] + result
                R = img[i,j][2] + result

                #判断边界 防止越界
                B = min(255, max(0, B))
                G = min(255, max(0, G))
                R = min(255, max(0, R))

                dst[i,j] = np.uint8((B, G, R))
            else:
                dst[i,j] = np.uint8((B, G, R)) 
    return dst

def adjust_exposure(image, exposure_value):
    # Load the image
    #image = cv_imread(image_path)
    plt.imshow(image)
    if image is None:
        print("Failed to read image: {}".format(image_path))
        return None
    # Convert the image to the float type
    image = image.astype(np.float32)

    # Adjust the exposure value
    image = exposure_value * image

    # Clip the pixel values that exceed the maximum value of 255
    image = np.clip(image, 0, 255)

    # Convert the image back to the uint8 type
    image = image.astype(np.uint8)

    return image
#显示图像



## 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    # 核心就是下面这句，一般直接用这句就行，直接把图片转为mat数据
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

def adjust_exposure(image, exposure_value):
    # Load the image
    #image = cv_imread(image_path)
    # plt.imshow(image)
    if image is None:
        print("Failed to read image: {}".format(image_path))
        return None
    # Convert the image to the float type
    image = image.astype(np.float32)

    # Adjust the exposure value
    image = exposure_value * image

    # Clip the pixel values that exceed the maximum value of 255
    image = np.clip(image, 0, 255)

    # Convert the image back to the uint8 type
    image = image.astype(np.uint8)

    return image

def visualize_image(image):
    # Visualize the image
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Example usage
exposure_value = 20 #exposure整体亮度
strength = 300 #blooming中心亮度

# current_dir = os.getcwd()
# print(current_dir)
# 获取当前脚本所在的目录
# script_directory = os.path.dirname(os.path.abspath(__file__))
# # 将当前工作目录更改为脚本所在的目录
# os.chdir('/home/usslab/SensorFusion/Dataset')
# # 确认当前工作目录已更改
# print("Current working directory:", os.getcwd())

# os.chdir('/home/usslab/SensorFusion/Dataset')
for i in tqdm(range(3617,7481)):

    img_path = '/home/usslab/SensorFusion/kitti/training/image_3/'+str(i).zfill(6)+'.png'
    image = cv2.imread(img_path)
    # image = cv_imread(img_path)
    image = adjust_exposure(image, exposure_value)
    image = blooming(image,strength)

    #visualize_image(image)
    cv2.imwrite("/home/usslab/SensorFusion/kitti_attack/image_3_attack/camera_laser_hiding/"+str(i).zfill(6)+'.png',image)