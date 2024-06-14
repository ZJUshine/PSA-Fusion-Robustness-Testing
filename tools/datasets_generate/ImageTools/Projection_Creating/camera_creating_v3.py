# scripts for camera_creating
# with the method of watermark

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm

def cv_imread(filePath):
    # 核心就是下面这句，一般直接用这句就行，直接把图片转为mat数据
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    # imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img


def rightPaste(waterMark,origin):
    waterMark = waterMark.convert('RGBA');
    origin = origin.convert('RGBA');

    new_width = (int)(origin.width/1);
    new_height = (int)(waterMark.height*(new_width/waterMark.width));# 等比缩放

    wMark = waterMark.resize((new_width,new_height))

    # 创建底图
    baseImg = Image.new('RGBA',origin.size);

    # paste 原图和水印
    baseImg.paste(origin,(0,0),origin);
    baseImg.paste(wMark,((int)(origin.width/4),(int)((origin.height-wMark.height)/2)),wMark);

    return baseImg.convert('RGB')


def visualize_image(image):
    # Visualize the image
    plt.imshow(image)
    plt.axis("off")
    plt.show()



# Example usage
overlay_x = 0
overlay_y = 0
# os.chdir('/home/usslab/SensorFusion/Dataset')
creating_object_path = '/home/usslab/SensorFusion/sensorfusion/tools/datasets_generate/ImageTools/car.png'

waterMark = Image.open(creating_object_path)





for i in tqdm(range(0,7481)):
    img_path = '/home/usslab/SensorFusion/kitti/training/image_3/'+str(i).zfill(6)+'.png'
    origin = Image.open(img_path)
    image = rightPaste(waterMark,origin)
    visualize_image(image)
    image.save("/home/usslab/SensorFusion/kitti_attack/image_3_attack/camera_projection_creating/"+str(i).zfill(6)+'.png')