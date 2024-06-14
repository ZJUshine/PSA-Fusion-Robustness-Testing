# scripts for camera_creating
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def overlay_image(background_image_path, overlay_image_path, x, y):
    # 读取道路图片和车的图片
    road = cv2.imread(background_image_path)
    car = cv2.imread(overlay_image_path)

    # 缩放车的图片以适应道路
    car_resized = cv2.resize(car, (100, 200))

    # 创建掩码
    mask = np.zeros(car_resized.shape[:2], dtype=np.uint8)
    mask[car_resized[:,:,0] > 0] = 255

    # 获取车的轮廓
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 将车的图片叠加到道路上
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        car_roi = car_resized[y:y+h, x:x+w]
        road_roi = road[y:y+h, x:x+w]
        car_mask = mask[y:y+h, x:x+w]
        car_mask = cv2.cvtColor(car_mask, cv2.COLOR_GRAY2BGR)
        car_mask = car_mask / 255.0
        road_roi[:] = road_roi * (1.0 - car_mask) + car_roi * car_mask

    return road

    # # Load the background and overlay images
    # background_image = cv2.imread(background_image_path)
    # overlay_image = cv2.imread(overlay_image_path)

    # # Get the height and width of the overlay image
    # height, width = overlay_image.shape[:2]

    # # Define the region of interest on the background image
    # roi = background_image[y:y+height, x:x+width]

    # # Create a mask of the overlay image
    # gray = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2GRAY)
    # ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # mask = cv2.bitwise_not(mask)

    # # Black out the region of interest on the background image
    # background_image = cv2.bitwise_and(roi, roi, mask=mask)

    # # Add the overlay image to the background image
    # result = cv2.add(background_image, overlay_image)

    # # Assign the result back to the region of interest on the background image
    # background_image[y:y+height, x:x+width] = result

    # return background_image

def visualize_image(image):
    # Visualize the image
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# Example usage
overlay_x = 0
overlay_y = 0
os.chdir('/home/usslab/SensorFusion/Dataset')
creating_object_path = './Tools/ImageTools/car.png'
for i in range(0,1):
    img_path = './KITTI/object/training/image_2/'+str(i).zfill(6)+'.png'
    image = overlay_image(img_path, creating_object_path, overlay_x, overlay_y)
    visualize_image(image)
    #cv2.imwrite("./kitti_attack/camera_creating/"+str(i).zfill(6)+'.png',image)