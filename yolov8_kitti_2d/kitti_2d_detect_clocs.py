import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from ultralytics import YOLO
import os
from glob import glob
from tqdm import tqdm
for dataset_name in ['camera_acoustic_blur_linear','camera_emi_strip_loss','camera_emi_truncation','camera_laser_hiding','camera_laser_strip_injection','camera_projection_creating']:
    image_paths = sorted(glob(f"/home/usslab/SensorFusion/kitti_attack/{dataset_name}/*.png"))

    os.makedirs(f'/home/usslab/SensorFusion/sensorfusion/CLOCs/d2_detection_data/{dataset_name}/data',exist_ok=True)
    model = YOLO('yolov8x.pt')
    for image_path in tqdm(image_paths):
        file_temp = open(f'/home/usslab/SensorFusion/sensorfusion/CLOCs/d2_detection_data/{dataset_name}/data/'+os.path.basename(image_path).split('.')[0]+'.txt', 'w')
        results = model(image_path)
        boxes = results[0].boxes.boxes.cpu().numpy()
        for t in range(boxes.shape[0]):
            if boxes[t, 5] == 2:
                file_temp.write('Car -1 -1 -10 %.2f %.2f %.2f %.2f -1 -1 -1 -1000 -1000 -1000 -10 %.4f\n' % (boxes[t, 0],boxes[t, 1],boxes[t, 2],boxes[t, 3],boxes[t,4]))
        file_temp.close()
