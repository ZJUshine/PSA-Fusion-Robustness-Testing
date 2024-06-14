import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='kitti', type=str, help='攻击数据集')
    parser.add_argument('--attack_target', default='none', type=str, help='攻击传感器种类')
    args = parser.parse_args()

dataset_name = args.dataset_name

# dataset_name = "kitti"
# # attack lidar
# dataset_name = "lidar_emi_gaussian_noise"
# dataset_name = "lidar_laser_arbitrary_point_injection"
# dataset_name = "lidar_laser_background_noise_injection"
dataset_name = "lidar_laser_creating_car"
# dataset_name = "lidar_laser_hiding"
# # attack camera
# dataset_name = "camera_acoustic_blur_linear"
# dataset_name = "camera_emi_strip_loss"
# dataset_name = "camera_emi_truncation"
# dataset_name = "camera_laser_hiding"
# dataset_name = "camera_laser_strip_injection"
# dataset_name = "camera_projection_creating"


ROOT_PATH = "/home/usslab/Documents3/xuancun/"
DATASET_PATH = ROOT_PATH + "sensorfusion-magician/frustum-pointnets/dataset/"+dataset_name+"/object/"

# if ("kitti" in dataset_name):
#     # 配置原始数据集
#     os.makedirs(DATASET_PATH, exist_ok=True)
#     os.symlink(ROOT_PATH + "kitti/training", DATASET_PATH + "training")
    # os.symlink(ROOT_PATH + "kitti/testing", DATASET_PATH + "testing")

if ("lidar" in dataset_name):
#     os.makedirs(DATASET_PATH+"training", exist_ok=True)
#     os.symlink(ROOT_PATH + "kitti/training/calib", DATASET_PATH + "training/calib")
#     os.symlink(ROOT_PATH + "kitti/training/image_2", DATASET_PATH + "training/image_2")
#     os.symlink(ROOT_PATH + "kitti/training/label_2", DATASET_PATH + "training/label_2")
    os.symlink(ROOT_PATH + f"kitti_attack/{dataset_name}", DATASET_PATH + "training/velodyne")
#     os.symlink(ROOT_PATH + "kitti/testing", DATASET_PATH + "testing")

# if ("camera" in dataset_name):
#     os.makedirs(DATASET_PATH+"training", exist_ok=True)
#     os.symlink(ROOT_PATH + "kitti/training/calib", DATASET_PATH + "training/calib")
#     os.symlink(ROOT_PATH + f"kitti/{dataset_name}", DATASET_PATH + "training/image_2")
#     os.symlink(ROOT_PATH + "kitti/training/label_2", DATASET_PATH + "training/label_2")
#     os.symlink(ROOT_PATH + "kitti_attack/training/velodyne", DATASET_PATH + "training/velodyne")
#     os.symlink(ROOT_PATH + "kitti/testing", DATASET_PATH + "testing")

pickle_path = "/home/usslab/Documents3/xuancun/sensorfusion-magician/frustum-pointnets/kitti/" + dataset_name

# # 生成中间数据集文件
os.makedirs("/home/usslab/Documents3/xuancun/sensorfusion-magician/frustum-pointnets/kitti/"+dataset_name, exist_ok=True)
file = open("/home/usslab/Documents3/xuancun/sensorfusion-magician/frustum-pointnets/kitti/prepare_data.py", "r+")
lines = file.readlines()
lines[478] = f"        output_prefix = '{dataset_name}_frustum_caronly_'\n"
lines[487] = f"            os.path.join(BASE_DIR+'/{dataset_name}', output_prefix+'train.pickle'), \n"
lines[495] = f"            os.path.join(BASE_DIR+'/{dataset_name}', output_prefix+'val.pickle'), \n"
lines[503] = f"            os.path.join(BASE_DIR+'/{dataset_name}', output_prefix+'val_rgb_detection.pickle'), \n"
lines[165] = f"    dataset = kitti_object(os.path.join(ROOT_DIR,'dataset/{dataset_name}/object'), split)\n"
lines[281] = f"    dataset = kitti_object(os.path.join(ROOT_DIR,'dataset/{dataset_name}/object'))\n"
lines[338] = f"    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/{dataset_name}/object'), split)\n"
lines[438] = f"    dataset = kitti_object(os.path.join(ROOT_DIR, 'dataset/{dataset_name}/object'), split)\n"

if "kitti" in dataset_name:
                lines[501] = f"            os.path.join('/home/usslab/Documents3/xuancun/sensorfusion-magician/frustum-pointnets/kitti/', 'rgb_detections_{dataset_name}/rgb_detection_val.txt'), \n"
if "camera" in dataset_name:
                lines[501] = f"            os.path.join('/home/usslab/Documents3/xuancun/sensorfusion-magician/frustum-pointnets/kitti/', 'rgb_detections_{dataset_name}/rgb_detection_val.txt'), \n"
if "lidar" in dataset_name:
                lines[501] = f"            os.path.join('/home/usslab/Documents3/xuancun/sensorfusion-magician/frustum-pointnets/kitti/', 'rgb_detections_kitti/rgb_detection_val.txt'), \n"
file.seek(0)
file.writelines(lines)
file.close()

os.system("python /home/usslab/Documents3/xuancun/sensorfusion-magician/frustum-pointnets/kitti/prepare_data.py --gen_train --gen_val --gen_val_rgb_detection --car_only")


# 测试攻击数据集
os.system(f"python /home/usslab/Documents3/xuancun/sensorfusion-magician/frustum-pointnets/train/test.py --gpu 0 --num_point 1024 \
          --model frustum_pointnets_v1 --model_path /home/usslab/Documents3/xuancun/sensorfusion-magician/frustum-pointnets/train/log_v1/model.ckpt \
          --output /home/usslab/Documents3/xuancun/sensorfusion-magician/frustum-pointnets/train/detection_results_v1_{dataset_name} \
          --data_path /home/usslab/Documents3/xuancun/sensorfusion-magician/frustum-pointnets/kitti/{dataset_name}/{dataset_name}_frustum_caronly_val_rgb_detection.pickle \
          --idx_path /home/usslab/Documents3/xuancun/sensorfusion-magician/frustum-pointnets/kitti/image_sets/val.txt --from_rgb_detection")


attack_type = ["lidar_laser_creating_car"]
for attack in attack_type:
    os.system(f"/home/usslab/Documents3/xuancun/kitti_AP/evaluate_object_3d_offline_3d \
            /home/usslab/Documents3/xuancun/kitti/label_val \
            /home/usslab/Documents3/xuancun/sensorfusion-magician/frustum-pointnets/train/detection_results_v1_{attack}")