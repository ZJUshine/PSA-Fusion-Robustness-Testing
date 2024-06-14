import os
import argparse
from glob import glob
from tqdm import tqdm
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
DATASET_PATH = ROOT_PATH + "sensorfusion-magician/avod/"+dataset_name+"/object/"

# if ("Kitti" in dataset_name):
#     # 配置原始数据集
#     os.makedirs(DATASET_PATH, exist_ok=True)
#     os.symlink(ROOT_PATH + "kitti/training", DATASET_PATH + "training")
#     os.symlink(ROOT_PATH + "kitti/testing", DATASET_PATH + "testing")
#     os.symlink(ROOT_PATH + "kitti/ImageSets/train.txt", DATASET_PATH + "train.txt")
#     os.symlink(ROOT_PATH + "kitti/ImageSets/val.txt", DATASET_PATH + "val.txt")

if ("lidar" in dataset_name):
#     os.makedirs(DATASET_PATH+"training", exist_ok=True)
#     os.symlink(ROOT_PATH + "kitti/training/calib", DATASET_PATH + "training/calib")
#     os.symlink(ROOT_PATH + "kitti/training/planes", DATASET_PATH + "training/planes")
#     os.symlink(ROOT_PATH + "kitti/training/image_2", DATASET_PATH + "training/image_2")
#     os.symlink(ROOT_PATH + "kitti/training/label_2", DATASET_PATH + "training/label_2")
    os.symlink(ROOT_PATH + f"kitti_attack/{dataset_name}", DATASET_PATH + "training/velodyne")
#     os.symlink(ROOT_PATH + "kitti/ImageSets/train.txt", DATASET_PATH + "train.txt")
#     os.symlink(ROOT_PATH + "kitti/ImageSets/val.txt", DATASET_PATH + "val.txt")

# if ("camera" in dataset_name):
#     os.makedirs(DATASET_PATH+"training", exist_ok=True)
#     os.symlink(ROOT_PATH + "kitti/training/calib", DATASET_PATH + "training/calib")
#     os.symlink(ROOT_PATH + "kitti/training/planes", DATASET_PATH + "training/planes")
#     os.symlink(ROOT_PATH + f"kitti_attack/{dataset_name}", DATASET_PATH + "training/image_2")
#     os.symlink(ROOT_PATH + "kitti/training/label_2", DATASET_PATH + "training/label_2")
#     os.symlink(ROOT_PATH + "kitti/training/velodyne", DATASET_PATH + "training/velodyne")
#     os.symlink(ROOT_PATH + "kitti/ImageSets/train.txt", DATASET_PATH + "train.txt")
#     os.symlink(ROOT_PATH + "kitti/ImageSets/val.txt", DATASET_PATH + "val.txt")

file = open("/home/usslab/Documents3/xuancun/sensorfusion-magician/avod/avod/data/outputs/pyramid_cars_with_aug_example/pyramid_cars_with_aug_example.config", "r+")
lines = file.readlines()
lines[135] = f"    dataset_dir: '/home/usslab/Documents3/xuancun/sensorfusion-magician/avod/{dataset_name}/object'\n"
lines[126] = "    ckpt_indices: 120\n"
lines[127] = "    evaluate_repeatedly: False\n"
file.seek(0)
file.writelines(lines)
file.close()

file = open("/home/usslab/Documents3/xuancun/sensorfusion-magician/avod/avod/core/evaluator.py", "r+")
lines = file.readlines()
lines[145] = f"        predictions_base_dir = '/home/usslab/Documents3/xuancun/sensorfusion-magician/avod/avod/data/outputs/{dataset_name}/predictions'\n"
file.seek(0)
file.writelines(lines)
file.close()

file = open("/home/usslab/Documents3/xuancun/sensorfusion-magician/avod/avod/core/evaluator_utils.py", "r+")
lines = file.readlines()
lines[29] = f"    checkpoint_name = '{dataset_name}'\n"
file.seek(0)
file.writelines(lines)
file.close()

os.system("python avod/experiments/run_evaluation.py \
          --pipeline_config /home/usslab/Documents3/xuancun/sensorfusion-magician/avod/avod/data/outputs/pyramid_cars_with_aug_example/pyramid_cars_with_aug_example.config \
          --data_split val --device 2")


attack_type = ["lidar_laser_creating_car"]
for attack in attack_type:
    os.system(f"/home/usslab/Documents3/xuancun/kitti_AP/evaluate_object_3d_offline_3d \
            /home/usslab/Documents3/xuancun/kitti/label_val \
            /home/usslab/Documents3/xuancun/sensorfusion-magician/avod/avod/data/outputs/{attack}/predictions/kitti_native_eval/0.1/120000")

# conda activate py36tf13
# export PYTHONPATH=$PYTHONPATH:'/home/usslab/Documents3/xuancun/sensorfusion-magician/avod'
# export PYTHONPATH=$PYTHONPATH:'/home/usslab/Documents3/xuancun/sensorfusion-magician/avod/wavedata'