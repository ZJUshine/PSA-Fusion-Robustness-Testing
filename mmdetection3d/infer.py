import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pickle

# original dataset
# dataset_name = "kitti"
# # attack lidar
# dataset_name = "lidar_emi_gaussian_noise"
# dataset_name = "lidar_laser_arbitrary_point_injection"
# dataset_name = "lidar_laser_background_noise_injection"
# dataset_name = "lidar_laser_creating_car"
# dataset_name = "lidar_laser_hiding"
# # attack camera
# dataset_name = "camera_acoustic_blur_linear"
# dataset_name = "camera_emi_strip_loss"
# dataset_name = "camera_emi_truncation"
# dataset_name = "camera_laser_hiding"
# dataset_name = "camera_laser_strip_injection"
dataset_name = "camera_projection_creating"

ROOT_PATH = "/home/usslab/SensorFusion/"
DATASET_PATH = ROOT_PATH + "sensorfusion/mmdetection3d/data/"+dataset_name
# if ("kitti" in dataset_name):
#     # 配置原始数据集
#     os.makedirs(DATASET_PATH, exist_ok=True)
#     os.symlink(ROOT_PATH + "kitti/training", DATASET_PATH + "/training")
#     os.makedirs(DATASET_PATH + "/training/velodyne_reduced", exist_ok=True)
#     os.symlink(ROOT_PATH + "kitti/ImageSets", DATASET_PATH + "/ImageSets")

# # 配置攻击数据集
# if ("lidar" in dataset_name):
#     os.makedirs(DATASET_PATH+"/training", exist_ok=True)
#     os.symlink(ROOT_PATH + "kitti/training/calib", DATASET_PATH + "/training/calib")
#     os.symlink(ROOT_PATH + "kitti/training/image_2", DATASET_PATH + "/training/image_2")
#     os.symlink(ROOT_PATH + "kitti/training/label_2", DATASET_PATH + "/training/label_2")
#     os.symlink(ROOT_PATH + "kitti/training/planes", DATASET_PATH + "/training/planes")
#     os.symlink(ROOT_PATH + f"kitti_attack/{dataset_name}", DATASET_PATH + "/training/velodyne")
#     os.makedirs(DATASET_PATH + "/training/velodyne_reduced", exist_ok=True)
#     os.symlink(ROOT_PATH + "kitti/ImageSets", DATASET_PATH + "/ImageSets")

# # 配置攻击数据集
# if ("camera" in dataset_name):
#     os.makedirs(DATASET_PATH+"/training", exist_ok=True)
#     os.symlink(ROOT_PATH + "kitti/training/calib", DATASET_PATH + "/training/calib")
#     os.symlink(ROOT_PATH + f"kitti_attack/{dataset_name}", DATASET_PATH + "/training/image_2")
#     os.symlink(ROOT_PATH + f"kitti_attack/image_3_attack/{dataset_name}", DATASET_PATH + "/training/image_3")
#     os.symlink(ROOT_PATH + "kitti/training/label_2", DATASET_PATH + "/training/label_2")
#     os.symlink(ROOT_PATH + "kitti/training/planes", DATASET_PATH + "/training/planes")
#     os.symlink(ROOT_PATH + "kitti/training/velodyne", DATASET_PATH + "/training/velodyne")
#     os.symlink(ROOT_PATH + "kitti/ImageSets", DATASET_PATH + "/ImageSets")


# 配置攻击数据集
# os.system(f"python tools/create_data.py kitti --root-path ./data/{dataset_name} --out-dir ./data/{dataset_name} --extra-tag kitti --with-plane")

# file = open("/home/usslab/SensorFusion/sensorfusion/mmdetection3d/configs/_base_/datasets/kitti-3d-car.py", "r+")
# lines = file.readlines()
# lines[2] = f"data_root = 'data/{dataset_name}/'\n"
# file.seek(0)
# file.writelines(lines)
# file.close()

# # 雷达模型
# # second
# config_file = 'configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py'
# checkpoint_file = 'checkpoints/second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth'
# os.system(f"python ./tools/test.py {config_file} {checkpoint_file} --cfg-options 'test_evaluator.submission_prefix=./second_{dataset_name}_results'")

# #相机模型
# # imvoxelnet
# file = open("/home/usslab/SensorFusion/sensorfusion/mmdetection3d/configs/imvoxelnet/imvoxelnet_8xb4_kitti-3d-car.py", "r+")
# lines = file.readlines()
# lines[80] = f"data_root = 'data/{dataset_name}/'\n"
# file.seek(0)
# file.writelines(lines)
# file.close()
# config_file = 'configs/imvoxelnet/imvoxelnet_8xb4_kitti-3d-car.py'
# checkpoint_file = 'checkpoints/imvoxelnet_4x8_kitti-3d-car_20210830_003014-3d0ffdf4.pth'
# os.system(f"python ./tools/test.py {config_file} {checkpoint_file} --cfg-options 'test_evaluator.submission_prefix=./imvoxelnet_{dataset_name}_results'")

# # mvxnet
# file = open("/home/usslab/SensorFusion/sensorfusion/mmdetection3d/configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py", "r+")
# lines = file.readlines()
# lines[144] = f"data_root = 'data/{dataset_name}/'\n"
# file.seek(0)
# file.writelines(lines)
# file.close()
# config_file = 'configs/mvxnet/mvxnet_fpn_dv_second_secfpn_8xb2-80e_kitti-3d-3class.py'
# checkpoint_file = 'checkpoints/dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class_20210831_060805-83442923.pth'
# os.system(f"python ./tools/test.py {config_file} {checkpoint_file} --cfg-options 'test_evaluator.submission_prefix=./mvxnet_{dataset_name}_results'")


# # 计算AP
# attack_type = ["kitti", "lidar_emi_gaussian_noise", "lidar_laser_arbitrary_point_injection", "lidar_laser_background_noise_injection", "lidar_laser_creating_car", "lidar_laser_hiding"]
# for attack in attack_type:
#     os.system(f"/home/usslab/SensorFusion/sensorfusion/tools/kitti_AP/evaluate_object_3d_offline_3d \
#             /home/usslab/SensorFusion/kitti/training/label_2 \
#             /home/usslab/SensorFusion/sensorfusion/mmdetection3d/results/second_{attack}_results")

# 计算AP
attack_type = ["kitti", "camera_laser_hiding", "camera_projection_creating", "camera_laser_strip_injection", "camera_emi_strip_loss", "camera_emi_truncation","camera_acoustic_blur_linear",\
               "lidar_emi_gaussian_noise", "lidar_laser_arbitrary_point_injection", "lidar_laser_background_noise_injection", "lidar_laser_creating_car", "lidar_laser_hiding"]
for attack in attack_type:
    os.system(f"/home/usslab/SensorFusion/sensorfusion/tools/kitti_AP/evaluate_object_3d_offline_3d \
            /home/usslab/SensorFusion/kitti/training/label_2 \
            /home/usslab/SensorFusion/sensorfusion/mmdetection3d/results/mvxnet/mvxnet_{attack}_results")
    





