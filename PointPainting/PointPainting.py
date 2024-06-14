import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='kitti', type=str, help='攻击数据集')
    parser.add_argument('--attack_target', default='none', type=str, help='攻击传感器种类')
    args = parser.parse_args()

dataset_name = args.dataset_name

# original dataset
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

ROOT_PATH = "/home/usslab/SensorFusion/"
DATASET_PATH = ROOT_PATH + "sensorfusion/PointPainting/detector/data/"+dataset_name

# if ("kitti" in dataset_name):
    # # 配置原始数据集
    # os.makedirs(DATASET_PATH, exist_ok=True)
    # os.symlink(ROOT_PATH + "kitti/training", DATASET_PATH + "/training")
    # os.symlink(ROOT_PATH + "kitti/kitti_infos_train.pkl", DATASET_PATH + "/kitti_infos_train.pkl")
    # os.symlink(ROOT_PATH + "kitti/kitti_infos_val.pkl", DATASET_PATH + "/kitti_infos_val.pkl")

# if ("lidar" in dataset_name):
#     os.makedirs(DATASET_PATH+"/training", exist_ok=True)
#     os.symlink(ROOT_PATH + "kitti/training/calib", DATASET_PATH + "/training/calib")
#     os.symlink(ROOT_PATH + "kitti/training/image_2", DATASET_PATH + "/training/image_2")
#     os.symlink(ROOT_PATH + "kitti/training/image_3", DATASET_PATH + "/training/image_3")
#     os.symlink(ROOT_PATH + "kitti/training/label_2", DATASET_PATH + "/training/label_2")
#     os.symlink(ROOT_PATH + f"kitti_attack/{dataset_name}", DATASET_PATH + "/training/velodyne")
    # os.symlink(ROOT_PATH + "kitti/kitti_infos_train.pkl", DATASET_PATH + "/kitti_infos_train.pkl")
    # os.symlink(ROOT_PATH + "kitti/kitti_infos_val.pkl", DATASET_PATH + "/kitti_infos_val.pkl")

# if ("camera" in dataset_name):
#     os.makedirs(DATASET_PATH+"/training", exist_ok=True)
#     os.symlink(ROOT_PATH + "kitti/training/calib", DATASET_PATH + "/training/calib")
#     os.symlink(ROOT_PATH + f"kitti_attack/{dataset_name}", DATASET_PATH + "/training/image_2")
    
#     os.symlink(ROOT_PATH + "kitti/training/label_2", DATASET_PATH + "/training/label_2")
#     os.symlink(ROOT_PATH + "kitti/training/velodyne", DATASET_PATH + "/training/velodyne")
#     os.symlink(ROOT_PATH + "kitti/kitti_infos_train.pkl", DATASET_PATH + "/kitti_infos_train.pkl")
#     os.symlink(ROOT_PATH + "kitti/kitti_infos_val.pkl", DATASET_PATH + "/kitti_infos_val.pkl")
#     os.symlink(ROOT_PATH + f"kitti_attack/image_3_attack/{dataset_name}", DATASET_PATH + "/training/image_3")


# 生成中间数据集文件
# file = open("/home/usslab/SensorFusion/sensorfusion/PointPainting/painting/painting.py", "r+")
# lines = file.readlines()
# lines[15] = f"TRAINING_PATH = '/home/usslab/SensorFusion/sensorfusion/PointPainting/detector/data/{dataset_name}/training/'\n"
# file.seek(0)
# file.writelines(lines)
# file.close()

# os.system("python /home/usslab/SensorFusion/sensorfusion/PointPainting/painting/painting.py")



# file = open("/home/usslab/SensorFusion/sensorfusion/PointPainting/detector/tools/cfgs/dataset_configs/painted_kitti_dataset.yaml", "r+")
# lines = file.readlines()
# lines[1] = f"DATA_PATH: '/home/usslab/SensorFusion/sensorfusion/PointPainting/detector/data/{dataset_name}'\n"
# file.seek(0)
# file.writelines(lines)
# file.close()


# os.chdir("/home/usslab/SensorFusion/sensorfusion/PointPainting/detector/tools")
# os.system(f"python test.py \
#           --cfg_file '/home/usslab/SensorFusion/sensorfusion/PointPainting/detector/tools/cfgs/kitti_models/pointpillar_painted.yaml' \
#           --ckpt_dir '/home/usslab/SensorFusion/sensorfusion/PointPainting/detector/output/kitti_models/pointpillar_painted/default/ckpt' \
#           --ckpt '/home/usslab/SensorFusion/sensorfusion/PointPainting/detector/output/kitti_models/pointpillar_painted/default/ckpt/checkpoint_epoch_80.pth' \
#           --eval_tag '{dataset_name}' \
#           --save_to_file")


# os.system(f"/home/usslab/SensorFusion/sensorfusion/tools/kitti_AP/evaluate_object_3d_offline_3d \
#         /home/usslab/SensorFusion/kitti/training/label_2 \
#         /home/usslab/SensorFusion/sensorfusion/PointPainting/detector/output/kitti_models/pointpillar_painted/default/eval/epoch_80/val/default/final_result")
# 计算AP
# 
# attack_type = ["kitti", "lidar_emi_gaussian_noise", "lidar_laser_arbitrary_point_injection", "lidar_laser_background_noise_injection", "lidar_laser_creating_car", "lidar_laser_hiding", \
#                "camera_acoustic_blur_linear", "camera_emi_strip_loss", "camera_emi_truncation", "camera_laser_hiding", "camera_laser_strip_injection", "camera_projection_creating"]
attack_type = ["lidar_laser_creating_car"]
for attack in attack_type:
    os.system(f"/home/usslab/SensorFusion/sensorfusion/tools/kitti_AP/evaluate_object_3d_offline_3d \
            /home/usslab/SensorFusion/kitti/training/label_2 \
            /home/usslab/SensorFusion/sensorfusion/PointPainting/detector/output/home/usslab/SensorFusion/sensorfusion/PointPainting/detector/tools/cfgs/kitti_models/pointpillar_painted/default/eval/epoch_80/val/{attack}/final_result")