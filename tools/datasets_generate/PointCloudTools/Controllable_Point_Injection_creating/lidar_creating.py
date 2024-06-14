# creating一个目标，并遮盖掉目标背后的点
import numpy as np
import os
from tqdm import tqdm

# 将当前工作目录更改为脚本所在的目录
# os.chdir('/home/usslab/SensorFusion/Dataset')
# # 确认当前工作目录已更改
# print("Current working directory:", os.getcwd())

def remove_points_in_angle_range(point_cloud, angle_range_azimuth,angle_range_vertical):
    # 将角度范围转换为弧度,若外面已经转换了则注释掉防止二次转换
    # angle_range = np.radians(angle_range)
    # 计算每个点的极角（相对于横轴）
    polar_angles_azimuth = np.arctan2(point_cloud[:, 1], point_cloud[:, 0])
    polar_angles_vertical = np.arctan2(point_cloud[:, 2], point_cloud[:, 0])
    # 删除水平和竖直角度范围内的点
    filtered_indices = np.logical_or(np.logical_or(polar_angles_azimuth < angle_range_azimuth[0], polar_angles_azimuth > angle_range_azimuth[1]), \
        np.logical_or(polar_angles_vertical < angle_range_vertical[0], polar_angles_vertical > angle_range_vertical[1]))
    return point_cloud[filtered_indices]
    


# 读取bin文件中的数据，reshape成每行四列
#point_cloud = np.fromfile("000000.bin", dtype=np.float32).reshape(-1, 6)


creating_object = np.fromfile("/home/usslab/SensorFusion/sensorfusion/tools/datasets_generate/PointCloudTools/car_000010_bmw.bin",dtype=np.float32, count=-1).reshape([-1, 4])



# print(point_cloud_benign.shape)
# print(point_cloud.shape[0])
for i in tqdm(range(0000,5)):
    point_path = '/home/usslab/SensorFusion/kitti/training/velodyne/'+str(i).zfill(6)+'.bin'
    point_cloud_benign = np.fromfile(point_path, dtype=np.float32, count=-1).reshape([-1, 4])
    filtered_point_cloud = point_cloud_benign
    # 创建遮挡形成的阴影
    for j in range(0,creating_object.shape[0]):
        polar_angles_azimuth = np.arctan2(creating_object[j, 1], creating_object[j, 0])
        polar_angles_vertical = np.arctan2(creating_object[j, 2], creating_object[j, 0])
        
        # 定义每个点要hiding的角度范围，
        delta = 0.01
        angle_range_azimuth = (polar_angles_azimuth-delta, polar_angles_azimuth+delta)
        angle_range_vertical = (polar_angles_vertical-delta, polar_angles_vertical+delta)
        # hiding角度范围内的点
        filtered_point_cloud = remove_points_in_angle_range(filtered_point_cloud, angle_range_azimuth, angle_range_vertical)
    # 将结果写入新的bin文件中
    point_cloud_merge = np.concatenate((filtered_point_cloud,creating_object),axis=0)
    point_cloud_merge.tofile("/home/usslab/SensorFusion/kitti_attack/lidar_creating_car/"+str(i).zfill(6)+'.bin')


