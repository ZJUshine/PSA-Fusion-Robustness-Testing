# 在一定区域内注入随机噪声

import numpy as np
import os
from tqdm import tqdm



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
    

#point_cloud = np.fromfile("000000.bin", dtype=np.float32).reshape(-1, 6)

def convert_to_polar_coordinates(point_cloud):
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]
    intensity = point_cloud[:, 3]

    distance = np.sqrt(x**2 + y**2 + z**2)  # 距离
    azimuth = np.arctan2(y, x)  # 方位角
    elevation = np.arcsin(z / distance)  # 仰角

    polar_coordinates = np.column_stack((distance, azimuth, elevation,intensity))
    return polar_coordinates

def convert_to_cartesian_coordinates(polar_coordinates):
    distance = polar_coordinates[:, 0]
    azimuth = polar_coordinates[:, 1]
    elevation = polar_coordinates[:, 2]

    x = distance * np.cos(azimuth) * np.cos(elevation)
    y = distance * np.sin(azimuth) * np.cos(elevation)
    z = distance * np.sin(elevation)
    intensity = polar_coordinates[:,3]

    cartesian_coordinates = np.column_stack((x, y, z, intensity))

    cartesian_coordinates_2 = np.column_stack((x, y, -z, intensity))

    cartesian_coordinates_m = np.concatenate((cartesian_coordinates,cartesian_coordinates_2),axis=0)

    num_points = cartesian_coordinates_m.shape[0]
    indices_to_delete = np.random.choice(num_points, num_points // 2, replace=False)
    filtered_point_cloud = np.delete(cartesian_coordinates_m, indices_to_delete, axis=0)

    return filtered_point_cloud

def random_noise_in_angle_range(point_cloud, angle_range_azimuth,angle_range_vertical):
    polar_angles_azimuth = np.arctan2(point_cloud[:, 1], point_cloud[:, 0])
    polar_angles_vertical = np.arctan2(point_cloud[:, 2], point_cloud[:, 0])
    filtered_indices = np.logical_or(np.logical_or(polar_angles_azimuth < angle_range_azimuth[0], polar_angles_azimuth > angle_range_azimuth[1]), \
        np.logical_or(polar_angles_vertical < angle_range_vertical[0], polar_angles_vertical > angle_range_vertical[1]))
   
    pointcloud_benign = point_cloud[filtered_indices]

    inverted_indices = np.logical_not(filtered_indices)
    pointcloud_random = point_cloud[inverted_indices]

    return pointcloud_benign, pointcloud_random


def add_random_noise(polar_coordinates,Noise):
    noise = np.random.uniform(-1, Noise, size=polar_coordinates.shape[0])
    polar_coordinates[:, 0] += noise
    
    # polar_coordinates_2 = polar_coordinates_1
    # polar_coordinates_2[:,2] = -polar_coordinates_1[:,2]

    # polar_coordinates = np.concatenate((polar_coordinates_1,polar_coordinates_2),axis=0)

    return polar_coordinates




angle_range_azimuth = (np.deg2rad(-10), np.deg2rad(10))
angle_range_vertical = (np.deg2rad(-10), np.deg2rad(0))
Noise = 10


for i in tqdm(range(6319,7481)):
    point_path = os.path.dirname(os.path.abspath(__file__))+'/../../../kitti/training/velodyne/'+str(i).zfill(6)+'.bin'
    # print(point_path)
    point_cloud = np.fromfile(point_path, dtype=np.float32, count=-1).reshape([-1, 4])

    pointcloud_benign, pointcloud_random = random_noise_in_angle_range(point_cloud, angle_range_azimuth,angle_range_vertical)
    point_cloud_noise = convert_to_cartesian_coordinates(add_random_noise(convert_to_polar_coordinates(pointcloud_random),Noise))

    point_cloud_merge = np.concatenate((point_cloud,point_cloud_noise),axis=0)

    point_cloud_merge.tofile(os.path.dirname(os.path.abspath(__file__))+'/../../../kitti_attack/lidar_background_noise_injection/'+str(i).zfill(6)+'.bin')
