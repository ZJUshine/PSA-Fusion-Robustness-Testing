# 给点云添加噪声
import numpy as np
import os
from tqdm import tqdm

# 获取当前脚本文件的绝对路径
script_path = os.path.abspath(__file__)

# 获取文件所在的目录路径
file_directory = os.path.dirname(script_path)

# 将当前工作目录更改为脚本所在的目录
# os.chdir(file_directory)
os.chdir('/home/usslab/SensorFusion/Dataset')
# 确认当前工作目录已更改
print("Current working directory:", os.getcwd())

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

def add_random_noise(polar_coordinates,Noise):
    noise = np.random.uniform(-Noise, Noise, size=polar_coordinates.shape[0])
    polar_coordinates[:, 0] += noise
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
    return cartesian_coordinates




# print(point_cloud_benign.shape)
# print(point_cloud.shape[0])

Noise = 0.1 #set noise level, meters

#批量生成数据用
for i in tqdm(range(0,7481)):
    point_path = './KITTI/object/training/velodyne/'+str(i).zfill(6)+'.bin'
    point_cloud_benign = np.fromfile(point_path, dtype=np.float32, count=-1).reshape([-1, 4])
    point_cloud_noise = convert_to_cartesian_coordinates(add_random_noise(convert_to_polar_coordinates(point_cloud_benign),Noise))
    point_cloud_noise.tofile("./kitti_attack/lidar_gaussian_noise/"+str(i).zfill(6)+'.bin')


#测试用

# Noise = 0.1 #set noise level, meters
# point_path = '../000007.bin'
# point_cloud_benign = np.fromfile(point_path, dtype=np.float32, count=-1).reshape([-1, 4])
# point_cloud_noise = convert_to_cartesian_coordinates(add_random_noise(convert_to_polar_coordinates(point_cloud_benign),Noise))


# point_cloud_noise.tofile('./000007_Noise.bin')

