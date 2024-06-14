# hiding掉指定锥形范围内的点
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
    
# 将当前工作目录更改为脚本所在的目录
os.chdir('/home/usslab/SensorFusion/Dataset')
# 确认当前工作目录已更改
print("Current working directory:", os.getcwd())

# 定义要hiding的角度范围，
# angle_range = (np.deg2rad(0), np.deg2rad(45))
angle_range_azimuth = (np.deg2rad(-10), np.deg2rad(10))
angle_range_vertical = (np.deg2rad(-20), np.deg2rad(20))

for i in tqdm(range(1000,7481)):
    point_path = './KITTI/object/training/velodyne/'+str(i).zfill(6)+'.bin'
    # 读取bin文件中的数据，reshape成每行四列
    point_cloud = np.fromfile(point_path, dtype=np.float32, count=-1).reshape([-1, 4])
    # hiding角度范围内的点
    filtered_point_cloud = remove_points_in_angle_range(point_cloud, angle_range_azimuth, angle_range_vertical)
    # 将结果写入新的bin文件中
    filtered_point_cloud.tofile("./kitti_attack/lidar_hiding/"+str(i).zfill(6)+'.bin')





