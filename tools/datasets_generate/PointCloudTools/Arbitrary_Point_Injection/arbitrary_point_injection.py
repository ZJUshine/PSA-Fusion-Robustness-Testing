# 随机生成一个目标并注入，并遮盖掉目标背后的点
import numpy as np
import os
from tqdm import tqdm

# 将当前工作目录更改为脚本所在的目录
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

    distance = np.sqrt(x**2 + y**2 + z**2)  # 距离
    azimuth = np.arctan2(y, x)  # 方位角
    elevation = np.arcsin(z / distance)  # 仰角

    polar_coordinates = np.column_stack((distance, azimuth, elevation))
    return polar_coordinates

def add_random_noise(polar_coordinates):
    noise = np.random.uniform(-3, 3, size=polar_coordinates.shape[0])
    polar_coordinates[:, 0] += noise
    return polar_coordinates

def convert_to_cartesian_coordinates(polar_coordinates):
    distance = polar_coordinates[:, 0]
    azimuth = polar_coordinates[:, 1]
    elevation = polar_coordinates[:, 2]

    x = distance * np.cos(azimuth) * np.cos(elevation)
    y = distance * np.sin(azimuth) * np.cos(elevation)
    z = distance * np.sin(elevation)
    intensity = creating_object[:,3]

    cartesian_coordinates = np.column_stack((x, y, z, intensity))
    return cartesian_coordinates

 
# 读取bin文件中的数据，reshape成每行四列
creating_object = np.fromfile("./Tools/PointCloudTools/car.bin",dtype=np.float32, count=-1).reshape([-1, 4])

# truncate = creating_object[:len(creating_object)//2]
# truncate.tofile('./Arbitrary_Point_Injection/truncate.bin')

arbitrary_object = convert_to_cartesian_coordinates(add_random_noise(convert_to_polar_coordinates(creating_object)))

# print(point_cloud_benign.shape)
# print(point_cloud.shape[0])

#批量生成数据用
for i in tqdm(range(1,7481)):
    point_path = './KITTI/object/training/velodyne/'+str(i).zfill(6)+'.bin'
    arbitrary_object = convert_to_cartesian_coordinates(add_random_noise(convert_to_polar_coordinates(creating_object)))

    point_cloud_benign = np.fromfile(point_path, dtype=np.float32, count=-1).reshape([-1, 4])
    filtered_point_cloud = point_cloud_benign
    # 创建遮挡形成的阴影
    for j in range(0,arbitrary_object.shape[0]):
        polar_angles_azimuth = np.arctan2(arbitrary_object[j, 1], arbitrary_object[j, 0])
        polar_angles_vertical = np.arctan2(arbitrary_object[j, 2], arbitrary_object[j, 0])
        # 定义每个点要hiding的角度范围，
        delta = 0.01
        angle_range_azimuth = (polar_angles_azimuth-delta, polar_angles_azimuth+delta)
        angle_range_vertical = (polar_angles_vertical-delta, polar_angles_vertical+delta)
        # hiding角度范围内的点
        filtered_point_cloud = remove_points_in_angle_range(filtered_point_cloud, angle_range_azimuth, angle_range_vertical)
    # 将结果写入新的bin文件中
    point_cloud_merge = np.concatenate((filtered_point_cloud,arbitrary_object),axis=0)
    point_cloud_merge.tofile("./kitti_attack/lidar_arbitrary_point_injection/"+str(i).zfill(6)+'.bin')


# # #测试用
# point_path = './000007.bin'
# point_cloud_benign = np.fromfile(point_path, dtype=np.float32, count=-1).reshape([-1, 4])
# filtered_point_cloud = point_cloud_benign
# # 创建遮挡形成的阴影
# for j in range(0,arbitrary_object.shape[0]):
#     polar_angles_azimuth = np.arctan2(arbitrary_object[j, 1], arbitrary_object[j, 0])
#     polar_angles_vertical = np.arctan2(arbitrary_object[j, 2], arbitrary_object[j, 0])
    
#     # 定义每个点要hiding的角度范围，
#     delta = 0.01
#     angle_range_azimuth = (polar_angles_azimuth-delta, polar_angles_azimuth+delta)
#     angle_range_vertical = (polar_angles_vertical-delta, polar_angles_vertical+delta)
#     # hiding角度范围内的点
#     filtered_point_cloud = remove_points_in_angle_range(filtered_point_cloud, angle_range_azimuth, angle_range_vertical)
# # 将结果写入新的bin文件中
# point_cloud_merge = np.concatenate((filtered_point_cloud,arbitrary_object),axis=0)
# point_cloud_merge.tofile('./Arbitrary_Point_Injection/000007_API.bin')

