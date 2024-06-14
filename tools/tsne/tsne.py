import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import cv2
import os

size = 224

def get_data_Image(Input_path,Label): 
    Image_names=os.listdir(Input_path)
    data=np.zeros((len(Image_names),size*size*3))
    label=np.zeros((len(Image_names),1))
 
    #为当前文件下所有图片分配自定义标签Label
    for k in range(len(Image_names)):
        label[k][0]=Label
        
    for i in range(len(Image_names)):
        image_path=os.path.join(Input_path,Image_names[i])
        img=cv2.imread(image_path)
        img=cv2.resize(img,(size,size)) #(size,size,3)
        img = img.flatten() #(3*size*size,)
        data[i]=img
    return data, label

def get_data_Lidar(Input_path,Label): 
    Lidar_names=os.listdir(Input_path)
    data=np.zeros((len(Lidar_names),400000))
    label=np.zeros((len(Lidar_names),1))
 
    #为当前文件下所有图片分配自定义标签Label
    for k in range(len(Lidar_names)):
        label[k][0]=Label
        
    for i in range(len(Lidar_names)):
        lidar_path=os.path.join(Input_path,Lidar_names[i])
        pcd_array = np.fromfile(lidar_path, dtype=np.float32)[:data.shape[1]]
        data[i]=pcd_array
    return data, label
 
# 适用于一个文件夹目录下有很多种类的情况
# 请更改路径
path = './lidar'
temp = 1
for item in os.listdir(path):
    item_path = os.path.join(path, item)
    if os.path.isdir(item_path):
        if temp == 1:
            if (path.split("/")[-1] == "lidar"):
                data, label = get_data_Lidar(item_path,temp)
            if (path.split("/")[-1] == "image"):
                data, label = get_data_Image(item_path,temp)
        else:
            if (path.split("/")[-1] == "lidar"):
                data_temp, label_temp = get_data_Lidar(item_path,temp)
            if (path.split("/")[-1] == "image"):
                data_temp, label_temp = get_data_Image(item_path,temp)
            data = np.vstack((data,data_temp))
            label = np.vstack((label,label_temp))
        temp = temp + 1
    # print(label,item_path)

# 适用于分散在不用文件夹目录下的情况

# data1, label1 = get_data('../../Dataset/KITTI/object/training/image_2/',1)
# data2, label2 = get_data('../../Dataset/kitti_attack/camera_hiding/',2)
# data1, label1 = get_data('1',1)
# data2, label2 = get_data('2',2)
# data = np.vstack((data1,data2))
# label = np.vstack((label1,label2))

tsne = TSNE(perplexity=40,n_components=3, init='pca', random_state=0)
tsne_results = tsne.fit_transform(data)
result_min, result_max = np.min(tsne_results, 0), np.max(tsne_results, 0)
tsne_results = (tsne_results - result_min) / (result_max - result_min)     # 对数据进行归一化处理

# 画图
fig = plt.figure( figsize=(8,8) )
# 2维度
ax = fig.add_subplot(1, 1, 1, title='TSNE')
scatter = ax.scatter(x=tsne_results[:,0],y=tsne_results[:,1],c=label,s=10)

# # 3维度
# ax = fig.add_subplot(111, projection='3d',title='TSNE')
# scatter = ax.scatter(tsne_results[:,0], tsne_results[:,1], tsne_results[:,2],c=label,s=10)

# legend = ax.legend(*scatter.legend_elements(),loc="best", title="Classes")
# ax.add_artist(legend)

#显示图片
plt.show()
#保存图片
plt.savefig(f'./tSNE_{path.split("/")[-1]}.jpg')