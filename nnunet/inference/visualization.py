import numpy as np
import os
import glob
import SimpleITK as sitk
from scipy import ndimage
import matplotlib.pyplot as plt  # 载入需要的库

# 指定数据root路径，其中data目录下是volume数据，label下是segmentation数据，都是.nii格式
data_path = r'F:\LiTS_dataset\data'
label_path = r'F:\LiTS_dataset\label'  

dataname_list = os.listdir(data_path)
dataname_list.sort()
ori_data = sitk.ReadImage(os.path.join(data_path,dataname_list[3])) # 读取其中一个volume数据
data1 = sitk.GetArrayFromImage(ori_data) # 提取数据中心的array
print(dataname_list[3],data1.shape,data1[100,255,255]) #打印数据name、shape和某一个元素的值

plt.imshow(data1[100,:,:]) # 对第100张slice可视化
plt.show()