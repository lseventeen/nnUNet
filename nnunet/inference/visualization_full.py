from tkinter import N
import numpy as np
import os
import glob
import SimpleITK as sitk
from scipy import ndimage
import cv2
from PIL import Image
import matplotlib.pyplot as plt  # 载入需要的库
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
def set_colour(img_path,data_path,name):
    img_nii = sitk.ReadImage(img_path)
    pre_nii = sitk.ReadImage(data_path) # 读取其中一个volume数据

    img = sitk.GetArrayFromImage(img_nii) # 提取数据中心的array
    pre = sitk.GetArrayFromImage(pre_nii) # 提取数据中心的array

    # img = img_3D[slice,...]
    # pre = pre_3D[slice,...]
    path = f"nnunet/inference/save_picture/{name}"
    if isdir(path):
        shutil.rmtree(path)
        maybe_mkdir_p(path)
    else:
        maybe_mkdir_p(path)
    S,H,W = img.shape
    for s in range(S):
        colour = np.zeros((H,W,3))
        for i in range(H):
            for j in range(W):
                if pre[s,i,j] == 1:
                    colour[i,j,:] = [0,0,255] # 纯红
                elif pre[s,i,j] == 2:
                    colour[i,j,:] = [255,255,0] #青色
                elif pre[s,i,j] == 3:
                    colour[i,j,:] = [0,128,0] #纯绿
                elif pre[s,i,j] == 4:
                    colour[i,j,:] = [203,255,192] #粉红  128,128,0
                elif pre[s,i,j] == 5:
                    colour[i,j,:] = [0,128,128] #橄榄
                elif pre[s,i,j] == 6:
                    colour[i,j,:] = [255,0,0] #蓝色
                elif pre[s,i,j] == 7:
                    colour[i,j,:] = [255,0,255] #纯黄
                elif pre[s,i,j] == 8:
                    colour[i,j,:] = [128,0,128] #紫色
                elif pre[s,i,j] == 9:
                    colour[i,j,:] = [80,127,255] #珊瑚 	255,127,80
                elif pre[s,i,j] == 10:
                    colour[i,j,:] = [0,0,128] #栗色  	128,0,0
                elif pre[s,i,j] == 11:
                    colour[i,j,:] = [218,185,255] #桃色
                elif pre[s,i,j] == 12:
                    colour[i,j,:] = [235,206,135] #	天蓝色  135,206,235
                elif pre[s,i,j] == 13:
                    colour[i,j,:] = [221,160,221] #李子 	221,160,221
                else:
                    colour[i,j,:] = [img[s,i,j],img[s,i,j],img[s,i,j]]
        print(s)
        cv2.imwrite(join(path,f"{s}_{name}.png"),colour)
    # colour = Image.fromarray(colour.astype('uint8')).convert('RGB')
    # colour.save(f"nnunet/inference/save_picture/{name}.png")
    

def remove_files(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))

# def picture2video(name):
#     path = f"nnunet/inference/save_picture/{name}"
#     filelist = os.listdir(path)
#     size = cv2.imread(join(path,filelist[0])).shape
#     N = len(filelist)
#     # filelist.sort()
#     fps = 24 #视频每秒24帧
#     size = (640, 480) #需要转为视频的图片的尺寸
#     #可以使用cv2.resize()进行修改
#     video = cv2.VideoWriter("VideoTest1.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
#     #视频保存在当前目录下

#     for item in filelist:
        
#         if item.endswith('.png'): 
#         #找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
#             item = path + item
#             img = cv2.imread(item)
#             video.write(img)

#     video.release()
#     cv2.destroyAllWindows()




if __name__ == '__main__':
    img="/home/lwt/data/synapse/RawData/Training/img/img0001.nii"
    phtrans="/home/lwt/code/nnUNet_trained_models/nnUNet/3d_fullres/Task017_AbdominalOrganSegmentation/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/PFTC_dpR0.2_dpC_0.1_220224_114831/model_best/ABD_001.nii"
    nnformer="/home/lwt/code/nnUNet_trained_models/nnUNet/3d_fullres/Task017_AbdominalOrganSegmentation/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/nnformer_pretrain_211130_190450/model_best/ABD_001.nii"
    cotr="/home/lwt/code/nnUNet_trained_models/nnUNet/3d_fullres/Task017_AbdominalOrganSegmentation/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/cotr_220102_220809/model_best/ABD_001.nii"
    nnunet="/home/lwt/code/nnUNet_trained_models/nnUNet/3d_fullres/Task017_AbdominalOrganSegmentation/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/nnUNet_220127_001435/model_best/ABD_001.nii"
    gt = "/home/lwt/data_pro/nnUNet_preprocessed/Task017_AbdominalOrganSegmentation/gt_segmentations/ABD_001.nii"


    # set_colour(img,phtrans,"phtrans")
    # set_colour(img,gt,"gt")
    picture2video("phtrans")