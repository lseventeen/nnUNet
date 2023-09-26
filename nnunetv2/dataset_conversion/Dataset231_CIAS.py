from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import train_test_split
import shutil
import argparse
import random
from collections import OrderedDict
def convert_CIAS(dataset_path,task_id):

    task_name = "CIAS"
    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")
    if isdir(imagestr):
        shutil.rmtree(imagestr)
        shutil.rmtree(imagests)
        shutil.rmtree(labelstr)
        shutil.rmtree(labelsts)
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)
    maybe_mkdir_p(labelsts)

    images_path = join(dataset_path, "images")
    labels_path = join(dataset_path, "labels")


    image_files = subfiles(images_path, join=False, suffix='nii.gz')
    random.shuffle(image_files)

    # 计算划分数量
    num_train = 150
    num_test = 30
    num_val = 20
    train_files = []
    val_files = []
    test_files = []
    # 划分图像并将它们复制到目标文件夹
    for i, image_file in enumerate(image_files):
        if i < num_train:
            tag_image_dir = imagestr
            tag_label_dir = labelstr

            train_files.append(f"CIAS_{str(i).zfill(4)}")
        elif i < num_train + num_test:
            tag_image_dir = imagests
            tag_label_dir = labelsts
            test_files.append(f"CIAS_{str(i).zfill(4)}")
        elif i < num_train + num_test + num_val:
            tag_image_dir = imagestr
            tag_label_dir = labelstr
            val_files.append(f"CIAS_{str(i).zfill(4)}")
        else:
            continue



        # 复制图像文件到目标文件夹
        shutil.copy(join(images_path,image_file), join(tag_image_dir, f"CIAS_{str(i).zfill(4)}_0000.nii.gz"))
        shutil.copy(join(labels_path,image_file.split("_")[0]+".nii.gz"), join(tag_label_dir, f"CIAS_{str(i).zfill(4)}.nii.gz"))
       



                

    generate_dataset_json(out_base, {0: "CT"}, 
                          labels={
                            "background": 0,
                            "vessel": 1,
                            
                            },
                          num_training_cases=len(train_files)+len(val_files), 
                          file_ending='.nii.gz',
                          dataset_name=task_name,
                          reference='',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          )
    
    splits = []
    splits.append(OrderedDict())
    splits[-1]['train'] = [i for i in train_files]
    splits[-1]['val'] = [i for i in val_files]
    save_json(splits, join(out_base, "splits_final.json"), sort_keys=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_path", type=str,
                        default='/ai/data/data/CTA_celebral_vessel')
    parser.add_argument('-d', required=False, type=int, default=231, help='nnU-Net Dataset ID, default: 231')
    args = parser.parse_args()
    convert_CIAS(args.dataset_path, args.d)


       