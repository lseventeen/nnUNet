from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import argparse
import random
from collections import OrderedDict
def convert_flare2023(dataset_path,task_id):

    task_name = "SEGA2023"
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


    # Dongyang_folder = subdirs(join(dataset_path, "Dongyang"), join=True)
    # KiTS_folder = subdirs(join(dataset_path, "KiTS"), join=True)    
    # Rider_folder = subdirs(join(dataset_path, "Rider"), join=True)  

    # data_folder = []
    # data_folder.extend(Dongyang_folder)
    # data_folder.extend(KiTS_folder)
    # data_folder.extend(Rider_folder)

    image_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".nrrd") and file[-8:] != "seg.nrrd":
                image_files.append(os.path.join(root, file))
    random.shuffle(image_files)
    num_train = 40
    num_test = 16

    train_files = []
    test_files = []
    # 划分图像并将它们复制到目标文件夹
    for i, image_file in enumerate(image_files):
        if i < num_train:

            train_files.append(os.path.basename(image_file.split(".")[0]))
        else:
            test_files.append(os.path.basename(image_file.split(".")[0]))
            shutil.copy(image_file, join(imagests, os.path.basename(image_file).split(".")[0]+"_0000.nrrd"))
            shutil.copy(image_file.split(".")[0]+".seg.nrrd", join(labelsts, os.path.basename(image_file).split(".")[0]+".nrrd"))
      



        # 复制图像文件到目标文件夹
        shutil.copy(image_file, join(imagestr, os.path.basename(image_file).split(".")[0]+"_0000.nrrd"))
        shutil.copy(image_file.split(".")[0]+".seg.nrrd", join(labelstr, os.path.basename(image_file).split(".")[0]+".nrrd"))
       
    # for i in data_folder:
    #     image = subfiles(i, join=False, suffix='.nrrd')
    #     for j in image:
       
    #         if j.split(".nrrd")[0][-3:]=="seg":
    #             shutil.copy(join(i,j), join(labelstr, j.split(".")[0]+".nrrd"))
          
    #         else:
    #             shutil.copy(join(i,j), join(imagestr, j.split(".")[0]+"_0000.nrrd"))
                

    generate_dataset_json(out_base, {0: "CT"}, 
                          labels={
                            "background": 0,
                            "Vessel": 1,
                            
                            },
                          num_training_cases=len(train_files)+len(test_files), 
                          file_ending='.nrrd',
                          dataset_name=task_name,
                          reference='https://multicenteraorta.grand-challenge.org/multicenteraorta/',
                        #   overwrite_image_reader_writer='NibabelIOWithReorient',
                          )
    splits = []
    splits.append(OrderedDict())
    splits[-1]['train'] = [i for i in train_files]
    splits[-1]['val'] = [i for i in test_files]
    save_json(splits, join(out_base, "splits_final.json"), sort_keys=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_path", type=str,
                        default='/ai/data/data/sega')
    parser.add_argument('-d', required=False, type=int, default=230, help='nnU-Net Dataset ID, default: 230')
    args = parser.parse_args()
    convert_flare2023(args.dataset_path, args.d)


       