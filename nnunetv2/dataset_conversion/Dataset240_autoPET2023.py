from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import argparse


def convert_flare2023(dataset_path,task_id):

    task_name = "autoPET2023"
    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    if isdir(imagestr):
        shutil.rmtree(imagestr)
        shutil.rmtree(imagests)
        shutil.rmtree(labelstr)
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    all_data_folder = []
    data_folder = subdirs(dataset_path, join=True)
    for i in data_folder:
        data_subfolder = subdirs(i, join=True)
        all_data_folder.extend(data_subfolder)


    # KiTS_folder = subdirs(join(dataset_path, "KiTS"), join=True)    
    # Rider_folder = subdirs(join(dataset_path, "Rider"), join=True)  


    for id,folder in enumerate(all_data_folder):
        data = subfiles(folder, join=False, suffix='.nii.gz')
        print(id)
        for i in data:
       
            if i[:5]=="CTres":
                shutil.copy(join(folder,i), join(imagestr, "autoPET_"+'{:0>4}'.format(id)+"_0000.nii.gz"))
            elif i[:3]=="SUV":
                shutil.copy(join(folder,i), join(imagestr, "autoPET_"+'{:0>4}'.format(id)+"_0001.nii.gz"))
            elif i[:3]=="SEG":
                shutil.copy(join(folder,i), join(labelstr, "autoPET_"+'{:0>4}'.format(id)+".nii.gz"))

                

    generate_dataset_json(out_base, {0: "CT",1: "PET"}, 
                          labels={
                            "background": 0,
                            "lesion": 1,
                            
                            },
                          num_training_cases=len(all_data_folder), 
                          file_ending='.nii.gz',
                          dataset_name=task_name,
                          reference='https://multicenteraorta.grand-challenge.org/multicenteraorta/',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_path", type=str,
                        default='/ai/data/FDG-PET-CT-Lesions')
    parser.add_argument('-d', required=False, type=int, default=240, help='nnU-Net Dataset ID, default: 240')
    args = parser.parse_args()
    convert_flare2023(args.dataset_path, args.d)


       