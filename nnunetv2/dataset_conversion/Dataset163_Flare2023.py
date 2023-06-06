from nnunetv2.paths import nnUNet_raw
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
import argparse


def convert_flare2023(dataset_path,task_id):

    task_name = "FLARE2023"
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


    train_image_folder = join(dataset_path, "imagesTr2200")
    train_label_folder = join(dataset_path, "labelsTr2200")
    test_image_folder = join(dataset_path, "validation")

    train_image = subfiles(train_image_folder, join=False, suffix='nii.gz')
    train_label = subfiles(train_label_folder, join=False, suffix='nii.gz')
    test_image = subfiles(test_image_folder, join=False, suffix='nii.gz')

    data_size = len(train_image)
    print(f"DATA SIZE: {data_size}")

    train_names = []
    test_names = []
   
    for i in train_image:
        shutil.copy(join(train_image_folder, i), join(imagestr, i))
    for i in train_label:
        train_names.append(i)
        shutil.copy(join(train_label_folder, i), join(labelstr, i))
    for i in test_image:
        test_names.append(i.split("_0000.nii.gz")[0]+".nii.gz")
        shutil.copy(join(test_image_folder, i), join(imagests, i))

    generate_dataset_json(out_base, {0: "CT"}, 
                          labels={
                            "background": 0,
                            "Liver": 1,
                            "Right kidney": 2,
                            "Spleen": 3,
                            "Pancreas": 4,
                            "Aorta":5 ,
                            "inferior vena cava": 6,
                            "right adrenal gland": 7,
                            "left adrenal gland": 8,
                            "Gallbladder": 9,
                            "Esophagus": 10,
                            "Stomach": 11,
                            "Duodenum": 12,
                            "Left kidney": 13,
                            "Tumor": 14
                            },
                          num_training_cases=len(train_image), 
                          file_ending='.nii.gz',
                          dataset_name=task_name,
                          reference='https://codalab.lisn.upsaclay.fr/competitions/12239',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_path", type=str,
                        default='/ai/data/flare2023')
    parser.add_argument('-d', required=False, type=int, default=163, help='nnU-Net Dataset ID, default: 163')
    args = parser.parse_args()
    convert_flare2023(args.dataset_path, args.d)


       