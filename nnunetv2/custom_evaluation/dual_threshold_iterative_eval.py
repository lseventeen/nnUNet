#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import shutil
from copy import deepcopy

from nnunet.inference.segmentation_export import save_segmentation_nifti
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
from multiprocessing import Pool
from nnunet.postprocessing.connected_components import apply_postprocessing_to_folder, load_postprocessing
from nnunet.paths import *
from phtrans.postprocessing.dual_threshold_iterative import DTI_3D
from phtrans.evaluation.SurfaceDice import compute_dice_coefficient
from phtrans.evaluation.cldice import clDice
import argparse
import os 
import numpy as np
import nibabel as nb
from nnunet.paths import *
from collections import OrderedDict
import pandas as pd
from skimage.measure import label,regionprops
def DTI(files, properties_files, out_file, high_threshold,low_threshold,override, store_npz):
    if override or not isfile(out_file):
        softmax = np.load(files)['softmax']
        # seg = np.where(softmax[1] > 0.7,1,0)
        seg = DTI_3D(softmax,h_thresh= high_threshold, l_thresh=low_threshold)
        props = load_pickle(properties_files) 
        # Softmax probabilities are already at target spacing so this will not do any resampling (resampling parameters
        # don't matter here)
        save_segmentation_nifti(seg, out_file,  props, order=3)
        if store_npz:
            np.savez_compressed(out_file[:-7] + ".npz", softmax=seg)
            save_pickle(props, out_file[:-7] + ".pkl")

       


def mutil_process_eval(folder, threads, high_threshold,low_threshold,override=True, store_npz=False):
    tag=str(high_threshold)+str(low_threshold)
    output_folder = join(folder,tag)
    maybe_mkdir_p(output_folder)
    
    patient_ids = subfiles(folder, suffix=".npz", join=False) 
    
    patient_ids = [i[:-4] for i in patient_ids]
    patient_ids = np.unique(patient_ids)

    assert all([isfile(join(folder, i + ".npz")) for i in patient_ids]), "Not all patient npz are available in " \
                                                                       "all folders"
    assert all([isfile(join(folder, i + ".pkl")) for i in patient_ids]), "Not all patient pkl are available in " \
                                                                        "all folders"

    files = []
    property_files = []
    out_files = []
    for p in patient_ids:
        files.append(join(folder, p + ".npz"))
        property_files.append(join(folder, p + ".pkl") )
        out_files.append(join(output_folder, p + ".nii.gz"))

    p = Pool(threads)
    p.starmap(DTI, zip(files, property_files,out_files,[high_threshold] * len(out_files),[low_threshold] * len(out_files),[override] * len(out_files), [store_npz] * len(out_files)))
    p.close()
    p.join()


    if postprocessing_file is not None:
        for_which_classes, min_valid_obj_size = load_postprocessing(postprocessing_file)
        print('Postprocessing...')
        apply_postprocessing_to_folder(output_folder, output_folder_orig,
                                       for_which_classes, min_valid_obj_size, threads)
        shutil.copy(postprocessing_file, output_folder_orig)

    gt_path = join(nnUNet_raw_data,"Task210_cerebral_vessel_cta/labelsTs")
    save_path = '/root/code/PHTrans_seg/PHTrans/evaluation_results/cerebral_vessel_cta'
    vessel_eval(tag,output_folder,gt_path,save_path)



def vessel_eval(tag,seg_path,gt_path,save_path):

    join = os.path.join
    
   

    save_name = tag+".cvs"
    filenames = os.listdir(seg_path)
    filenames = [x for x in filenames if x.endswith('.nii.gz')]
    filenames.sort()
    dice_all = []
    cldice_all = []
    connected_domain_num_all = []
    seg_metrics = OrderedDict()
    seg_metrics['Case'] = list()
    seg_metrics['Dice'] = list()
    seg_metrics['clDice'] = list()
    seg_metrics['CD_num'] = list()

    for name in filenames:
        gt_data = np.uint8(nb.load(join(gt_path, name)).get_fdata())
        seg_data = np.uint8(nb.load(join(seg_path, name)).get_fdata())
        dice = compute_dice_coefficient(gt_data, seg_data)
        cldice = clDice(gt_data, seg_data)
        _,connected_domain_num = label(seg_data,connectivity = 2,return_num=True)
        dice_all.append(dice)
        cldice_all.append(cldice)
        connected_domain_num_all.append(connected_domain_num)
        seg_metrics['Case'].append(name)
        seg_metrics['Dice'].append(round(dice, 4))
        seg_metrics['clDice'].append(round(cldice, 4))
        seg_metrics['CD_num'].append(connected_domain_num)
        print(name, "Done")
    seg_metrics['Case'].append("Mean")
    seg_metrics['Dice'].append(round(np.array(dice_all).mean(), 4))
    seg_metrics['clDice'].append(round(np.array(cldice_all).mean(), 4))
    seg_metrics['CD_num'].append(round(np.array(connected_domain_num_all).mean(), 4))
    
    print(round(np.array(dice_all).mean(), 4))
    print(round(np.array(cldice_all).mean(), 4))
    print(round(np.array(connected_domain_num_all).mean(), 4))
    dataframe = pd.DataFrame(seg_metrics)
    dataframe.to_csv(join(save_path, save_name), index=False)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Do dual_threshold_iterative on softmax npz file")
    parser.add_argument('-f', '--folder',help="the folder of npz files", default=None, required=False)
#     parser.add_argument('-o', '--output_folder', help="where to save the results", default=None,required=True, type=str)
    parser.add_argument('-t', '--threads', help="number of threads used to saving niftis", required=False, default=8,
                        type=int)
    parser.add_argument('-ht', '--high_threshold', help="high threshold for dual threshold iterative", required=False, default=0.7,
                        type=float)
    parser.add_argument('-lt', '--low_threshold', help="low threshold for dual threshold iterative", required=False, default=0.4,
                        type=float)
    
    parser.add_argument('--npz', action="store_true", required=False, help="stores npz and pkl")
    parser.add_argument('-chk', help='checkpoint name, default: model_final_checkpoint',
                        required=False,
                        default='model_final_checkpoint')
    parser.add_argument("-ei", "--experiment_id", required=True)
    args = parser.parse_args()
    threads = args.threads
    folder = args.folder
#     output_folder = args.output_folder
    npz = args.npz
    if folder == None:
        folder = join(network_training_output_dir,"3d_fullres/Task210_cerebral_vessel_cta/PHTransTrainer__nnUNetPlansv2.1/fold_0", args.experiment_id, args.chk)


    mutil_process_eval(folder, threads, high_threshold =args.high_threshold, low_threshold=args.low_threshold, override=True, store_npz=npz)


if __name__ == "__main__":
    main()
