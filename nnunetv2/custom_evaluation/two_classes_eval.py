# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:59:48 2022
@author: 12593
"""
import argparse
import numpy as np
import nibabel as nb
import os
from collections import OrderedDict
import pandas as pd
from nnunetv2.paths import *
from nnunetv2.custom_evaluation.SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from batchgenerators.utilities.file_and_folder_operations import *

def eval(seg_path,gt_path,experiment_id,task_id):
   
    # seg_path = join(nnUNet_results,f"3d_fullres/Task230_WORD/PHTransTrainer__nnUNetPlansv2.1/fold_0/{experiment_id}/model_final_checkpoint_No_DDA")
    # gt_path = join(nnUNet_raw,"Task180_WORD/labelsTs")
    # save_path = '/home/admin/nvme8t/code/PHTransV2/PHTrans/evaluation_results/WORD'
    
    save_path =  join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),f"evaluation_results/{task_id}")
    makedirs(save_path)

    save_name = experiment_id+".cvs"
    filenames = os.listdir(seg_path)
    filenames = [x for x in filenames if x.endswith('.nii.gz')]
    filenames.sort()

    seg_metrics = OrderedDict()
    seg_metrics['Name'] = list()
    seg_metrics['DSC'] = list()
    seg_metrics['NSD'] = list()

    for name in filenames:
        seg_metrics['Name'].append(name)
        # load grond truth and segmentation
        gt_nii = nb.load(join(gt_path, name))
        case_spacing = gt_nii.header.get_zooms()
        gt_data = np.uint8(gt_nii.get_fdata())
        seg_data = np.uint8(nb.load(join(seg_path, name)).get_fdata())

        
        if np.sum(gt_data==1)==0 and np.sum(seg_data==1)==0:
            DSC = 1
            NSD = 1
        elif np.sum(gt_data==1)==0 and np.sum(seg_data==1)>0:
            DSC = 0
            NSD = 0
        else:
            gt, seg = gt_data==1, seg_data==1
            surface_distances = compute_surface_distances(gt, seg, case_spacing)
            DSC = compute_dice_coefficient(gt, seg)
            # NSD_i = compute_surface_dice_at_tolerance(surface_distances, label_tolerance[organ])
            NSD = compute_surface_dice_at_tolerance(surface_distances, 1)
        print(f'{name}_DSC:',round(DSC, 4))
        print(f'{name}_NSD:',round(NSD, 4))

    
        seg_metrics['DSC'].append(round(DSC, 4))
        seg_metrics['NSD'].append(round(NSD, 4))

    seg_metrics['Name'].append(f"{experiment_id}")

    seg_metrics['DSC'].append(round(np.mean(seg_metrics['DSC']), 4))
    seg_metrics['NSD'].append(round(np.mean(seg_metrics['NSD']), 4))
    print('Mean_DSC:',seg_metrics['DSC'][-1])
    print('Mean_NSD:',seg_metrics['NSD'][-1])

    dataframe = pd.DataFrame(seg_metrics)
    dataframe.to_csv(join(save_path, save_name), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("PHTrans_eval")
    parser.add_argument("-sp","--seg_path", help='tag of experiment id')
    parser.add_argument("-gp", "--gt_path", help='tag of experiment id')
    parser.add_argument("-ei","--experiment_id", help='tag of experiment id')
    parser.add_argument("-t","--task", help="task, only support BCV and WORD")
    args = parser.parse_args()
   

    eval(args.seg_path,args.gt_path,args.experiment_id,args.task)
