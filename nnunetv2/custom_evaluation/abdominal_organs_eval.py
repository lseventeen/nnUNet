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
from nnunet.paths import *
from phtrans.evaluation.SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient

def find_lower_upper_zbound(organ_mask):
    """
    Parameters
    ----------
    seg : TYPE
        DESCRIPTION.
    Returns
    -------
    z_lower: lower bound in z axis: int
    z_upper: upper bound in z axis: int
    """
    organ_mask = np.uint8(organ_mask)
    assert np.max(organ_mask) ==1, print('mask label error!')
    z_index = np.where(organ_mask>0)[2]
    z_lower = np.min(z_index)
    z_upper = np.max(z_index)
    
    return z_lower, z_upper


def abdominal_organs_eval(task,experiment_id):
   

    join = os.path.join


    if task == 180:
        label_tolerance = OrderedDict({
        "liver": 1,
        "spleen":1,
        "left_kidney":1,
        "right_kidney":1,
        "stomach":1,
        "gallbladder":1,
        "esophagus":1,
        "pancreas":1,
        "duodenum":1,
        "colon":1,
        "intestine":1,
        "adrenal":1,
        "rectum":1,
        "bladder":1,
        "Head_of_femur_L":1,
        "Head_of_femur_R":1
        })
    
        seg_path = join(network_training_output_dir,f"3d_fullres/Task180_WORD/PHTransTrainer__nnUNetPlansv2.1/fold_0/{experiment_id}/model_final_checkpoint_No_DDA")
        gt_path = join(nnUNet_raw_data,"Task180_WORD/labelsTs")
        # save_path = '/home/admin/nvme8t/code/PHTransV2/PHTrans/evaluation_results/WORD'
        save_path =  join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),"evaluation_results/WORD")
    elif task == 171:
        label_tolerance = OrderedDict({
        "spleen": 1, 
        "right kidney": 1, 
        "left kidney": 1, 
        "gall bladder": 1, 
        "esophagus": 1, 
        "liver": 1, 
        "stomach": 1, 
        "arota": 1, 
        "postcava": 1, 
        "pancreas": 1, 
        "right adrenal gland": 1, 
        "left adrenal gland": 1, 
        "duodenum": 1, 
        "bladder": 1, 
        "prostate/uterus" :1
        })

        seg_path = join(network_training_output_dir,f"3d_fullres/Task171_AMOS22st1/PHTransTrainer__nnUNetPlansv2.1/fold_0/{experiment_id}/model_final_checkpoint_No_DDA")
        gt_path = join(nnUNet_raw_data,"Task171_AMOS22st1/labelsTs")
        # save_path = '/home/admin/nvme8t/code/PHTransV2/PHTrans/evaluation_results/AMOS'
        save_path =  join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),"evaluation_results/AMOS")
    else:
        raise NotImplementedError(f"Unkown task: {task}")

    save_name = experiment_id+".cvs"
    filenames = os.listdir(seg_path)
    filenames = [x for x in filenames if x.endswith('.nii.gz')]
    filenames.sort()

    seg_metrics = OrderedDict()
    seg_metrics['Name'] = list()



    for organ in label_tolerance.keys():
        seg_metrics['{}_DSC'.format(organ)] = list()
    seg_metrics['Mean_DSC'] = list()
    for organ in label_tolerance.keys():
        seg_metrics['{}_NSD'.format(organ)] = list()
    seg_metrics['Mean_NSD'] = list()




    for name in filenames:
        seg_metrics['Name'].append(name)
        # load grond truth and segmentation
        gt_nii = nb.load(join(gt_path, name))
        case_spacing = gt_nii.header.get_zooms()
        gt_data = np.uint8(gt_nii.get_fdata())
        seg_data = np.uint8(nb.load(join(seg_path, name)).get_fdata())
        mean_DSC = 0
        mean_NSD = 0
        for i, organ in enumerate(label_tolerance.keys(),1):
            if np.sum(gt_data==i)==0 and np.sum(seg_data==i)==0:
                DSC_i = 1
                NSD_i = 1
            elif np.sum(gt_data==i)==0 and np.sum(seg_data==i)>0:
                DSC_i = 0
                NSD_i = 0
        # else:
        #     if i==5 or i==6 or i==10: # for Aorta, IVC, and Esophagus, only evaluate the labelled slices in ground truth
        #         z_lower, z_upper = find_lower_upper_zbound(gt_data==i)
        #         organ_i_gt, organ_i_seg = gt_data[:,:,z_lower:z_upper]==i, seg_data[:,:,z_lower:z_upper]==i
            else:
                organ_i_gt, organ_i_seg = gt_data==i, seg_data==i
                surface_distances = compute_surface_distances(organ_i_gt, organ_i_seg, case_spacing)
                DSC_i = compute_dice_coefficient(organ_i_gt, organ_i_seg)
                # NSD_i = compute_surface_dice_at_tolerance(surface_distances, label_tolerance[organ])
                NSD_i = compute_surface_dice_at_tolerance(surface_distances, 1)
            seg_metrics['{}_DSC'.format(organ)].append(round(DSC_i, 4))
            seg_metrics['{}_NSD'.format(organ)].append(round(NSD_i, 4))  
            # print(name, organ, round(DSC_i,4), 'tol:', label_tolerance[organ], round(NSD_i,4))
            print(name, organ, round(DSC_i,4), 'tol:', 1, round(NSD_i,4))
            mean_DSC += DSC_i
            mean_NSD += NSD_i
        seg_metrics['Mean_DSC'].append(round(mean_DSC/len(label_tolerance), 4))
        seg_metrics['Mean_NSD'].append(round(mean_NSD/len(label_tolerance), 4))

    seg_metrics['Name'].append(f"{experiment_id}")
    for organ in label_tolerance.keys():
        seg_metrics['{}_DSC'.format(organ)].append(round(np.mean(seg_metrics['{}_DSC'.format(organ)]), 4))
        seg_metrics['{}_NSD'.format(organ)].append(round(np.mean(seg_metrics['{}_NSD'.format(organ)]), 4))
    seg_metrics['Mean_DSC'].append(round(np.mean(seg_metrics['Mean_DSC']), 4))
    seg_metrics['Mean_NSD'].append(round(np.mean(seg_metrics['Mean_NSD']), 4))
    print('Mean_DSC:',round(np.mean(seg_metrics['Mean_DSC']), 4))
    print('Mean_NSD:',round(np.mean(seg_metrics['Mean_NSD']), 4))

    dataframe = pd.DataFrame(seg_metrics)
    dataframe.to_csv(join(save_path, save_name), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("PHTrans_eval")
    parser.add_argument("--ei", help='tag of experiment id',default="PHTrans_Conv_P_221223_053833")
    parser.add_argument("-t","--task", help='task, only support BCV and WORD ',type=int, default=180)
    args = parser.parse_args()
   

    abdominal_organs_eval(args.task,args.ei)
