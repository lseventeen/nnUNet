from phtrans.evaluation.SurfaceDice import compute_dice_coefficient
from phtrans.evaluation.cldice import clDice
import argparse
import os 
import numpy as np
import nibabel as nb
from nnunet.paths import *
from collections import OrderedDict
import pandas as pd
def online_eval(task_id,experiment_id):
  
   
    join = os.path.join
    if task_id == 200:
        task = "Task200_shiny_icarus"
        save_path = '/home/admin/nvme8t/code/PHTransV2/PHTrans/evaluation_results/isbi2023_SI'
    elif task_id == 201:
        task = "Task201_smile_uhura"
        save_path = '/home/admin/nvme8t/code/PHTransV2/PHTrans/evaluation_results/isbi2023_SU'
    else:
        raise NotImplementedError(f"Unkown task: {task_id}")
    seg_path = join(network_training_output_dir,f"3d_fullres/{task}/PHTransTrainer__nnUNetPlansv2.1/fold_0/{experiment_id}/model_final_checkpoint")
    gt_path = join(nnUNet_raw_data,f"{task}/labelsTs")
    vessel_seg_eval(experiment_id,seg_path,gt_path,save_path)

def vessel_seg_eval(experiment_id,seg_path,gt_path,save_path):
    save_name = experiment_id+".cvs"
    filenames = os.listdir(seg_path)
    filenames = [x for x in filenames if x.endswith('.nii.gz')]
    filenames.sort()
    dice_all = []
    cldice_all = []
    seg_metrics = OrderedDict()
    seg_metrics['Case'] = list()
    seg_metrics['Dice'] = list()
    seg_metrics['clDice'] = list()

    for name in filenames:
        gt_data = np.uint8(nb.load(join(gt_path, name)).get_fdata())
        seg_data = np.uint8(nb.load(join(seg_path, name)).get_fdata())
        dice = compute_dice_coefficient(gt_data, seg_data)
        cldice = clDice(gt_data, seg_data)
        dice_all.append(dice)
        cldice_all.append(cldice)
        seg_metrics['Case'].append(name)
        seg_metrics['Dice'].append(round(dice, 4))
        seg_metrics['clDice'].append(round(cldice, 4))
        print(name, "Done")
    seg_metrics['Case'].append(experiment_id)
    seg_metrics['Dice'].append(round(np.array(dice_all).mean(), 4))
    seg_metrics['clDice'].append(round(np.array(cldice_all).mean(), 4))
    
    print(round(np.array(dice_all).mean(), 4))
    print(round(np.array(cldice_all).mean(), 4))
    dataframe = pd.DataFrame(seg_metrics)
    dataframe.to_csv(join(save_path, save_name), index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("PHTrans_eval")
    parser.add_argument("--ei", help='tag of experiment id',default="nnunet_230103_081238")
    parser.add_argument("--id", help='task id', type=int, default="200")
    args = parser.parse_args()
    online_eval(args.id,args.ei)

    



