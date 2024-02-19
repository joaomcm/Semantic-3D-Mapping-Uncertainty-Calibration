import os
import numpy as np
import open3d as o3d
import open3d.core as o3c
import time
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm.notebook import tqdm
import pickle
import pickle
from glob import glob
from copy import deepcopy
from scipy.stats import entropy
import time
import torch.nn as nn
import torch
from torchmetrics.functional import jaccard_index
from scene_definitions import get_larger_test_and_validation_scenes,get_smaller_balanced_validation_scenes,get_original_small_validation_scenes,get_smaller_test_scenes,get_classes
import json
import pdb

evaluation_start = 0
no_void =  True
results_dir = '/scratch/bbuq/jcorreiamarques/3d_calibration/Results'
def get_experiments_and_short_names():
    a = json.load(open('./experiments_and_short_names.json','r'))
    experiments = a['experiments']
    short_names = a['short_names']
    return experiments,short_names

def compute_mIoUs():
    accuracies = []
    experiments,short_names = get_experiments_and_short_names()
#     experiments = ["FineTuned Segformer unCalibrated","Naive Averaging", 
# "Histogram Integration","Vector Scaling Averaged"]
#     short_names =["Naive Bayesian","Naive Averaging", 
#  "Histogram Integration","Vector Scaling Averaged"]
    
    
    val_scenes,test_scenes = get_larger_test_and_validation_scenes()

    # selected_scenes = test_scenes
    selected_scenes = test_scenes

    COLORS = np.array([
        [0,0,0],[151,226,173],[174,198,232],[31,120,180],[255,188,120],[188,189,35],[140,86,74],[255,152,151],[213,39,40],[196,176,213],[148,103,188],[196,156,148],
        [23,190,208],[247,183,210],[218,219,141],[254,127,14],[227,119,194],[158,218,229],[43,160,45],[112,128,144],[82,83,163]
    ])


    pcds_template = '{}/{}/{}/*.pcd'
    labels_template = '{}/{}/{}/labels*.p'
    per_exp_IoUs = {}
    per_exp_absents = {}
    for experiment in experiments:
        print(experiment)
        IoUs = {}
        absents = {}
        totals_gt = []
        totals = []
        for scene in tqdm(selected_scenes,desc = '{} mIoUs'.format(experiment),position = 1):
            pcd_colors = []
            gt_pcd_file = '{}/sanity_checks/gt_pcd_{}.pcd'.format(results_dir,scene)
            gt_labels_file = '{}/sanity_checks/gt_labels_{}.p'.format(results_dir,scene)

            gt_pcd = o3d.io.read_point_cloud(gt_pcd_file)
            gt_labels= pickle.load(open(gt_labels_file,'rb'))
            pcd_files = sorted(glob(pcds_template.format(results_dir,experiment,scene)))
            label_files = sorted(glob(labels_template.format(results_dir,experiment,scene)))
            pcd_file = pcd_files[-1]
            labels_file = label_files[-1]
            pcd = o3d.io.read_point_cloud(pcd_file)
            pcd2 = o3d.io.read_point_cloud(pcd_file)
            labels= pickle.load(open(labels_file,'rb'))
            
#             if(np.any(labels<0)):
#                 labels = labels - labels.max(axis = 1).reshape((-1,1))
#                 exp_labels = np.exp(labels)
#                 labels = exp_labels/exp_labels.sum(axis=1).reshape((-1,1))
#             if(np.any(labels.sum(axis = 1) > 1)):
#                 labels = labels/labels.sum(axis = 1,keepdims = True)
#             labels = labels/labels.sum(axis = 1,keepdims = True)
            #making absolutely sure this sums to 1
#             labels = labels/labels.sum(axis = 1,keepdims = True)

            # we then unscramble the labels:

            pcd_tree = o3d.geometry.KDTreeFlann(gt_pcd)
            points = np.asarray(pcd.points)
            gt_points = np.asarray(gt_pcd.points)
            unscrambler = np.zeros(points.shape[0]).astype(np.int64)
            scrambler = np.zeros(points.shape[0]).astype(np.int64)
            for i in range(points.shape[0]):
                [k, idx, dist] = pcd_tree.search_knn_vector_3d(points[i],1)
            #     print(points[i]-gt_points[idx],i,idx)
                unscrambler[idx[0]] = i
                scrambler[i] = idx[0]
            diff_labels = labels[unscrambler]


            gt_labels = np.argmax(gt_labels,axis =1)
            diff_labels = np.argmax(diff_labels,axis =1)
            # print(diff_labels.shape)
            # print(gt_labels.shape)
            totals_gt.extend(gt_labels.tolist())
            totals.extend(diff_labels.tolist())
            unique_gt = np.unique(gt_labels).tolist()
            unique_pred = np.unique(diff_labels).tolist()
            absent = set(list(range(21))) - set(unique_gt + unique_pred)
            IoU = jaccard_index(task = 'multiclass',preds = torch.from_numpy(diff_labels),target= torch.from_numpy(gt_labels),num_classes = 21,average = None)
            IoUs.update({scene:IoU})
            absents.update({scene:absent})
            gt_labels_torch = torch.nn.functional.one_hot(torch.from_numpy(gt_labels.astype(np.int64)),num_classes = 21)
            diff_labels_torch = torch.nn.functional.one_hot(torch.from_numpy(diff_labels.astype(np.int64)),num_classes = 21)

        IoU = jaccard_index(task = 'multiclass',preds=torch.from_numpy(np.array(totals)),target = torch.from_numpy(np.array(totals_gt)),num_classes = 21,average = None)
        non_null = np.array(totals_gt) != 0
        accuracy = (np.array(totals)[non_null] == np.array(totals_gt)[non_null]).sum()/np.array(totals)[non_null].shape[0]
        accuracies.append(accuracy)
        absent = set(list(range(21))) - set(totals + totals_gt)
        IoUs.update({'aggregate':IoU})
        absents.update({'aggregate':absent})
        per_exp_IoUs.update({experiment:IoUs})
        per_exp_absents.update({experiment:absents})

    a = IoUs['aggregate'].numpy()
    a[a!=0].mean()

    dfs = []
    for experiment,short_name in zip(experiments,short_names):
        IoUs = per_exp_IoUs[experiment]
        absents = per_exp_absents[experiment]
        expanded_scenes = selected_scenes+['aggregate']
        metrics = np.zeros((len(expanded_scenes),21))
        for i,scene in enumerate(expanded_scenes):
        #     print(scene)
            metrics[i,:] = IoUs[scene]
            if(absents[scene]):
                metrics[i,np.array(list(absents[scene]))] = np.nan

        import pandas as pd 

        df = pd.DataFrame(metrics)
        classes = get_classes()
        df.columns = classes
        df.loc[:,'scene'] = expanded_scenes
        df.loc[:,'experiment'] = short_name
        dfs.append(df)
    # df.to_pickle('./Results/3D IoUs/Larger/{} 3D IoUs.p'.format(experiment))

    final_df = pd.concat(dfs).reset_index(drop = True)
    final_df.to_excel('{}/quant_eval/3D IoUs per experiment.xlsx'.format(results_dir))

    # final_df.to_excel('{}/Comparative Experiments 3D IoUs.xlsx'.format(results_dir))
    selected_df = final_df.loc[final_df.scene == 'aggregate',:].reset_index(drop = True)
    df = selected_df.transpose()
    df.rename(columns = df.loc['experiment',:],inplace = True)
    df.drop('experiment',inplace = True)
    df.drop('scene',inplace = True)
    df.drop('irrelevant',inplace = True)
    df.loc['mIoU',:] = df.mean(axis = 0)
    df.loc['Accuracy'] = accuracies
    selected_df = df.reset_index(drop = False)
    transposed_df = selected_df.transpose()
    
    transposed_df.to_excel('{}/quant_eval/3D IoUs.xlsx'.format(results_dir))


import os
import numpy as np
import open3d as o3d
import open3d.core as o3c
import time
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm.notebook import tqdm
import pickle
from glob import glob
from copy import deepcopy
from scipy.stats import entropy
import time
import torch.nn as nn
import torch
import seaborn as sns
import pdb
import torch
from my_calibration import Calibration_calc_3D,mECE_Calibration_calc_3D_fix,mECE_Calibration_calc_3D
from ESANet_loader import ESANetClassifier, TSegmenter
from sklearn.metrics import confusion_matrix
from scene_definitions import get_larger_test_and_validation_scenes,get_smaller_balanced_validation_scenes,get_original_small_validation_scenes,get_smaller_test_scenes,get_classes
import pickle
import json


def compute_mECEs():
    
    experiments,short_names = get_experiments_and_short_names()

    # true_cm = pickle.load(open('3D_confusion_matrix.p','rb'))
    # pred_cm = pickle.load(open('3D_confusion_matrix_pred_normalized.p','rb'))
    classes = get_classes()

    # experiment = 'Segformer Direct 3D unCalibrated Large Raw Logits'
    cal_scenes,test_scenes = get_larger_test_and_validation_scenes()
    selected_scenes = test_scenes




    # experiments = ['FineTuned Segformer Calibrated',
    #                'Segformer Direct 3D Vector Scaled Large - Further Opt']
    # short_names = ['3D temp scaled','Vector Scaling Extra']


    ECE_by_experiment = []
    all_cals = {}
    per_scene_mECEs = []
    for g,experiment in enumerate(experiments):
        multiply = False
        mECE_cal = mECE_Calibration_calc_3D(no_void = no_void)
        TL_ECE_cal = mECE_Calibration_calc_3D_fix(no_void = no_void)
        # cc_3d = Calibration_calc_3D(no_void = True)
        pcds_template = '{}/{}/{}/*.pcd'
        labels_template = '{}/{}/{}/labels*.p'
        splt = experiment.split('|')
        difs = []
        gts = []
        preds = []
        print('\n\n experiment = {} \n\n\n '.format(experiment))
        per_scene_mECE = []
        for scene in tqdm(selected_scenes,desc = 'ECEs',position = 2):
            # per_scene_cal =  mECE_Calibration_calc_3D(no_void = no_void)
            # per_scene_tl_ECE = mECE_Calibration_calc_3D_fix(no_void = no_void)
            gt_pcd_file = '{}/sanity_checks/gt_pcd_{}.pcd'.format(results_dir,scene)
            gt_labels_file = '{}/sanity_checks/gt_labels_{}.p'.format(results_dir,scene)
            gt_pcd = o3d.io.read_point_cloud(gt_pcd_file)
            gt_labels= pickle.load(open(gt_labels_file,'rb'))
            pcd_file = sorted(glob(pcds_template.format(results_dir,experiment,scene)))[-1]
            labels_file = sorted(glob(labels_template.format(results_dir,experiment,scene)))[-1]
        #     for pcd_file,labels_file in zip(pcd_files,label_files):
            pcd = o3d.io.read_point_cloud(pcd_file)
            labels= pickle.load(open(labels_file,'rb')).astype(np.float64)
#             if(experiment == 'Naive Averaging'):
#                 pdb.set_trace()
        #     label_logits = labels
            if(np.any(labels<0)):
                labels = labels - labels.max(axis = 1).reshape((-1,1))
                exp_labels = np.exp(labels)
                labels = exp_labels/exp_labels.sum(axis=1).reshape((-1,1))
            if(np.any(labels.sum(axis = 1) > 1)):
                labels = labels/labels.sum(axis = 1,keepdims = True)
            labels = labels/labels.sum(axis = 1,keepdims = True)

    #         if(multiply):
    #             if(splt[1] == 'Post Adjustment'):
    #                 labels = np.matmul(labels,true_cm.transpose())
    #             else:
    #                 labels = np.matmul(labels,pred_cm)
    #             labels = labels/labels.sum(axis = 1).reshape((-1,1))
            pcd_tree = o3d.geometry.KDTreeFlann(gt_pcd)
            points = np.asarray(pcd.points)
            gt_points = np.asarray(gt_pcd.points)
            unscrambler = np.zeros(points.shape[0]).astype(np.int64)
            scrambler = np.zeros(points.shape[0]).astype(np.int64)
            for i in range(points.shape[0]):
                [k, idx, dist] = pcd_tree.search_knn_vector_3d(points[i],1)
            #     print(points[i]-gt_points[idx],i,idx)
                scrambler[i] = idx[0]
            labels[labels.sum(axis =1)==0] = 1/21.0
            this_stage_labels = deepcopy(gt_labels)
            this_stage_labels[:] = 1/21.0
            this_stage_labels[scrambler] = labels
            map_label = this_stage_labels.argmax(axis = 1)
            map_gt = gt_labels.argmax(axis = 1)
            gts.extend(map_gt)
            preds.extend(map_label)

            # update all_bins
            mECE_cal.update_bins(this_stage_labels,gt_labels)
            # per_scene_cal.update_bins(this_stage_labels,gt_labels)
            TL_ECE_cal.update_bins(this_stage_labels,gt_labels)
            # per_scene_tl_ECE.update_bins(this_stage_labels,gt_labels)
            # local_ECEs = per_scene_cal.get_ECEs()
            # local_ECEs.append(per_scene_tl_ECE.get_TL_ECE())
            # per_scene_mECE.append(local_ECEs)

    #         agg_cal.update_bins(this_stage_labels,gt_labels)
    #         for i in range(21):
    #             if(map_gt[map_gt == i].shape[0]>0):
    #                 cals[i].update_bins(this_stage_labels[map_gt == i],gt_labels[map_gt == i])
    #     ECEs = []
    #     for i in range(0,21):
    #         ECEs.append(cals[i].get_ECE())
        # per_scene_mECEs.append(per_scene_mECE)
        ECEs =  mECE_cal.get_ECEs()
        ECEs.append(TL_ECE_cal.get_TL_ECE())
        # all_cals.update({experiments[g]:mECE_cal})
        ECE_by_experiment.append(ECEs)
    #     pickle.dump(mECE_cal,open('3D calibration Results.p','wb'))
    
    import pandas as pd
    dfs = []
    mtx = np.array(ECE_by_experiment)
    # for l in ECE_by_experiment:
    #     df = pd.DataFrame(l)
    df = pd.DataFrame(mtx)
    classes = ['null','wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture',
               'counter','desk','curtain','refrigerator','shower curtain','toilet','sink','bathtub','otherfurniture','aggregate','TL-ECE']
    df.columns = classes
    df.loc[:,'experiments'] = short_names
    df = df.loc[:,[df.columns[-1]]+df.columns[:-1].tolist()]
    df.loc[:,'mECE - things'] = df.loc[:,df.columns[4:-2]].mean(axis = 1)
    if(no_void):
        df.loc[:,'mECE -all'] = df.loc[:,df.columns[2:-3]].mean(axis = 1)
        df.loc[:,'mECE - stuff'] = df.loc[:,df.columns[2:4]].mean(axis = 1)
    else:
        df.loc[:,'mECE -all'] = df.loc[:,df.columns[1:-3]].mean(axis = 1)
        df.loc[:,'mECE - stuff'] = df.loc[:,df.columns[1:4]].mean(axis = 1)
    cols = list(df.columns)
    cols2 = ['aggregate','mECE - things', 'mECE -all', 'mECE - stuff','TL-ECE']
    cols[-5:] = cols2
    df = df.loc[:,cols]
    df.to_excel('{}/quant_eval/ECEs by class and experiment_alt_finetuned.xlsx'.format(results_dir),index = False)
    
    
    
    # import seaborn as sns
    # sns.set_theme()
    # for i,c in tqdm(enumerate(classes[:21])):
    #     fig,axis = plt.subplots(1,len(experiments))

    #     for experiment,short_name in tqdm(zip(experiments,short_names)):
    #         ax = axis[experiments.index(experiment)]
    #         cc = all_cals[experiment].cals[i]
    #         cal,conf,lims = cc.return_calibration_results()
    #         lims = lims-0.05
    #         cal = cc.correct_bin_members/cc.total_bin_members
    #         cal = np.nan_to_num(cal,0)
    # #         fig, ax = plt.subplots()
    #         fig.set_size_inches(len(experiment)*8,8)
    #         p1 = ax.bar(x= lims, height = cal,  width = 0.8*(lims[1]-lims[0]),color = 'b',alpha = 0.5)
    #         ax.bar(x= lims, height = conf, width = 0.8*(lims[1]-lims[0]),color = 'r',alpha = 0.2)
    #         ax.plot(np.arange(11)/10,np.arange(11)/10)
    #         membership = (cc.total_bin_members/cc.total_bin_members.sum()*100)
    #         membership = np.round(membership,decimals = 1)
    #         membership = ['(' +str(i) + '%)' for i in membership]

    #         # membership = eval(np.array_str(membership, precision=2, suppress_small=True))
    #         ax.bar_label(p1, labels = membership,label_type='edge')
    #         ax.set_title('{} - {} - ECE = {:.3f}'.format(short_name,c,cc.get_ECE()))
    #         ax.set_xlabel('Upper Confidence')
    #         ax.set_ylabel('Empirical Accuracy within bin (total pixel %)')
    #     plt.savefig('{}/mECE Analysis/plots/Segformer3/Rel_diagrams_by_experiment_{}.png'.format(results_dir,c),bbox_inches = 'tight')
    #     plt.show()

import os
import numpy as np
import open3d as o3d
import open3d.core as o3c
import time
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm.notebook import tqdm
import pickle
import pickle
from glob import glob
from copy import deepcopy
from scipy.stats import entropy
import time
import torch.nn as nn
import torch
from torchmetrics.functional import jaccard_index
from my_calibration import Calibration_calc_3D, mECE_Calibration_calc_3D,BrierScore3D
from scene_definitions import get_larger_test_and_validation_scenes
import json
import pdb

# experiment = 'Segformer Direct 3D unCalibrated Large Raw Logits'
def compute_brier_scores():
    cal_scenes,test_scenes = get_larger_test_and_validation_scenes()
    selected_scenes = test_scenes

    experiments,short_names = get_experiments_and_short_names()
    # experiments = ['FineTuned Segformer Calibrated',
    #                'Segformer Direct 3D Vector Scaled Large - Further Opt']
    # short_names = ['3D temp scaled','Vector Scaling Extra']

    metric = BrierScore3D
    ECE_by_experiment = []
    all_cals = {}
    per_scene_mECEs = []
    for g,experiment in enumerate(experiments):
        multiply = False
        mECE_cal = metric(no_void = no_void)

        # cc_3d = Calibration_calc_3D(no_void = True)
        pcds_template = '{}/{}/{}/*.pcd'
        labels_template = '{}/{}/{}/labels*.p'
        splt = experiment.split('|')
        difs = []
        gts = []
        preds = []
        print('\n experiment = {} \n '.format(experiment))
        per_scene_mECE = []
        for scene in selected_scenes:
            per_scene_cal =  metric(no_void = no_void)

            gt_pcd_file = '{}/sanity_checks/gt_pcd_{}.pcd'.format(results_dir,scene)
            gt_labels_file = '{}/sanity_checks/gt_labels_{}.p'.format(results_dir,scene)
            gt_pcd = o3d.io.read_point_cloud(gt_pcd_file)
            gt_labels= pickle.load(open(gt_labels_file,'rb'))
            pcd_file = sorted(glob(pcds_template.format(results_dir,experiment,scene)))[-1]
            labels_file = sorted(glob(labels_template.format(results_dir,experiment,scene)))[-1]
        #     for pcd_file,labels_file in zip(pcd_files,label_files):
            pcd = o3d.io.read_point_cloud(pcd_file)
            labels= pickle.load(open(labels_file,'rb'))
    #         print(np.unique(labels))
        #     label_logits = labels
            if(np.any(labels<0)):
                labels = labels - labels.max(axis = 1).reshape((-1,1))
                exp_labels = np.exp(labels)
                labels = exp_labels/exp_labels.sum(axis=1).reshape((-1,1))
            if(np.any(labels > 1)):
                labels = labels/labels.sum(axis = 1,keepdims = True)
            labels = labels/labels.sum(axis = 1,keepdims = True)
            labels[np.isnan(labels)] = 1/21
#             pdb.set_trace()
    #         print(np.sum(np.isnan(labels))/labels.shape[0],scene)
    #         if(multiply):
    #             if(splt[1] == 'Post Adjustment'):
    #                 labels = np.matmul(labels,true_cm.transpose())
    #             else:
    #                 labels = np.matmul(labels,pred_cm)
    #             labels = labels/labels.sum(axis = 1).reshape((-1,1))
            pcd_tree = o3d.geometry.KDTreeFlann(gt_pcd)
            points = np.asarray(pcd.points)
            gt_points = np.asarray(gt_pcd.points)
            unscrambler = np.zeros(points.shape[0]).astype(np.int64)
            scrambler = np.zeros(points.shape[0]).astype(np.int64)
            for i in range(points.shape[0]):
                [k, idx, dist] = pcd_tree.search_knn_vector_3d(points[i],1)
            #     print(points[i]-gt_points[idx],i,idx)
                scrambler[i] = idx[0]
            labels[labels.sum(axis =1)==0] = 1/21.0
            this_stage_labels = deepcopy(gt_labels)
            this_stage_labels[:] = 1/21.0
            this_stage_labels[scrambler] = labels
            map_label = this_stage_labels.argmax(axis = 1)
            map_gt = gt_labels.argmax(axis = 1)

            gts.extend(map_gt)
            preds.extend(map_label)
            mECE_cal.update_bins(this_stage_labels,map_gt)
            per_scene_cal.update_bins(this_stage_labels,map_gt)
            per_scene_mECE.append(per_scene_cal.return_score())

    #         agg_cal.update_bins(this_stage_labels,gt_labels)
    #         for i in range(21):
    #             if(map_gt[map_gt == i].shape[0]>0):
    #                 cals[i].update_bins(this_stage_labels[map_gt == i],gt_labels[map_gt == i])
    #     ECEs = []
    #     for i in range(0,21):
    #         ECEs.append(cals[i].get_ECE())
        per_scene_mECEs.append(per_scene_mECE)
        all_cals.update({experiments[g]:mECE_cal})
        ECEs = mECE_cal.return_score()
        ECE_by_experiment.append(ECEs)

    import pandas as pd
    dfs = []
    mtx = np.array(ECE_by_experiment)
    # for l in ECE_by_experiment:
    #     df = pd.DataFrame(l)
    df = pd.DataFrame(mtx)
    df.columns = ['Brier']
    df.loc[:,'experiments'] = short_names
    df = df.loc[:,[df.columns[-1]]+df.columns[:-1].tolist()]
    df = df.transpose()
    df.columns = df.loc['experiments',:]
    df = df.drop('experiments')
    df = df.reset_index(drop = False)
#     pdb.set_trace()
    df.to_excel('{}/quant_eval/brier_scores.xlsx'.format(results_dir),index = False)


from multiprocessing import Process

p1 = Process(target = compute_mIoUs)
p2 = Process(target = compute_mECEs)
p3 = Process(target = compute_brier_scores)

p1.start()
p2.start()
p3.start()
p1.join()
p2.join()
p3.join()

# compute_mIoUs()
# compute_mECEs()
# compute_brier_scores()
