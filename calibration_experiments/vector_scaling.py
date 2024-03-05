# from agents.utils.semantic_prediction import SemanticPredMaskRCNN
import argparse
import json

# from multiprocessing import Pool
import multiprocessing
import os
import pdb
from functools import partial
from glob import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.my_calibration import (
    mECE_Calibration_calc_3D_fix as mECE_Calibration_calc_3D,
)

# from scene_definitions import get_larger_test_and_validation_scenes,get_smaller_balanced_validation_scenes,get_original_small_validation_scenes,get_smaller_test_scenes
from utils.scene_definitions import get_larger_test_and_validation_scenes,get_fixed_train_and_val_splits


def reconstruct_scene(scene,T,rec_type = 'Naive Bayesian',segmentation_model = 'ESANET'):
    assert rec_type in ['Naive Bayesian','Naive Averaging','Histogram','Geometric Mean']
    this_T = torch.Tensor(T).to('cuda:0').view(1,21)
    sm = nn.Softmax(dim = 1)
    # filename = '/scratch/bbuq/jcorreiamarques/3d_calibration/h5py_datasets/calibration_validation_logits_lzf.hdf5'
    if(segmentation_model == 'ESANET'):
        filename = '/tmp/ESANET_calibration_validation_logits_lzf.hdf5'
    elif(segmentation_model == 'Segformer'):
        filename = '/tmp/calibration_validation_logits_lzf.hdf5'

    f =  h5py.File(filename,'r')
    g = f[scene]
    logits = g['logits']
    indices = g['indices']
    poses = g['poses']
    triangle_gts = g['triangle_gts']
    max_triangle_index = g.attrs['max_triangle_index']+1
    reconstruction = torch.from_numpy(np.zeros((max_triangle_index,21))).to('cuda:0')
    weights = torch.from_numpy(np.zeros((max_triangle_index))).to('cuda:0')
    for i in range(indices.shape[0]):
        logit = torch.from_numpy(logits[i,:,:,:].reshape(-1,21)).to('cuda:0').float()/this_T
        index = torch.from_numpy(indices[i,:,:].reshape(-1).astype(np.int64)).to('cuda:0')
        index[index > max_triangle_index] = 0
        probs = sm(logit)
        if(rec_type in ['Naive Bayesian','Geometric Mean']):
            probs += 0.001
            probs = probs/probs.sum(axis = 1,keepdims = True)
            probs = torch.log(probs).float()
            reconstruction[index] += probs
        elif(rec_type == 'Naive Averaging'):
            reconstruction[index] += probs
        elif(rec_type == 'Histogram'):
            pred = probs.argmax(axis =1)
            reconstruction[index,pred] += 1
        weights[index] += 1
    if(rec_type == 'Naive Bayesian'):
        reconstruction = sm(reconstruction)
    elif(rec_type == 'Geometric Mean'):
        reconstruction = reconstruction/weights.view(-1,1)
        reconstruction = sm(reconstruction)
    elif(rec_type in ['Naive Averaging','Histogram']):
        reconstruction = reconstruction/weights.view(-1,1)

    return reconstruction.cpu().numpy(),triangle_gts[:,:]


class Calibrator:
    def __init__(self,selected_scenes,rec_type,segmentation_model):
        self.selected_scenes = selected_scenes
        self.p = multiprocessing.get_context('forkserver').Pool(processes = 16)
        self.rec_type = rec_type
        self.segmentation_model = segmentation_model
    def eval_calibration(self,T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20):

        res = []

        T = [T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20]
        T = np.exp(np.array(T)).tolist()

        for a in tqdm(self.p.imap_unordered(partial(reconstruct_scene,T = T,rec_type = self.rec_type,segmentation_model = self.segmentation_model),self.selected_scenes), total= len(self.selected_scenes)):
            res.append(a) 
        cc_3d = mECE_Calibration_calc_3D(no_void = True,one_hot= False)
        for item in res:
            pred,gt = item
            cc_3d.update_bins(pred,gt.reshape(-1))
        calib_result = -cc_3d.get_mECE()
        print("\n\nthis is the calibration with T = {} | {}\n\n".format(T,calib_result))
        return calib_result
    

def main():
    import numpy as np
    from bayes_opt import (
        BayesianOptimization,
        SequentialDomainReductionTransformer,
        UtilityFunction,
    )
    from bayes_opt.event import Events
    from bayes_opt.logger import JSONLogger
    from bayes_opt.util import load_logs
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings_file", help="the name of the settings file to load that fully specifies this experiment")
    args = parser.parse_args()
    settings_file_name = args.settings_file
    experiment_settings = json.load(open('./calibration_experiments/{}.json'.format(settings_file_name),'rb'))
    # cal_scenes,test_scenes = get_larger_test_and_validation_scenes()
    # cal_scenes = get_smaller_balanced_validation_scenes()
    train_scenes,cal_scenes = get_fixed_train_and_val_splits()
    exp_name = experiment_settings['experiment_name']
    max_bounds = np.log(float(experiment_settings['max_bounds']))
    min_bounds = np.log(float(experiment_settings['min_bounds']))
    rec_type = experiment_settings['rec_type']
    segmentation_model = experiment_settings['segmentation_model']
    mycal = Calibrator(cal_scenes,rec_type,segmentation_model)
    pbounds = {}
    for i in range(0,21):
        pbounds.update({'T{}'.format(i):(min_bounds,max_bounds)})

    bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.5)
    # print(pbounds)
    # pbounds = {'T': (1, 500)}
    optimizer =  BayesianOptimization(
        f=mycal.eval_calibration,
        pbounds=pbounds,
        random_state=np.random.randint(1,100000),
        allow_duplicate_points = False
    )
    from glob import glob
    try:
        os.makedirs('./scaling_results/{}/'.format(exp_name))
    except Exception as e:
        print(e)
        pass
        
    attempts = glob("./scaling_results/{}/*.json".format(exp_name))
    logger = JSONLogger(path="./scaling_results/{}/optimization_logs.json".format(exp_name),reset = False)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    has_logs = False
    if(len(attempts)> 0):
        has_logs = True
        load_logs(optimizer, logs=["./scaling_results/{}/optimization_logs.json".format(exp_name)])
    if(len(attempts)==0):
        #always start exploring the neutral calibration, max and min points
        # initial_points = {}
        # for i in range(0,21):
        #     initial_points.update({'T{}'.format(i):np.log(1)})
        # output = mycal.eval_calibration(**initial_points)
        # optimizer.register(params = initial_points,target = output)        
        initial_points = {} 
        initial_points_range = np.linspace(min_bounds,max_bounds,num = 50,endpoint = True)
        for temp in initial_points_range:
            for i in range(0,21):
                initial_points.update({'T{}'.format(i):temp})
            output = mycal.eval_calibration(**initial_points)
            optimizer.register(params = initial_points,target = output)


    acquisition_function = UtilityFunction(kind="ucb", kappa=3)
    init_points = 20
    if(has_logs):
        init_points = 0
    optimizer.maximize(init_points = init_points,n_iter = 1000,verbose = 2,acquisition_function = acquisition_function)

    print(optimizer.max)
if __name__ == '__main__':
    main()
# results = Parallel(backend = '',n_jobs =5,verbose = 22)(delayed(reconstruct_scene)(scene) for scene in scenes )