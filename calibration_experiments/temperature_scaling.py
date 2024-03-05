# from agents.utils.semantic_prediction import SemanticPredMaskRCNN
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import pdb
from glob import glob
# from scene_definitions import get_larger_test_and_validation_scenes,get_smaller_balanced_validation_scenes,get_original_small_validation_scenes,get_smaller_test_scenes
from scene_definitions import get_larger_test_and_validation_scenes,get_fixed_train_and_val_splits
import h5py
import torch
import torch.nn as nn
from functools import partial
# from multiprocessing import Pool
import multiprocessing
# from my_calibration import Calibration_calc_3D
from my_calibration import mECE_Calibration_calc_3D
import json
import argparse

def reconstruct_scene(scene,T,rec_type,segmentation_model):
    assert rec_type in ['Naive Bayesian','Naive Averaging','Histogram','Geometric Mean'],"Reconstruction type must be in ['Naive Bayesian','Naive Averaging','Histogram','Geometric Mean']"
    this_T = torch.Tensor(T).to('cuda:0').view(1,-1)
    sm = nn.Softmax(dim = 1)
    # filename = '/scratch/bbuq/jcorreiamarques/3d_calibration/h5py_datasets/ESANET_calibration_validation_logits_lzf.hdf5'
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
    reconstruction = torch.from_numpy(np.zeros((max_triangle_index,21))).float().to('cuda:0')
    weights = torch.from_numpy(np.zeros((max_triangle_index))).long().to('cuda:0')
    for i in range(indices.shape[0]):
        logit = torch.from_numpy(logits[i,:,:,:].reshape(-1,21)).to('cuda:0').float()/this_T
        index = torch.from_numpy(indices[i,:,:].reshape(-1).astype(np.int64)).to('cuda:0')
        index[index > max_triangle_index] = 0
        probs = sm(logit)
        if(rec_type in ['Naive Bayesian','Geometric Mean']):
            probs += 0.001
            probs = probs/probs.sum(axis=1,keepdims = True)
            probs = torch.log(probs).float()
            reconstruction[index] += probs
        elif(rec_type == 'Naive Averaging'):
            reconstruction[index] += probs
        elif(rec_type == 'Histogram'):
            reconstruction[index] += 1
        weights[index] += 1
    weights[weights == 0] = 1
    if(rec_type == 'Naive Bayesian'):
        reconstruction = sm(reconstruction)
    elif(rec_type == 'Geometric Mean'):
        reconstruction = reconstruction/weights.view(-1,1)
        reconstruction = sm(reconstruction)
    elif(rec_type in ['Naive Averaging','Histogram']):
        reconstruction = reconstruction/weights.view(-1,1)
    # renorm just in case:
    reconstruction = reconstruction/reconstruction.sum(axis = 1,keepdims = True)
    return reconstruction.cpu().numpy(),triangle_gts[:,:]


class Calibrator:
    def __init__(self,selected_scenes,rec_type,segmentation_model):
        self.selected_scenes = selected_scenes
        self.p = multiprocessing.get_context('forkserver').Pool(processes = 16)
        self.rec_type = rec_type
        self.segmentation_model = segmentation_model
    def eval_calibration(self,T):

        res = []
        T = np.exp(np.array([T]))


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
    from bayes_opt import BayesianOptimization,SequentialDomainReductionTransformer,UtilityFunction
    from bayes_opt.logger import JSONLogger
    from bayes_opt.event import Events
    from bayes_opt.util import load_logs
    import numpy as np
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
    pbounds.update({'T':(min_bounds,max_bounds)})
    bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.5)
    # print(pbounds)
    # pbounds = {'T': (1, 500)}
    optimizer =  BayesianOptimization(
        f=mycal.eval_calibration,
        pbounds=pbounds,
        random_state=1,
        allow_duplicate_points = True
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
    if(len(attempts)> 0):
        load_logs(optimizer, logs=["./scaling_results/{}/optimization_logs.json".format(exp_name)])
    if(len(attempts)==0):
        #always start exploring the neutral calibration, max and min points
        initial_points = {}
        T = 1
        output = mycal.eval_calibration(np.log(T))
        optimizer.register(params = {'T':np.log(1)},target = output)        
        initial_points = {} 
        initial_points_range = np.linspace(min_bounds,max_bounds,num = 50,endpoint = True)
        for temp in initial_points_range:
            output = mycal.eval_calibration(temp)
            optimizer.register(params = {'T':temp},target = output)


    acquisition_function = UtilityFunction(kind="ucb", kappa=2)

    optimizer.maximize(init_points = 20,n_iter = 100,verbose = 2,acquisition_function = acquisition_function)

    print(optimizer.max)
if __name__ == '__main__':
    main()
# results = Parallel(backend = '',n_jobs =5,verbose = 22)(delayed(reconstruct_scene)(scene) for scene in scenes )