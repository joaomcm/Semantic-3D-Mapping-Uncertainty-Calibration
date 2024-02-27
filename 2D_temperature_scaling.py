# from agents.utils.semantic_prediction import SemanticPredMaskRCNN
import os
import numpy as np
from tqdm import tqdm
from glob import glob
import h5py
import torch
import torch.nn as nn
from my_calibration import mECE_Calibration_calc_3D_fix as mECE_Calibration_calc_3D
import argparse
from scene_definitions import get_filenames


class Calibrator:
    def __init__(self,filename):
        self.filename = filename
    def Test_T(self,T):
        T = np.exp(T)
        filename = self.filename
        f = h5py.File(filename,'r')
        cal = mECE_Calibration_calc_3D(n_classes =21,no_void = True,one_hot = False)
        for scene in tqdm(list(f.keys())):
            g=f[scene]
            gts = g['gt']
            logits = g['logits']
            sm = nn.Softmax(dim = 3)

            for i in range(logits.shape[0]//100 + 1):
                logit = torch.from_numpy(logits[100*i:(100*i+1)]).to('cuda:0').float()
                gt = gts[100*i:(100*i+1)].reshape(-1)
                logit = logit/T
                probit = sm(logit).cpu().numpy().reshape(-1,21)
                cal.update_bins(probit,gt)
    #     break
        f.close()
        return -cal.get_mECE()

def main():
    from bayes_opt import BayesianOptimization,UtilityFunction
    from bayes_opt.logger import JSONLogger
    from bayes_opt.event import Events
    from bayes_opt.util import load_logs

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type = str,help="The name of the model used for semantic segmentation [ESANet,Segformer]")
    parser.add_argument("--debug",action="store_true")
    args = parser.parse_args()
    model_name = args.model_name

    exp_name = '2D {} Temperature Calibration'.format(model_name)
    min_bounds = 0.01
    max_bounds = 200
    pbounds= {'T':(np.log(0.01),np.log(200))}
    fnames = get_filenames()
    if(model_name == 'Segformer'):
        if(args.debug):
            filename = '/scratch/bbuq/jcorreiamarques/3d_calibration/h5py_datasets/calibration_validation_logits_lzf.hdf5'
        else:
            filename = '/tmp/calibration_validation_logits_lzf.hdf5'
    elif(model_name == 'ESANet'):
        if(args.debug):
            filename = '/scratch/bbuq/jcorreiamarques/3d_calibration/h5py_datasets/ESANET_calibration_validation_logits_lzf.hdf5'
        else:
            filename = '/tmp/ESANET_calibration_validation_logits_lzf.hdf5'
    mycal = Calibrator(filename = filename)



    optimizer =  BayesianOptimization(
        f=mycal.Test_T,
        pbounds=pbounds,
        random_state=1,
        allow_duplicate_points = True
    )


    try:
        os.makedirs('./bayes_logs/{}/'.format(exp_name))
    except Exception as e:
        print(e)
        pass
    attempts = glob("./bayes_logs/{}/*.json".format(exp_name))
    logger = JSONLogger(path="./bayes_logs/{}/optimization_logs.json".format(exp_name),reset = True)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    initial_points_range = np.linspace(np.log(min_bounds),np.log(max_bounds),num = 50,endpoint = True)
    for temp in initial_points_range:
        output = mycal.Test_T(temp)
        optimizer.register(params = {'T':temp},target = output)

    acquisition_function = UtilityFunction(kind="ucb", kappa=1)

    optimizer.maximize(init_points = 0,n_iter = 50,verbose = 2,acquisition_function = acquisition_function)

    pass

if __name__ == '__main__':
    main()