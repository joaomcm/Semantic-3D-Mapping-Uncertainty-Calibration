import faulthandler
from tkinter import E
# from agents.utils.semantic_prediction import SemanticPredMaskRCNN
faulthandler.enable()

import os


os.environ["OMP_NUM_THREADS"] = "8" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "8" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "8" # export NUMEXPR_NUM_THREADS=6

# import os
import numpy as np
import open3d as o3d
import open3d.core as o3c
import cv2
from tqdm import tqdm
from klampt.math import se3
from sens_reader import scannet_scene_reader
import pickle
import traceback
from scene_definitions import get_larger_test_and_validation_scenes,get_filenames
import gc
from experiment_setup import Experiment_Generator
from functools import partial
import pickle
import multiprocessing
import argparse
import json

processes = 8

def reconstruct_scene(scene,experiment_name,experiment_settings,debug,oracle):


    EG = Experiment_Generator()
    fnames = get_filenames()
    rec,model = EG.get_reconstruction_and_model(experiment = experiment_settings,process_id = multiprocessing.current_process()._identity[0])
    if(experiment_settings['integration'] == 'Generalized'):
        get_semantics = model.get_raw_logits
    # elif(experiment_settings['integration'] == 'Histogram'):
    #     get_semantics = model.classify
    else:
        get_semantics = model.get_pred_probs

    # if(not debug):
    #     root_dir = "/tmp/scannet_v2"
    # else:
    #     root_dir = "/scratch/bbuq/jcorreiamarques/3d_calibration/scannet_v2"
    root_dir = fnames['ScanNet_root_dir']
    savedir = "{}/{}/".format(fnames['results_dir'],experiment_name)
    # savedir = '/scratch/bbuq/jcorreiamarques/3d_calibration/Results/{}/'.format(experiment_name)
    if(not os.path.exists(savedir)):
        try:
            os.mkdir(savedir)
        except Exception as e:
            print(e)
    if debug:
        lim = -1
    else:
        lim = -1
    # pdb.set_trace()
    folder = '{}/{}'.format(savedir,scene)
    if not os.path.exists(folder):
        try:
            os.mkdir(folder)
        except Exception as e:
            print(e)
    try:
        device = o3d.core.Device('CUDA:0')



        my_ds = scannet_scene_reader(root_dir, scene ,lim = lim,disable_tqdm = True)
        total_len = len(my_ds)

        if(lim == -1):
            lim = total_len
        randomized_indices = np.array(list(range(lim)))
        np.random.seed(0)
        proc_num = multiprocessing.current_process()._identity[0]%(processes+1) + 1
        for idx,i in tqdm(enumerate(randomized_indices),total = lim,desc = 'proc {}'.format(proc_num),position = proc_num):
            
            try:
                data_dict = my_ds[i]
            except:
                print('\nerror while loading frame {} of scene {}\n'.format(i,scene))
                traceback.print_exc()
                continue
                
            depth = data_dict['depth']#o3d.t.io.read_image(depth_file_names[i]).to(device)
#                 print(depth.shape)
            # print(depth.max(),depth.min())
            intrinsic = o3c.Tensor(data_dict['intrinsics_depth'][:3,:3].astype(np.float64))
            depth = o3d.t.geometry.Image(depth).to(device)
            extrinsic = se3.from_ndarray(data_dict['pose'])

            semantic_label = get_semantics(data_dict['color'],depth = data_dict['depth'],x = depth.rows,y = depth.columns)

#                 semantic_label = None
            if(oracle):
                semantic_label_gt = cv2.resize(data_dict['semantic_label'],(depth.columns,depth.rows),interpolation= cv2.INTER_NEAREST)
                rec.update_vbg(data_dict['depth'],data_dict['intrinsics_depth'][:3,:3].astype(np.float64),
                            data_dict['pose'],semantic_label = semantic_label,semantic_label_gt = semantic_label_gt)
            else:
                rec.update_vbg(data_dict['depth'],data_dict['intrinsics_depth'][:3,:3].astype(np.float64),
                            data_dict['pose'],semantic_label = semantic_label)
            del intrinsic
            del depth
        pcd,labels = rec.extract_point_cloud(return_raw_logits = False)
        o3d.io.write_point_cloud(folder+'/pcd_{:05d}.pcd'.format(idx), pcd, write_ascii=False, compressed=True, print_progress=False)
        pickle.dump(labels,open(folder+'/labels_{:05d}.p'.format(idx),'wb'))
        # torch.cuda.empty_cache()
        # rec.save_vbg(folder+'/vbg.npz')
        # pickle.dump(rec.positions,open(folder+'/position.p','wb'))
        # pdb.set_trace()
        del rec

        gc.collect()

    except Exception as e:
        traceback.print_exc()
        del rec

def get_experiments():
    a = json.load(open('./experiments_and_short_names.json','r'))
    experiments = a['experiments']
    return experiments

def main():
    import torch

    torch.set_float32_matmul_precision('medium')

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true')
    parser.set_defaults(debug = False)
    parser.add_argument('--start', type=int, default=0,
                        help="""starting Reconstruction""")
    parser.add_argument('--end', type=int, default=-1,
                        help="""starting Reconstruction""")
    args = parser.parse_args()
    
    experiments = get_experiments()


    print('\n\n reconstructing {}\n\n'.format(experiments[args.start:args.end]))
    for experiment in experiments[args.start:args.end]:
        print(experiment)
        # experiment = 'ESANET Histogram.json'
        # experiment = 'ESANET Geometric Mean.json'

        # experiment = 'Learned Integration Ablation - vector only.json'
        # experiment = 'learned_generalized_integration.json'
        experiment_name = experiment
        experiment_settings = json.load(open('./experiments/{}.json'.format(experiment),'rb'))
        experiment_settings.update({'experiment_name':experiment_name})
        import multiprocessing
        debug = args.debug
        oracle = experiment_settings['oracle']
        val_scenes,test_scenes = get_larger_test_and_validation_scenes()
        # test_scenes = get_learned_calibration_validation_scenes()
        selected_scenes = sorted(test_scenes)
        # selected_scenes = sorted(["scene0307_00"])
        # reconstruct_scene(selected_scenes[0])
        # selected_scenes = ['scene0231_00']
        # selected_scenes =['scene0314_00']
        # if(not debug):
        p = multiprocessing.get_context('forkserver').Pool(processes = processes,maxtasksperchild = 1)

        res = []
        for a in tqdm(p.imap_unordered(partial(reconstruct_scene,experiment_name = experiment_name,experiment_settings=experiment_settings,debug = debug,oracle = oracle),selected_scenes,chunksize = 1), total= len(selected_scenes),position = 0,desc = 'tot_scenes'):
                res.append(a)
        # else:
        #     for scene in selected_scenes:
        #         reconstruct_scene(scene,experiment_name,experiment_settings,debug,oracle)
        torch.cuda.empty_cache()
        o3d.core.cuda.release_cache()

if __name__=='__main__':
    main()