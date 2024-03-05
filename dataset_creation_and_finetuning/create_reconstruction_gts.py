import faulthandler
from tkinter import E


import os
import sys

os.environ["OMP_NUM_THREADS"] = "2" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "2" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "2" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "2" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "2" # export NUMEXPR_NUM_THREADS=6

# import os
import argparse
import gc
import json
import multiprocessing
import pickle
import traceback
from functools import partial

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
from klampt.math import se3
from tqdm import tqdm

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)


from utils.scene_definitions import get_filenames, get_larger_test_and_validation_scenes
from reconstruction import GroundTruthGenerator
from utils.sens_reader import scannet_scene_reader






processes = 4
faulthandler.enable()


def reconstruct_scene(scene,experiment_name):


    fnames = get_filenames()
    voxel_size = 0.025 #3.0 / 512
    trunc_multiplier =  8
    res = 8
    n_labels = 21
    depth_scale = 1000.0
    depth_max = 5.0
    miu = 0.001

    rec = GroundTruthGenerator(depth_scale=depth_scale,depth_max = depth_max,
        res = res,voxel_size = voxel_size,trunc_multiplier=trunc_multiplier,
            n_labels=n_labels,integrate_color=False,miu = miu)


    root_dir = fnames['ScanNet_root_dir']
    savedir = "{}/{}/".format(fnames['results_dir'],experiment_name)
    # savedir = '/scratch/bbuq/jcorreiamarques/3d_calibration/Results/{}/'.format(experiment_name)
    if(not os.path.exists(savedir)):
        try:
            os.mkdir(savedir)
        except Exception as e:
            print(e)

    lim = -1
    # pdb.set_trace()
    folder = '{}/'.format(savedir)
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
            try:
                intrinsic = o3c.Tensor(data_dict['intrinsics_depth'][:3,:3].astype(np.float64))
                depth = o3d.t.geometry.Image(depth).to(device)
                semantic_label_gt = cv2.resize(data_dict['semantic_label'],(depth.columns,depth.rows),interpolation= cv2.INTER_NEAREST)
            except Exception as e: 
                print(e)
                continue
            rec.update_vbg(data_dict['depth'],data_dict['intrinsics_depth'][:3,:3].astype(np.float64),
                        data_dict['pose'],semantic_label = semantic_label_gt)

            del intrinsic
            del depth
        pcd,labels = rec.extract_point_cloud(return_raw_logits = False)
        o3d.io.write_point_cloud(folder+'/gt_pcd_{}.pcd'.format(scene), pcd, write_ascii=False, compressed=True, print_progress=False)
        pickle.dump(labels,open(folder+'/gt_labels_{}.p'.format(scene),'wb'))
        # torch.cuda.empty_cache()
        rec.save_vbg(folder+'/gt_vbg_{}.npz'.format(scene))
        # pickle.dump(rec.positions,open(folder+'/position.p','wb'))
        # pdb.set_trace()
        del rec

        gc.collect()

    except Exception as e:
        traceback.print_exc()
        del rec

def main():
    import torch
    import multiprocessing

    torch.set_float32_matmul_precision('medium')
    val_scenes,test_scenes = get_larger_test_and_validation_scenes()
    # test_scenes = get_learned_calibration_validation_scenes()
    selected_scenes = sorted(test_scenes+val_scenes)
    p = multiprocessing.get_context('forkserver').Pool(processes = processes,maxtasksperchild = 1)
    res = []
    for a in tqdm(p.imap_unordered(partial(reconstruct_scene,experiment_name = 'reconstruction_gts'),selected_scenes,chunksize = 1), total= len(selected_scenes),position = 0,desc = 'tot_scenes'):
            res.append(a)
    torch.cuda.empty_cache()
    o3d.core.cuda.release_cache()

if __name__=='__main__':
    main()