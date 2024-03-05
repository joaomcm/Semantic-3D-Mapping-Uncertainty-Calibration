import pdb
import cv2
import numpy as np
import pandas as pd
import gc
import os
import traceback

from joblib import Parallel, delayed


import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

from utils.scene_definitions import get_filenames
from utils.sens_reader import scannet_scene_reader


def reading_func(scene):
    lim = -1
    # print('starts function')
    try:
        fnames = get_filenames()
        root_dir = fnames['ScanNet_root_dir']
        save_dir = fnames['finetune_dir'] + "/{}/{}"
        save_dir2 = "{}/{}".format(fnames['finetune_dir'],scene)
        try:
            os.makedirs(save_dir.format(scene,'rgb'),exist_ok = True)
            os.makedirs(save_dir.format(scene,'depth'),exist_ok = True)
            os.makedirs(save_dir.format(scene,'gt'),exist_ok = True)

        except Exception as e:
            print(e)
            pass
        ds = scannet_scene_reader(root_dir, scene ,lim = lim,disable_tqdm=True)
        total_len = len(ds)
        if(lim == -1):
            lim = total_len
        total_images = 0
        for i,index in enumerate(range(0,lim,30)):
            try:
                data_dict = ds[index]
            except:
                traceback.print_exc()
                continue
            color = cv2.resize(data_dict["color"],(data_dict['depth'].shape[1],data_dict['depth'].shape[0]),interpolation= cv2.INTER_AREA).astype(np.uint8)
            # semantic_label_gt = cv2.resize(data_dict["semantic_label"],(data_dict['depth'].shape[1],data_dict['depth'].shape[0]),interpolation= cv2.INTER_NEAREST).astype(np.uint8)
            depth = data_dict['depth'].astype(np.uint16)
            depth_path = '{}/{}/{}.png'.format(save_dir2,'depth',total_images)
            color_path = '{}/{}/{}.png'.format(save_dir2,'rgb',total_images)
            gt_path = '{}/{}/{}.png'.format(save_dir2,'gt',total_images)
            # cv2.imwrite(depth_path,depth)
            if not cv2.imwrite(color_path,color):
                print('ohfuckitfailed\n')
                pdb.set_trace()
            # cv2.imwrite(gt_path,semantic_label_gt)
            total_images += 1

        del ds
        gc.collect()

    except:
        traceback.print_exc()
        pass
        
    return []

if __name__ == '__main__':
    train_splits = pd.read_csv('../settings/train_split.txt',header= None)
    train_splits.columns = ['scenes']
    selected_scenes = sorted(train_splits.scenes.tolist())
    Parallel(n_jobs = 8,backend = 'sequential',verbose = 100)(delayed(reading_func)(scene) for scene in selected_scenes)
    # reading_func(selected_scenes[0])