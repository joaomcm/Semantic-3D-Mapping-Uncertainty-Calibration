from datasets import Dataset, IterableDataset
import cv2
from glob import glob
import numpy as np
from PIL import Image
from utils.scene_definitions import get_filenames

fnames = get_filenames()
images_dir = fnames['finetune_dir']
huggingface_dset_dir = fnames['ScanNet_huggingface_dataset_dir']
color_pattern = images_dir + "/**/rgb/*.png"
label_pattern = images_dir + "/**/gt/*.png"
# color_= '~/scratch/scannet_sample/color/{}.jpg'
# label_file = '~/scratch/scannet_sample/label/{}.png'
# n_imgs = len(glob('~/scratch/scannet_sample/color/*.jpg'))
color_files = sorted(glob(color_pattern))
label_files = sorted(glob(label_pattern))
# n_imgs = 500
def gen():
    for color_file,label_file in zip(color_files,label_files):
        c1 = Image.fromarray(cv2.imread(color_file,cv2.IMREAD_UNCHANGED))
        s1 = Image.fromarray(cv2.imread(label_file,cv2.IMREAD_UNCHANGED))
        yield {'pixel_values':c1,'label':s1}

ds =  Dataset.from_generator(gen,cache_dir = './tmp')
ds.save_to_disk(huggingface_dset_dir)