"""Modified from Roger Qiu - which modified from ScanNet Source code"""

from PIL import Image

import os, struct
import numpy as np
import zlib
import imageio.v2 as imageio
import cv2
import csv
import shutil
from tqdm import tqdm
import zipfile
import pandas as pd



def unzip(zip_path, zip_type,scene_name):
    assert zip_type in ["instance-filt", "label-filt"]
    target_dir = f'/tmp/{zip_type}/{scene_name}'
    if os.path.exists(target_dir):
        pass
        # shutil.rmtree(target_dir)
    else:
        os.makedirs(target_dir)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
    return os.path.join(target_dir, zip_type)




class RGBDFrame():
    def load(self, file_handle):
        self.camera_to_world = np.asarray(struct.unpack('f'*16, file_handle.read(16*4)), dtype=np.float32).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = b''.join(struct.unpack('c'*self.color_size_bytes, file_handle.read(self.color_size_bytes)))
        self.depth_data = b''.join(struct.unpack('c'*self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))

    def decompress_depth(self, compression_type):
        if compression_type == 'zlib_ushort':
             return self.decompress_depth_zlib()
        else:
             raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == 'jpeg':
             return self.decompress_color_jpeg()
        else:
             raise

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)



class scannet_scene_reader:
    def __init__(self, root_dir, scene_name, lim = -1,just_size = False,disable_tqdm = False):
        self.lim = lim
        label_file = os.path.join(root_dir, 'scannetv2-labels.combined.tsv')
        scannet_id_nyu_dict = {}
        self.just_size = just_size
        self.disable_tqdm = disable_tqdm
        with open(label_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            for row_dict in reader:
                scannet_id = row_dict['id']
                nyu40_id = row_dict['nyu40id']
                scannet_id_nyu_dict[int(scannet_id)] = int(nyu40_id)

        # # map label as instructed in http://kaldir.vc.in.tum.de/scannet_benchmark/labelids.txt
        # VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

        # scannet_subset_map = np.zeros(41) # NYU40 has 40 labels
        # for i in range(len(VALID_CLASS_IDS)):
        #     scannet_subset_map[VALID_CLASS_IDS[i]] = i + 1

        # # This dict maps from fine-grained ScanNet ids (579 categories)
        # # to the 20 class subset as in the benchmark
        # scannet_mapping = np.zeros(max(scannet_id_nyu_dict) + 1)

        # for k in scannet_id_nyu_dict:
        #     scannet_mapping[k] = scannet_subset_map[scannet_id_nyu_dict[k]]
        df = pd.read_csv(label_file,sep = '\t')

        subset = df.loc[:,['id','raw_category','nyu40id','nyu40class']]
        subset.loc[:,'new_class'] = 0

        ordered_relevant_classes = np.array([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]).tolist()
        a = np.zeros(41)
        for i in ordered_relevant_classes:
            a[i] = ordered_relevant_classes.index(i)+1

        subset.loc[subset.nyu40id.isin(ordered_relevant_classes),'new_class'] = a[subset.loc[subset.nyu40id.isin(ordered_relevant_classes),'nyu40id']]
        subset.new_class = subset.new_class.astype(int)

        conversion = np.zeros(subset.id.max()+1)
        conversion[subset.id] = subset.new_class
        conversion
        # # HARDCODE FOR NOW
        # printer_scannet_id = 50

        # scannet_mapping[printer_scannet_id] = len(VALID_CLASS_IDS) + 1

        self.scannet_mapping = conversion




        self.version = 4
        
        # Get file paths
        sens_path = os.path.join(root_dir, 'scans', scene_name, f'{scene_name}.sens')
        semantic_zip_path = os.path.join(root_dir, 'scans', scene_name, f'{scene_name}_2d-label-filt.zip')
        # instance_zip_path = os.path.join(root_dir, 'scans', scene_name, f'{scene_name}_2d-instance-filt.zip')
        
        # Load
        if(self.just_size):
            tmp = self.load(sens_path)
            self.size= tmp
        else:
            self.load(sens_path)
        self.label_dir = unzip(semantic_zip_path, 'label-filt',scene_name)
        # self.inst_dir = unzip(instance_zip_path, 'instance-filt',scene_name)
    def load(self, filename):
        COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
        COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}
        with open(filename, 'rb') as f:
            # Read meta data
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack('Q', f.read(8))[0]
            self.sensor_name = b''.join(struct.unpack('c'*strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height =    struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height =    struct.unpack('I', f.read(4))[0]
            self.depth_shift =    struct.unpack('f', f.read(4))[0]
            num_frames =    struct.unpack('Q', f.read(8))[0]
            if(self.just_size):
                return num_frames
            # Read frames
            self.frames = []
            if(self.lim == -1):
                for i in tqdm(range(num_frames),disable = self.disable_tqdm):
                    frame = RGBDFrame()
                    frame.load(f)
                    self.frames.append(frame)
            else:
                for i in tqdm(range(self.lim),disable = self.disable_tqdm):
                    frame = RGBDFrame()
                    frame.load(f)
                    self.frames.append(frame)
    
    def __getitem__(self, idx):
        image_size = None
        assert idx >= 0
        assert idx < len(self.frames)
        depth_data = self.frames[idx].decompress_depth(self.depth_compression_type)
        depth = np.fromstring(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
        if image_size is not None:
            depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
        color = self.frames[idx].decompress_color(self.color_compression_type)
        if image_size is not None:
            color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
        pose = self.frames[idx].camera_to_world
        
        # Read label
        label_path = os.path.join(self.label_dir, f"{idx}.png")
        label_map = np.array(Image.open(label_path))
        label_map = self.scannet_mapping[label_map]
        
        # Read instance map
        # inst_path = os.path.join(self.inst_dir, f"{idx}.png")
        # inst_map = np.array(Image.open(inst_path))
        
        return {
            'color': color,
            'depth': depth,
            'pose': pose,
            'intrinsics_color': self.intrinsic_color,
            'intrinsics_depth':self.intrinsic_depth,
            'semantic_label': label_map,
            'depth_shift':self.depth_shift

            # 'inst_label': inst_map
        }
    
    def __len__(self):
        return len(self.frames)


if __name__ == '__main__':
    root_dir = "/home/motion/data/scannet_v2"
    my_ds = scannet_scene_reader(root_dir, 'scene0050_00')
    data_dict = my_ds[263 + 30]

    data_dict.keys()

    import matplotlib.pyplot as plt

    plt.imshow(data_dict['semantic_label'] == 21)

    # plt.imshow(data_dict['color'])


    plt.show()

    data_dict['color'].shape
