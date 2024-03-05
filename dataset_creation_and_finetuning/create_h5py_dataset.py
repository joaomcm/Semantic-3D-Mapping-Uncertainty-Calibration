if __name__ == "__main__":
    import argparse
    import gc
    import pdb
    import pickle
    import traceback
    from glob import glob
    from os import environ

    import cv2
    import h5py
    import numba as nb
    import numpy as np
    import open3d as o3d
    import open3d.core as o3c
    import torch
    from klampt.math import se3
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm


    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    sys.path.append(parent_dir)

    from utils.segmentation_model_loader import FineTunedESANet, FineTunedTSegmenter
    from calibration_experiments.experiment_setup import Experiment_Generator
    from utils.rendering_utils import get_camera_rays, render_depth_and_normals
    from utils.ScanNet_scene_definitions import get_filenames,get_fixed_train_and_val_splits
    from utils.sens_reader import scannet_scene_reader

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        help="split type for which you are generating the dataset, [train,validation]",
    )
    parser.add_argument("--start", help="scene number start saving from", type=int)
    parser.add_argument(
        "--end", help="scene number to end saving in this task", type=int
    )
    parser.add_argument(
        "--model_type",
        help="name of the model for which the logits are being stored, valid values = [ESANet,Segformer]",
    )
    args = parser.parse_args()

    fnames = get_filenames()
    split = args.split
    start_saving = args.start
    end_saving = args.end
    model_type = args.model_type
    assert (
        split in ["train", "validation"]
    ), "The selected split, {}, is an invalid split. Valid splits are [train,validation]".format(
        split
    )
    assert (
        model_type in ["ESANet", "Segformer"]
    ), "The selected model_type,{},is an invalid model. Valid model types are [ESANet,Segformer]".format(
        model_type
    )
    dataset_filename = fnames[
        "h5py_dataset_dir"
    ] + "/{}_calibration_{}_logits_lzf.hdf5".format(model_type, split)
    root_dir = fnames["ScanNet_root_dir"]
    vbg_file_template = fnames["results_dir"] + "/{}/gt_vbg_{}.npz"
    experiment = "reconstruction_gts"
    singleton_save_folder = fnames["singleton_dataset_dir"] + "/{}/{}/{}"
    singleton_save_dir = singleton_save_folder + "/{}.p"
    calibration_index_dict = "{}_calibration_index_dict.p".format(model_type)
    base_dataset = True
    get_normal = True
    do_the_rest = True
    update_the_index_dict = True
    singleton_voxels = True
    singleton_voxels_h5py = True
    train, val = get_fixed_train_and_val_splits()
    if split == "train":
        calibration_scenes = train
    else:
        calibration_scenes = val

    def render_indices(this_vbg, depth, intrinsic, pose):
        device = o3d.core.Device("CUDA:0")
        intrinsic = intrinsic[:3, :3].astype(np.float64)
        intrinsic = o3c.Tensor(intrinsic.astype(np.float64))
        depth = o3d.t.geometry.Image(depth).to(device)
        extrinsic = se3.from_ndarray(pose)
        extrinsic = se3.ndarray(se3.inv(extrinsic))
        extrinsic = o3c.Tensor(extrinsic)

        #     frustum_block_coords = this_vbg.compute_unique_block_coordinates(
        #         depth, intrinsic, extrinsic, 1000, 5.0)

        result = this_vbg.ray_cast(
            block_coords=depth,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            width=depth.columns,
            height=depth.rows,
            render_attributes=["index", "interp_ratio"],
            depth_scale=1000,
            depth_min=0,
            depth_max=5.0,
            weight_threshold=15,
            range_map_down_factor=8,
            trunc_voxel_multiplier=1,
        )
        return result["index"].cpu().numpy(), result["interp_ratio"].cpu().numpy()

    def render_indices2(rscene, intrinsic, pose, depth):
        device = o3d.core.Device("CUDA:0")
        intrinsic = intrinsic[:3, :3].astype(np.float32)
        intrinsic = o3c.Tensor(intrinsic.astype(np.float32))
        depth = o3d.t.geometry.Image(depth).to(device)
        extrinsic = se3.from_ndarray(pose)
        extrinsic = se3.ndarray(se3.inv(extrinsic))
        extrinsic = o3c.Tensor(extrinsic.astype(np.float32))
        #     pdb.set_trace()
        rays = rscene.create_rays_pinhole(
            intrinsic, extrinsic, depth.columns, depth.rows
        )
        res = rscene.cast_rays(rays, nthreads=8)
        return res["primitive_ids"]

    if base_dataset:
        f = h5py.File(dataset_filename, "a")

        if model_type == "Segformer":
            model = FineTunedTSegmenter(temperature=1)
        elif model_type == "ESANet":
            model = FineTunedESANet(temperature=1)

        device = o3d.core.Device("CUDA:0")

        for scene in tqdm(calibration_scenes):
            vbg_file = vbg_file_template.format(experiment, scene)
            vbg = o3d.t.geometry.VoxelBlockGrid.load(vbg_file)
            rscene = o3d.t.geometry.RaycastingScene()
            triangle_mesh = vbg.extract_triangle_mesh()

            #     tm = o3d.t.geometry.TriangleMesh.from_legacy(
            #                                     triangle_mesh.cpu().to_legacy())
            rscene.add_triangles(triangle_mesh.cpu())
            del triangle_mesh
            del vbg

            try:
                g = f.create_group(scene)
            except Exception as e:
                print(e)
                g = f[scene]
            lim = -1
            my_ds = scannet_scene_reader(root_dir, scene, lim=lim, disable_tqdm=True)
            data_dict = my_ds[0]
            total_len = len(my_ds)

            if lim == -1:
                lim = total_len
            try:
                g.attrs["depth_intrinsics"] = data_dict["intrinsics_depth"]
                g.attrs["color_intrinsics"] = data_dict["intrinsics_color"]
                logits = g.create_dataset(
                    "logits",
                    (lim, 480, 640, 21),
                    maxshape=(None, 480, 640, 21),
                    chunks=(1, 480, 640, 21),
                    dtype=np.float16,
                    compression="lzf",
                )
                depths = g.create_dataset(
                    "depth",
                    (lim, 480, 640),
                    maxshape=(None, 480, 640),
                    chunks=(1, 480, 640),
                    dtype=np.uint16,
                    compression="lzf",
                )
                poses = g.create_dataset(
                    "poses",
                    (lim, 4, 4),
                    maxshape=(None, 4, 4),
                    chunks=(1, 4, 4),
                    dtype=np.uint32,
                )
                indices = g.create_dataset(
                    "indices",
                    (lim, 480, 640),
                    maxshape=(None, 480, 640),
                    chunks=(1, 480, 640),
                    dtype=np.uint32,
                    compression="lzf",
                )
                gts = g.create_dataset(
                    "gt",
                    (lim, 480, 640),
                    maxshape=(None, 480, 640),
                    chunks=(1, 480, 640),
                    dtype=np.uint8,
                    compression="lzf",
                )
            except:
                logits = g["logits"]
                depths = g["depth"]
                poses = g["poses"]
                indices = g["indices"]
                gts = g["gt"]
            #     normals = g.create_dataset('')
            total_images = 0
            invalid_poses = 0
            for i in tqdm(range(lim), desc="main_dataset"):
                try:
                    data_dict = my_ds[i]
                    depth = data_dict["depth"]
                    depth2 = o3d.t.geometry.Image(depth).to(device)
                    color = data_dict["color"]
                    pose = data_dict["pose"]
                    if np.any(np.logical_not(np.isfinite((pose)))):
                        invalid_poses += 1
                        print(
                            "invalid poses {}, actual frame {}".format(invalid_poses, i)
                        )
                        continue

                    gt = cv2.resize(
                        data_dict["semantic_label"],
                        (depth2.columns, depth2.rows),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.uint8)
                    if np.any(np.logical_not(np.isfinite(gt))):
                        print("invalid gt")
                        continue
                except:
                    traceback.print_exc()
                    continue
                # pdb.set_trace()
                depths[total_images, :, :] = depth
                poses[total_images, :, :] = pose
                logit = model.get_raw_logits(
                    color, depth, depth2.rows, depth2.columns
                ).astype(np.float16)
                gts[total_images, :, :] = gt
                logits[total_images, :, :, :] = logit
                # print('got to rendering')
                index = render_indices2(
                    rscene, data_dict["intrinsics_depth"], pose, depth
                ).numpy()
                # print('rendered')
                index[index == rscene.INVALID_ID] = 0
                indices[total_images, :, :] = index
                total_images += 1
                # print('saving')
            # resizing the dataset to account for failed openings
            for dset in [logits, depths, poses, indices, gts]:
                dshape = list(dset.shape)
                dshape[0] = total_images
                dset.resize(dshape)
            old_max = 0
            for i in tqdm(indices):
                new_max = i.max()
                if new_max > old_max:
                    old_max = new_max
            g.attrs["max_triangle_index"] = old_max

        f.close()

    if get_normal:
        # from agents.utils.semantic_prediction import SemanticPredMaskRCNN


        f = h5py.File(dataset_filename, "a")
        device = o3d.core.Device("CUDA:0")
        EG = Experiment_Generator()

        for scene in calibration_scenes:
            experiment = {
                "integration": "Bayesian Update",
                "calibration": "None",
                "oracle": False,
                "L": 1,
                "epsilon": 0.35,
                "segmentation": model_type,
                "learned_params": {
                    "temperature": "sgd_optimized_temperatures.p",
                    "weights": "sgd_optimized_weights.p",
                },
            }

            rec, model = EG.get_reconstruction_and_model(experiment=experiment,process_id = 0)
            get_semantics = model.get_pred_probs
            # print(scene)
            g = f[scene]
            #     tm = o3d.t.geometry.TriangleMesh.from_legacy(
            #                                     triangle_mesh.cpu().to_legacy())
            lim = -1
            my_ds = scannet_scene_reader(root_dir, scene, lim=lim, disable_tqdm=False)
            data_dict = my_ds[0]
            total_len = len(my_ds)

            if lim == -1:
                lim = total_len

            try:
                normals = g.create_dataset(
                    "normals",
                    (lim, 480, 640, 3),
                    maxshape=(None, 480, 640, 3),
                    chunks=(1, 480, 640, 3),
                    dtype=np.float16,
                    compression="lzf",
                )
            except Exception as e:
                print(e)
                normals = g["normals"]
            total_images = 0
            for i in tqdm(range(lim), desc="normals"):
                try:
                    data_dict = my_ds[i]
                    depth = data_dict["depth"]
                    depth2 = o3d.t.geometry.Image(depth).to(device)
                    color = data_dict["color"]
                    pose = data_dict["pose"]
                    if np.any(np.logical_not(np.isfinite((pose)))):
                        print("invalid pose")
                        continue

                    gt = cv2.resize(
                        data_dict["semantic_label"],
                        (depth2.columns, depth2.rows),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.uint8)
                    if np.any(np.logical_not(np.isfinite(gt))):
                        print("invalid gt")
                        continue
                    # gt = cv2.resize(data_dict['semantic_label'],(depth2.columns,depth2.rows),interpolation= cv2.INTER_NEAREST)
                except:
                    traceback.print_exc()
                    continue
                # pdb.set_trace()
                with torch.no_grad():
                    semantic_label = get_semantics(
                        data_dict["color"],
                        depth=data_dict["depth"],
                        x=depth2.rows,
                        y=depth2.columns,
                    )
                rec.update_vbg(
                    data_dict["depth"],
                    data_dict["intrinsics_depth"][:3, :3].astype(np.float64),
                    data_dict["pose"],
                    semantic_label=semantic_label,
                )
                rendered_depth, rendered_normals = render_depth_and_normals(
                    rec.vbg,
                    depth,
                    data_dict["intrinsics_depth"][:3, :3],
                    data_dict["pose"],
                    use_depth=True,
                )
                normals[total_images, :, :, :] = rendered_normals.astype(np.float16)
                total_images += 1
            del rec
            torch.cuda.empty_cache()
            # resizing the dataset to account for failed openings
            for dset in [normals]:
                dshape = list(dset.shape)
                dshape[0] = total_images
                dset.resize(dshape)

        f.close()

    if do_the_rest:
        # from agents.utils.semantic_prediction import SemanticPredMaskRCNN


        f = h5py.File(dataset_filename, "a")
        for scene in calibration_scenes:
            lim = -1
            g = f[scene]
            logits = g["logits"]
            gts = g["gt"]
            indices = g["indices"]
            depths = g["depth"]
            poses = g["poses"]
            max_triangle_index = g.attrs["max_triangle_index"] + 1
            try:
                triangle_gts = g.create_dataset(
                    "triangle_gts",
                    (max_triangle_index, 1),
                    maxshape=(max_triangle_index, 1),
                    dtype=np.uint8,
                )
            except Exception as e:
                print(e)
                triangle_gts = g["triangle_gts"]
            triangle_gts2 = np.zeros((max_triangle_index, 21))
            max_idxs = []
            for i in tqdm(range(indices.shape[0]), desc="triangle_gts"):
                #         pass
                idx = indices[i, :, :]
                idx[idx == 2**32 - 1] = 0
                triangle_gts2[idx.flatten(), gts[i, :, :].flatten()] += 1
            triangle_gts[:, 0] = np.argmax(triangle_gts2, axis=1)

        f.close()

    if update_the_index_dict:
        f = h5py.File(dataset_filename, "r")
        if(os.path.exists(calibration_index_dict)):
            idx_dict = pickle.load(open(calibration_index_dict, "rb"))
        else:
            idx_dict = {}
        for key in tqdm(f.keys()):
            g = f[key]
            indices = g["indices"][:]
            unique_indices = np.unique(indices)
            idx_dict.update({key: unique_indices})
        pickle.dump(idx_dict, open(calibration_index_dict, "wb"))
        f.close()

    if singleton_voxels:

        f = h5py.File(dataset_filename, "r")


        @nb.jit(parallel=True)
        def is_in_set_pnb(a, b):
            shape = a.shape
            a = a.ravel()
            n = len(a)
            result = np.full(n, False)
            set_b = set(b)
            for i in nb.prange(n):
                if a[i] in set_b:
                    result[i] = True
            return result.reshape(shape)

        # dataset_filename = '/scratch/bbuq/jcorreiamarques/3d_calibration/h5py_datasets/calibration_{}_logits_lzf.hdf5'.format(split)

        threads = "16"
        environ["OMP_NUM_THREADS"] = threads  # export OMP_NUM_THREADS=4
        environ["OPENBLAS_NUM_THREADS"] = threads  # export OPENBLAS_NUM_THREADS=4
        environ["MKL_NUM_THREADS"] = threads  # export MKL_NUM_THREADS=6
        environ["VECLIB_MAXIMUM_THREADS"] = threads  # export VECLIB_MAXIMUM_THREADS=4
        environ["NUMEXPR_NUM_THREADS"] = threads

        idx_dict = pickle.load(open(calibration_index_dict, "rb"))
        end_saving = np.min([len(calibration_scenes), end_saving])
        start_saving = np.min([len(calibration_scenes), start_saving])
        for scene in tqdm(
            calibration_scenes[start_saving:end_saving], position=0, desc="scene_counts"
        ):
            this_singleton_save_folder = singleton_save_folder.format(
                model_type, split, scene
            )
            print("\n\n scene = {}  \n\n ".format(scene))
            if not os.path.exists(this_singleton_save_folder):
                os.makedirs(this_singleton_save_folder)
            total_voxels = 0
            idxs = idx_dict[scene]
            g = f[scene]
            logits_ds = g["logits"]
            indices_ds = g["indices"][:]
            poses_ds = g["poses"][:]
            depth_ds = g["depth"][:]
            triangle_gts_ds = g["triangle_gts"][:]
            # if(logits_ds.shape[0]>=3000):
            normals_ds = g["normals"]
            # else:
            #     normals_ds = g['normals'][:]
            depth_intrinsics = g.attrs["depth_intrinsics"][:]

            rays = get_camera_rays(
                depth_ds[0].shape[0],
                depth_ds[0].shape[1],
                depth_intrinsics[0, 0],
                depth_intrinsics[1, 1],
            ).reshape(normals_ds[0].shape)
            arrays = np.array_split(idx_dict[scene][1:-2], 10)
            for selected_idxs in tqdm(arrays, position=1, desc="array nums"):
                data_dicts = []
                for i in range(selected_idxs.shape[0]):
                    data_dicts.append(
                        {"logits": [], "depth": [], "angle": [], "gt": []}
                    )
                chunks = logits_ds.shape[0] // 100
                a = np.array_split(range(logits_ds.shape[0]), chunks)
                for i in tqdm(a, position=2, desc="images"):
                    # mask = np.isin(indices_ds[i],selected_idxs,assume_unique = False)
                    mask = is_in_set_pnb(indices_ds[i], selected_idxs)
                    logits = logits_ds[i][mask]
                    depth = (depth_ds[i].astype(float) / 1000)[mask]
                    normal = normals_ds[i]
                    p = np.abs((normal * rays).sum(axis=3))
                    # p = np.clip(p/(np.linalg.norm(normal,axis =3)*np.linalg.norm(rays,axis=2)),-1,1)
                    p[p > 1] = 1
                    projective_angle = np.arccos((p)) * 180 / np.pi
                    projective_angle = projective_angle[mask]
                    projective_angle = np.nan_to_num(
                        projective_angle, nan=0, posinf=0, neginf=0
                    )
                    indices = indices_ds[i][mask]
                    local_indices = selected_idxs.searchsorted(indices)
                    for local_idx, local_logits, local_angle, local_depth in zip(
                        local_indices, logits, projective_angle, depth
                    ):
                        data_dicts[local_idx]["logits"].append(local_logits)
                        data_dicts[local_idx]["depth"].append(local_depth)
                        data_dicts[local_idx]["angle"].append(local_angle)
                    del local_indices
                    del indices
                    del logits
                    del projective_angle
                    del depth
                    gc.collect()

                for local_idx, idx in enumerate(selected_idxs):
                    data_dicts[local_idx]["gt"] = np.array([triangle_gts_ds[idx]])
                for dct in tqdm(data_dicts, desc="saving"):
                    dct.update({"logits": np.array(dct["logits"])})
                    dct.update({"depth": np.array(dct["depth"])})
                    dct.update({"angle": np.array(dct["angle"])})
                    pickle.dump(
                        dct,
                        open(
                            singleton_save_dir.format(
                                model_type, split, scene, total_voxels
                            ),
                            "wb",
                        ),
                    )
                    total_voxels += 1
                del data_dicts
                gc.collect()

            del logits_ds
            del normals_ds
            del triangle_gts_ds
            del indices_ds
            gc.collect()

        f.close()
    if singleton_voxels_h5py:

        create = True
        evaluate = False
        filename = fnames['singleton_dataset_dir']+'/{}_{}.h5py'.format(model_type,split)
        train_directory = fnames["singleton_dataset_dir"] + '/{}/{}/**/*.p'.format(model_type,split)
        class voxel_readings_dataset(Dataset):
            def __init__(self,ds_directory,max_obs = 1000,start = 0):
                self.ds_directory = ds_directory
                self.files = sorted(glob(self.ds_directory,recursive=True))[start:]
                self.max_obs = max_obs
            def __len__(self):
                return len(self.files)
            def __getitem__(self,idx):
                sample = pickle.load(open(self.files[idx],'rb'))
                keys = ['logits', 'depth', 'angle']
                if(sample[keys[0]].shape[0] > self.max_obs):
                    chosen_indices = np.random.choice(sample[keys[0]].shape[0],self.max_obs, replace = False)
                    for key in keys:
                        tmp = sample[key][chosen_indices]
                        sample.update({key:tmp})
                    
                return sample

        def debug_collate(data):
            keys = ['logits', 'depth', 'angle']
            result = {}
            for key in keys:
                tmp = []
                for d in data:
                    sample = d[key]
                    tmp.append(torch.Tensor(sample))
                    
                result.update({key:tmp})
                
            labels = [d['gt'] for d in data]
            padded_ds = {}
            for key in keys:
                padded_ds.update({key:pad_sequence(result[key],batch_first = True)})
            padded_ds.update({'label':torch.tensor(np.array(labels))})
            del result
            del data
            return padded_ds
        
        start = 0
        train_ds = voxel_readings_dataset(train_directory,start = start)
        train_dataloader = DataLoader(train_ds,batch_size = 1,shuffle = False,collate_fn = debug_collate,num_workers = 12,prefetch_factor = 3)

        if(create):
            f = h5py.File(filename,'w')

            lim = len(train_ds)
            try:
                logits = f.create_dataset("logits", (lim,1000,21), maxshape=(None, 1000,21),chunks = (1,1000,21),dtype = np.float16,compression = 'lzf')
                depths = f.create_dataset("depth", (lim,1000), maxshape=(None, 1000),chunks = (1,1000),dtype = np.float16,compression = 'lzf')
                angles = f.create_dataset("angle", (lim,1000), maxshape=(None, 1000),chunks = (1,1000),dtype = np.float16,compression = 'lzf')
                labels = f.create_dataset("label", (lim,1), maxshape=(None, 1),chunks = (1,1),dtype = np.uint8,compression = 'lzf')
            except Exception as e:
                print(e)
                logits = f["logits"]
                depths = f["depth"]
                angles = f["angle"]
                labels = f["label"]
                
            # g = f.create_group('train')

            lim = len(train_ds)

            i = start
            for dct in tqdm(train_dataloader):
                logit = dct['logits'].cpu().numpy().astype(np.float16)
                depth = dct['depth'].cpu().numpy().astype(np.float16)
                angle = dct['angle'].cpu().numpy().astype(np.float16)
                label = dct['label'].cpu().numpy().astype(np.uint8).flatten()
                entries = logit.shape[1]  
                logits[i,:entries,:] = logit
                depths[i,:entries] = depth
                angles[i,:entries] = angle
                labels[i,0] = label
                i+=1
            f.close()

        if(evaluate):
            print('entering evaluation')
            class voxel_readings_dataset2(Dataset):
                def __init__(self,ds_directory,size = 900000):
                    self.ds_directory = ds_directory
                    self.size = size
                    self.split = split
                def h5py_worker_init(self):
                    print('starting h5py dataset')
                    self.f = h5py.File(self.ds_directory,'r')
                    self.logits = self.f['logits']
                    self.depth = self.f['depth']
                    self.angle = self.f['angle']
                    self.label = self.f['label']
                    
                def __len__(self):
                    return self.size
                def __getitem__(self,idx):
                    sample = {'logits':self.logits[idx,:,:],'depth':self.depth[idx,:],
                            'angle':self.angle[idx,:],'label':self.label[idx]}
                    return sample
            
            def worker_init_fn(worker_id):
                worker_info = torch.utils.data.get_worker_info()
                dataset = worker_info.dataset
                dataset.h5py_worker_init()


            size = h5py.File(filename,'r')['logits'].shape[0]
            train_ds2 = voxel_readings_dataset2(filename,size = size)
            train_dataloader2 = DataLoader(train_ds2,batch_size = 9000,shuffle = True,num_workers = 3,worker_init_fn = worker_init_fn,persistent_workers = True)
            # train_dataloader = DataLoader(train_ds,batch_size = 9000,shuffle = False,collate_fn = debug_collate,num_workers = 8,prefetch_factor = 3)

            for i in tqdm(train_dataloader2,desc = 'h5py dataloader'):
                c = i