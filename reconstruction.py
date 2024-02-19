"""
Heavily inspired by https://github.com/isl-org/Open3D/blob/master/examples/python/t_reconstruction_system/integrate_custom.py
"""

import numpy as np
import open3d as o3d
import open3d.core as o3c
from tqdm import tqdm
import cv2
from klampt.math import se3
import pickle
import torch
import pdb
from rendering_utils import render_depth_and_normals,get_camera_rays
import torch.utils.dlpack
from torch import linalg as LA
import torch.nn as nn


def get_properties(voxel_grid,points,attribute,res = 8,voxel_size = 0.025,device = o3d.core.Device('CUDA:0')):
    """ This function returns the coordinates of the voxels containing the query points specified by 'points' and their respective attributes
    stored the 'attribute' attribute within the voxel_block_grid 'voxel_grid'

    Args:
        voxel_grid (open3d.t.geometry.VoxelBlockGrid): The voxel block grid containing the attributes and coordinates you wish to extract
        points (np.array [Nx3]- dtype np.float32): array containing the XYZ coordinates of the points for which you wish to extract the attributes in global coordinates
        attribute (str): the string corresponding to the attribute you wish to obtain within your voxel_grid (say, semantic label, color, etc)
        res (int, optional): Resolution of the dense voxel blocks in your voxel block grid.  Defaults to 8.
        voxel_size (float, optional): side length of the voxels in the voxel_block_grid  Defaults to 0.025.
    """
    if(points.shape[0]>0):
        # we first find the coordinate of the origin of the voxel block of each query point and turn it into an open3d tensor
        query = np.floor((points/(res*voxel_size)))
        t_query = o3c.Tensor(query.astype(np.int32),device = device)

        # we then find the remainder between the voxel block origin and each point to find the within-block coordinates of the voxel containing this point
        query_remainder = points-query*res*voxel_size
        query_remainder_idx = np.floor(query_remainder/voxel_size).astype(np.int32)
        qri = query_remainder_idx

        # we then obtain the hashmap 
        hm = voxel_grid.hashmap()

        # we extract the unique voxel block origins for memory reasons and save the mapping for each individual entry
        block_c,mapping = np.unique(query,axis = 0,return_inverse = True)
        t_block_c = o3c.Tensor(block_c.astype(np.int32),device = device)

        # we query the hashmap to find the index corresponding to each of the voxel block origins
        r,m = hm.find(t_block_c)

        # we then find the flattened indices and coordinates of the individual voxels within each of these voxel blocks in memory
        coords,indices = voxel_grid.voxel_coordinates_and_flattened_indices(r.to(o3c.int32))
    #     print(mapping)
        # we then extract the attribute we wish to query from the voxel block grid and flatten it
        attr = voxel_grid.attribute(attribute)
        attr = attr.reshape((-1,attr.shape[-1]))

        # we then reshape the index array for easier querying according to the voxel block resolution
        idx = indices.reshape((-1,res,res,res)).cpu().numpy()
        
        # finally, we find the corresponding memory flattened index of the voxels containing the query points, remembering that the slicing order
        # for the dense voxel block is z-y-x for some reason. 
        selected_idx = idx[mapping,qri[:,2],qri[:,1],qri[:,0]]
        # we do the same for the coordinates
        coords = coords.reshape((-1,res,res,res,3))
        selected_coords = coords[mapping,qri[:,2],qri[:,1],qri[:,0],:]
        #finally, we return the selected attributes for those indices, as weel as the coordinates of the voxels containing the query points
        return attr[selected_idx,:],selected_coords
    else: 
        return None,points

def get_indices_from_points(voxel_grid,points,res = 8,voxel_size = 0.025,device = o3d.core.Device('CUDA:0')):
    """ This function returns the indices of the points designated by points

    Args:
        voxel_grid (open3d.t.geometry.VoxelBlockGrid): The voxel block grid containing the attributes and coordinates you wish to extract
        points (np.array [Nx3]- dtype np.float32): array containing the XYZ coordinates of the points for which you wish to extract the attributes in global coordinates
        attribute (str): the string corresponding to the attribute you wish to obtain within your voxel_grid (say, semantic label, color, etc)
        res (int, optional): Resolution of the dense voxel blocks in your voxel block grid.  Defaults to 8.
        voxel_size (float, optional): side length of the voxels in the voxel_block_grid  Defaults to 0.025.
    """
    if(points.shape[0]>0):
        # we first find the coordinate of the origin of the voxel block of each query point and turn it into an open3d tensor
        query = np.floor((points/(res*voxel_size)))
        t_query = o3c.Tensor(query.astype(np.int32),device = device)

        # we then find the remainder between the voxel block origin and each point to find the within-block coordinates of the voxel containing this point
        query_remainder = points-query*res*voxel_size
        query_remainder_idx = np.floor(query_remainder/voxel_size).astype(np.int32)
        qri = query_remainder_idx

        # we then obtain the hashmap 
        hm = voxel_grid.hashmap()

        # we extract the unique voxel block origins for memory reasons and save the mapping for each individual entry
        block_c,mapping = np.unique(query,axis = 0,return_inverse = True)
        t_block_c = o3c.Tensor(block_c.astype(np.int32),device = device)

        # we query the hashmap to find the index corresponding to each of the voxel block origins
        r,m = hm.find(t_block_c)

        # we then find the flattened indices and coordinates of the individual voxels within each of these voxel blocks in memory
        coords,indices = voxel_grid.voxel_coordinates_and_flattened_indices(r.to(o3c.int32))

        # we then reshape the index array for easier querying according to the voxel block resolution
        idx = indices.reshape((-1,res,res,res)).cpu().numpy()
        
        # finally, we find the corresponding memory flattened index of the voxels containing the query points, remembering that the slicing order
        # for the dense voxel block is z-y-x for some reason. 
        selected_idx = idx[mapping,qri[:,2],qri[:,1],qri[:,0]]
    return selected_idx

class Reconstruction:

    def __init__(self,depth_scale = 1000.0,depth_max=5.0,res = 8,voxel_size = 0.025,trunc_multiplier = 8,n_labels = None,integrate_color = True,device = o3d.core.Device('CUDA:0'),miu = 0.001):
        """Initializes the TSDF reconstruction pipeline using voxel block grids, ideally using a GPU device for efficiency. 

        Args:
            depth_scale (float, optional): Describes the conversion factor of your depth image to meters - defaults to 1000:1 (i.e. each unit of depth is 1/1000 m). Defaults to 1000.0.
            depth_max (float, optional): Maximum depth reading in meters. Defaults to 5.0 m.
            res (int, optional): The number of voxels per locally connected block in the voxel block grid . Defaults to 8.
            voxel_size (float, optional): The size of the voxels in the voxel grid, in meters. Defaults to 0.025.
            n_labels (_type_, optional): Number of semantic labels in the semantic map. Leave as None if not doing metric-semantic reconstruction. When provided, performs to metric semantic reconstruction. Defaults to None.
            integrate_color (bool, optional): Whether or not to add color to the reconstructed mesh. If false, color informaton is not integrated. Defaults to True.
            device (_type_, optional): Which (CPU or GPU) you wish to use to performs the calculation. CUDA devices ~strongly~ encouraged for performance. Defaults to o3d.core.Device('CUDA:0').
            miu (float, optional): Laplace smoothing factor used to ensure numeric stability in metric-semantic reconstruction. Defaults to 0.001.
        """
        self.depth_scale = depth_scale
        self.depth_max = depth_max
        self.res = res
        self.voxel_size = voxel_size
        self.n_labels = n_labels
        self.integrate_color = integrate_color
        self.device = device
        self.semantic_integration = self.n_labels is not None
        self.miu = miu
        self.trunc = self.voxel_size * trunc_multiplier
        self.initialize_vbg()
        self.rays = None
        self.torch_device = torch.device('cuda')

    def initialize_vbg(self):
        if(self.integrate_color and (self.n_labels is None)):
            self.vbg = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight', 'color'),
            (o3c.float32, o3c.float32, o3c.float32), ((1), (1), (3)),
            self.voxel_size,self.res, 30000, self.device)
        elif((self.integrate_color == False) and (self.n_labels is not None)):
            self.vbg = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight', 'label'),
            (o3c.float32, o3c.float32, o3c.float32), ((1), (1), (self.n_labels)),
            self.voxel_size,self.res, 30000, self.device)
        elif((self.integrate_color) and (self.n_labels is not None)):
            self.vbg = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight','color','label'),
            (o3c.float32, o3c.float32, o3c.float32,o3c.float32), ((1), (1),(3),(self.n_labels)),
            self.voxel_size,self.res, 30000, self.device)
        else:
            print('No color or Semantics')
            self.vbg = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight'),
            (o3c.float32, o3c.float32), ((1), (1)),
            self.voxel_size,self.res, 30000, self.device)


    def update_vbg(self,depth,intrinsic,pose,color = None,semantic_label = None):
        """Adds a new observation to the metric (or metric-semantic) map

        Args:
            depth (ndarray - HxWx1): Depth image as a numpy array
            intrinsic (ndarray - 3x3): Intrinsic matrix of the depth camera (supposes the color image has the same intrinsics)
            pose (ndarray - 4x4 np.float64): The camera's transform w.r.t. the world frame
            color (ndarray - np.uint8 HxWx3, optional): The color image of the observation. Must be present if performing colored metric reconstruction. Defaults to None.
            semantic_label (ndarray - HxWxn_labels np.float32, optional): The current observed logits for the semantic segmentation of this map. Must be present if performing metric-semantic reconstruction_description_. Defaults to None.
        """
        self.depth = depth
        self.pose = pose

        if(self.rays is None):
            # print('calculating rays')
            self.rays =  torch.from_numpy(get_camera_rays(depth.shape[0],depth.shape[1],intrinsic[0,0],intrinsic[1,1])).to(self.torch_device)
        self.intrinsic = intrinsic
        intrinsic = o3c.Tensor(intrinsic.astype(np.float64))
        depth = o3d.t.geometry.Image(depth).to(self.device)
        extrinsic = se3.from_ndarray(pose)
        extrinsic = se3.ndarray(se3.inv(extrinsic))
        extrinsic = o3c.Tensor(extrinsic)#extrinsics[i]
        # Get active frustum block coordinates from input
        frustum_block_coords = self.vbg.compute_unique_block_coordinates(
            depth, intrinsic, extrinsic, self.depth_scale,
            self.depth_max)
        # Activate them in the underlying hash map (may have been inserted)
        self.vbg.hashmap().activate(frustum_block_coords)

        # Find buf indices in the underlying engine
        buf_indices, masks = self.vbg.hashmap().find(frustum_block_coords)
        o3d.core.cuda.synchronize()
        voxel_coords, voxel_indices = self.vbg.voxel_coordinates_and_flattened_indices(
            buf_indices)
        o3d.core.cuda.synchronize()

        # Now project them to the depth and find association
        # (3, N) -> (2, N)
        extrinsic_dev = extrinsic.to(self.device, o3c.float32)
        xyz = extrinsic_dev[:3, :3] @ voxel_coords.T() + extrinsic_dev[:3,
                                                                    3:]
        intrinsic_dev = intrinsic.to(self.device, o3c.float32)
        uvd = intrinsic_dev @ xyz
        d = uvd[2]
        u = (uvd[0] / d).round().to(o3c.int64)
        v = (uvd[1] / d).round().to(o3c.int64)
        o3d.core.cuda.synchronize()
        mask_proj = (d > 0) & (u >= 0) & (v >= 0) & (u < depth.columns) & (
            v < depth.rows)

        v_proj = v[mask_proj]
        u_proj = u[mask_proj]
        d_proj = d[mask_proj]
        depth_readings = depth.as_tensor()[v_proj, u_proj, 0].to(
            o3c.float32) / self.depth_scale
        sdf = depth_readings - d_proj

        mask_inlier = (depth_readings > 0) \
            & (depth_readings < self.depth_max) \
            & (sdf >= -self.trunc)

        sdf[sdf >= self.trunc] = self.trunc
        sdf = sdf / self.trunc
        o3d.core.cuda.synchronize()

        weight = self.vbg.attribute('weight').reshape((-1, 1))
        tsdf = self.vbg.attribute('tsdf').reshape((-1, 1))

        valid_voxel_indices = voxel_indices[mask_proj][mask_inlier]
        w = weight[valid_voxel_indices]
        wp = w + 1

        tsdf[valid_voxel_indices] \
            = (tsdf[valid_voxel_indices] * w +
            sdf[mask_inlier].reshape(w.shape)) / (wp)
        
        o3d.core.cuda.synchronize()

        if(self.integrate_color):

            self.update_color(color,depth,valid_voxel_indices,mask_inlier,w,wp,v_proj,u_proj)
        if(self.semantic_integration):
            self.update_semantics(semantic_label,v_proj,u_proj,valid_voxel_indices,mask_inlier,weight)
        weight[valid_voxel_indices] = wp
        o3d.core.cuda.synchronize()
        

    def update_color(self,color,depth,valid_voxel_indices,mask_inlier,w,wp,v_proj,u_proj):
        #performing color integration
        color = cv2.resize(color,(depth.columns,depth.rows),interpolation= cv2.INTER_NEAREST)
        color = o3d.t.geometry.Image(o3c.Tensor(color.astype(np.float32))).to(self.device)
        color_readings = color.as_tensor()[v_proj,u_proj].to(o3c.float32)
        color = self.vbg.attribute('color').reshape((-1, 3))
        color[valid_voxel_indices] \
                = (color[valid_voxel_indices] * w +
                            color_readings[mask_inlier]) / (wp)
        o3d.core.cuda.synchronize()
        
    def update_semantics(self,semantic_label,v_proj,u_proj,valid_voxel_indices,mask_inlier,weight):
        # performing semantic integration
        #  Laplace Smoothing of the observation
        semantic_label += self.miu
        renormalizer = 1+self.miu*self.n_labels
        semantic_label = semantic_label/renormalizer

        semantic_label = np.log(semantic_label)

        semantic_image = o3d.t.geometry.Image(semantic_label).to(self.device)
        
        semantic_readings = semantic_image.as_tensor()[v_proj,
                                        u_proj].to(o3c.float32)
        semantic = self.vbg.attribute('label').reshape((-1, self.n_labels))
        # initializing previously unobserved voxels with uniform prior
        semantic[valid_voxel_indices[weight[valid_voxel_indices].flatten() == 0]] += o3c.Tensor(np.log(np.array([1.0/self.n_labels])).astype(np.float32)).to(self.device)
        #Bayesian update in log space    
        semantic[valid_voxel_indices] = semantic[valid_voxel_indices]+semantic_readings[mask_inlier]
        o3d.core.cuda.synchronize()

    def extract_point_cloud(self,return_raw_logits = False):

        """Returns the current (colored) point cloud and the current probability estimate for each of the points, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.t.geometry.PointCloud, np.array(N_points,n_labels) (or None)
        """
        pcd = self.vbg.extract_point_cloud()
        pcd = pcd.to_legacy()
        sm = nn.Softmax(dim = 1)
        target_points = np.asarray(pcd.points)
        if(self.semantic_integration):
            labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            labels = labels.cpu().numpy().astype(np.float64)
            if labels is not None:
                if(return_raw_logits):
                    return pcd,labels
                else:
                    labels = labels
                    labels = sm(torch.from_numpy(labels)).numpy()
                    return pcd,labels
            else:
                return None,None
        else:
            return pcd,None

    def extract_triangle_mesh(self):
        """Returns the current (colored) mesh and the current probability for each class estimate for each of the vertices, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.geometry.TriangleMesh, np.array(N_vertices,n_labels) (or None)
        """
        mesh = self.vbg.extract_triangle_mesh()
        mesh = mesh.to_legacy()
        sm = nn.Softmax(dim =1)
        if(self.semantic_integration):
            target_points = np.asarray(mesh.vertices)
            labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            labels = labels.cpu().numpy().astype(np.float64)
            # labels = self.precision_check(labels)
            vertex_labels = sm(torch.from_numpy(labels)).numpy()
            #getting the correct probabilities
            return mesh,vertex_labels
        else:
            return mesh,None
    def precision_check(self,labels):
        #compensating for machine precision of exponentials by setting the maximum log to 0 
        labels += -labels.max(axis = 1).reshape(-1,1)
        return labels
    def save_vbg(self,path):
        self.vbg.save(path)

class GroundTruthGenerator(Reconstruction):
    def __init__(self,depth_scale = 1000.0,depth_max=5.0,res = 8,voxel_size = 0.025,trunc_multiplier = 8,n_labels = None,integrate_color = True,device = o3d.core.Device('CUDA:0'),miu = 0.001):
        super().__init__(depth_scale,depth_max,res,voxel_size,trunc_multiplier,n_labels,integrate_color,device,miu)

    def update_semantics(self,semantic_label,v_proj,u_proj,valid_voxel_indices,mask_inlier,weight):
        "takes in the GT mask resized to the depth image size"
        # now performing semantic integration
        # semantic_label = cv2.resize(data_dict['semantic_label'],(depth.columns,depth.rows),interpolation= cv2.INTER_NEAREST)
        # semantic_label = model.classify(data_dict['color'],data_dict['depth'])
        # cv2.resize(semantic_label,(depth.columns,depth.rows),interpolation= cv2.INTER_NEAREST)
        # print(np.max(semantic_label),np.min(semantic_label))
        # one-hot encoding semantic label
        semantic_label = torch.nn.functional.one_hot(torch.from_numpy(semantic_label.astype(np.int64)),num_classes = self.n_labels).numpy().astype(np.float32)
        #  Laplace Smoothing #1
        color = o3d.t.geometry.Image(semantic_label).to(self.device)
        
        color_readings = color.as_tensor()[v_proj,
                                        u_proj].to(o3c.float32)
        color = self.vbg.attribute('label').reshape((-1, self.n_labels))
        # Detection Count update
        color[valid_voxel_indices]  = color[valid_voxel_indices]+color_readings[mask_inlier]
    def extract_triangle_mesh(self,return_raw_logits = False):
        """Returns the current (colored) mesh and the current probability for each class estimate for each of the vertices, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.geometry.TriangleMesh, np.array(N_vertices,n_labels) (or None)
        """
        mesh = self.vbg.extract_triangle_mesh()
        mesh = mesh.to_legacy()
        if(self.semantic_integration):
            target_points = np.asarray(mesh.vertices)
            labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            labels = labels.cpu().numpy().astype(np.float64)
            #getting the correct probabilities
            if(return_raw_logits):
                vertex_labels = labels
            else:
                vertex_labels = labels/labels.sum(axis=1,keepdims = True)
            return mesh,vertex_labels
        else:
            return mesh,None
    def extract_point_cloud(self,return_raw_logits = False):

        """Returns the current (colored) point cloud and the current probability estimate for each of the points, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.t.geometry.PointCloud, np.array(N_points,n_labels) (or None)
        """
        pcd = self.vbg.extract_point_cloud()
        pcd = pcd.to_legacy()
        target_points = np.asarray(pcd.points)
        if(self.semantic_integration):
            labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            if labels is not None:
                labels = labels.cpu().numpy().astype(np.float64)
                labels[labels.sum(axis =1)==0] = 1/21.0

                # pdb.set_trace()
                if(return_raw_logits):
                    labels = labels
                else:
                    labels = labels/labels.sum(axis=1,keepdims = True)
                return pcd,labels
            else:
                return None,None
        else:
            return pcd,None



class ProbabilisticAveragedReconstruction(Reconstruction):

    def initialize_vbg(self):
        if(not self.integrate_color):
            self.vbg = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight','label','semantic_weight'),
            (o3c.float32, o3c.float32, o3c.float32,o3c.float32), ((1), (1), (self.n_labels),(1)),
            self.voxel_size,self.res, 30000, self.device)
            self.original_size = self.vbg.attribute('label').shape[0]
        else:
            self.vbg = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight','label','semantic_weight','color'),
            (o3c.float32, o3c.float32, o3c.float32,o3c.float32,o3c.float32), ((1), (1), (self.n_labels),(1),(3)),
            self.voxel_size,self.res, 30000, self.device)
            self.original_size = self.vbg.attribute('label').shape[0]

    def update_semantics(self, semantic_label, v_proj, u_proj, valid_voxel_indices, mask_inlier, weight):


        semantic_label = semantic_label

        semantic_image = o3d.t.geometry.Image(semantic_label).to(self.device)
        
        semantic_readings = semantic_image.as_tensor()[v_proj,
                                        u_proj].to(o3c.float32)
        semantic = self.vbg.attribute('label').reshape((-1, self.n_labels))
        semantic_weight = self.vbg.attribute('semantic_weight').reshape((-1))
        # initializing previously unobserved voxels with uniform prior
        #naive summing of probabilities
        semantic[valid_voxel_indices[weight[valid_voxel_indices].flatten() == 0]] += o3c.Tensor(np.array([1.0/self.n_labels]).astype(np.float32)).to(self.device)
        semantic_weight[valid_voxel_indices[weight[valid_voxel_indices].flatten() == 0]] += o3c.Tensor(1).to(o3c.float32).to(self.device)

        #Bayesian update in log space    
        semantic[valid_voxel_indices] = (semantic_weight[valid_voxel_indices].reshape((-1,1))*semantic[valid_voxel_indices]+semantic_readings[mask_inlier])/(semantic_weight[valid_voxel_indices].reshape((-1,1))+1)
        semantic_weight[valid_voxel_indices] += 1
        o3d.core.cuda.synchronize()



    def extract_point_cloud(self,return_raw_logits = False):

        """Returns the current (colored) point cloud and the current probability estimate for each of the points, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.t.geometry.PointCloud, np.array(N_points,n_labels) (or None)
        """
        pcd = self.vbg.extract_point_cloud()
        pcd = pcd.to_legacy()
        target_points = np.asarray(pcd.points)
        if(self.semantic_integration):
            labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)

            if labels is not None:
                if(return_raw_logits):
                    return pcd,labels.cpu().numpy().astype(np.float64)
                else:
                    labels = labels.cpu().numpy().astype(np.float64)
                    return pcd,labels
            else:
                return None,None
        else:
            return pcd,None

    def extract_triangle_mesh(self):
        """Returns the current (colored) mesh and the current probability for each class estimate for each of the vertices, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.geometry.TriangleMesh, np.array(N_vertices,n_labels) (or None)
        """
        mesh = self.vbg.extract_triangle_mesh()
        mesh = mesh.to_legacy()
        if(self.semantic_integration):
            target_points = np.asarray(mesh.vertices)
            labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            labels = labels.cpu().numpy().astype(np.float64)
            vertex_labels = labels
            return mesh,vertex_labels
        else:
            return mesh,None


class HistogramReconstruction(GroundTruthGenerator):
    def update_semantics(self, semantic_label, v_proj, u_proj, valid_voxel_indices, mask_inlier, weight):
        semantic_label = np.argmax(semantic_label,axis = 2)
        # semantic_label = torch.nn.functional.one_hot(torch.from_numpy(semantic_label.astype(np.int64)),num_classes = self.n_labels).numpy().astype(np.float32)
    #  Laplace Smoothing #1
        semantic_image =  o3d.t.geometry.Image(semantic_label).to(self.device)
                
        semantic_readings = semantic_image.as_tensor()[v_proj,
                                                u_proj].to(o3c.int64)
        # pdb.set_trace()
        semantic = self.vbg.attribute('label').reshape((-1, self.n_labels))
        # initializing previously unobserved voxels with uniform prior

        # updating the histogram
        semantic[valid_voxel_indices,semantic_readings[mask_inlier].flatten()] = semantic[valid_voxel_indices,semantic_readings[mask_inlier].flatten()]+1
        
        o3d.core.cuda.synchronize()

class GeometricBayes(Reconstruction):
    def extract_point_cloud(self,return_raw_logits = False):

        """Returns the current (colored) point cloud and the current probability estimate for each of the points, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.t.geometry.PointCloud, np.array(N_points,n_labels) (or None)
        """
        pcd = self.vbg.extract_point_cloud()
        pcd = pcd.to_legacy()
        target_points = np.asarray(pcd.points)
        sm = nn.Softmax(dim = 1)
        if(self.semantic_integration):
            
            labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            labels = labels.cpu().numpy().astype(np.float64)
            weights,coords= get_properties(self.vbg,target_points,'weight',res = self.res,voxel_size = self.voxel_size,device = self.device)
            weights = weights.cpu().numpy().astype(np.float64)
            if labels is not None:
                if(return_raw_logits):
                    return pcd,(labels/weights)
                else:
                    labels = (labels/weights)
                    labels[np.isnan(labels)] = 1
                    labels = sm(torch.from_numpy(labels)).numpy()
                    #getting the correct probabilities
                    return pcd,labels
            else:
                return None,None
        else:
            return pcd,None

    def extract_triangle_mesh(self):
        """Returns the current (colored) mesh and the current probability for each class estimate for each of the vertices, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.geometry.TriangleMesh, np.array(N_vertices,n_labels) (or None)
        """
        mesh = self.vbg.extract_triangle_mesh()
        mesh = mesh.to_legacy()
        sm = nn.Softmax(dim=1)
        if(self.semantic_integration):
            target_points = np.asarray(mesh.vertices)
            labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            weights,coords= get_properties(self.vbg,target_points,'weight',res = self.res,voxel_size = self.voxel_size,device = self.device)
            labels = (labels/weights).cpu().numpy()
            labels[np.isnan(labels)] = 1
            labels = sm(torch.from_numpy(labels)).numpy()
            vertex_labels = labels
            return mesh,vertex_labels
        else:
            return mesh,None


class GeneralizedIntegration(Reconstruction):
    def __init__(self, depth_scale=1000, depth_max=5, res=8, voxel_size=0.025, trunc_multiplier=8, n_labels=None, integrate_color=True, device=o3d.core.Device('CUDA:0'), 
                 miu=0.001,epsilon = 0,L = 0,torch_device = 'cuda:0',T=np.array(1)):
        super().__init__(depth_scale, depth_max, res, voxel_size, trunc_multiplier, n_labels, integrate_color, device, miu)
        self.epsilon = epsilon
        self.L = L
        self.torch_device = torch_device
        self.T = torch.from_numpy(T).to(self.torch_device)
        self.sm = nn.Softmax(dim = 2)
    def initialize_vbg(self):
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
        ('tsdf', 'weight', 'log_label','label','semantic_weight'),
        (o3c.float32, o3c.float32, o3c.float32,o3c.float32,o3c.float32), ((1), (1), (self.n_labels),(self.n_labels),(1)),
        self.voxel_size,self.res, 30000, self.device)
        self.original_size = self.vbg.attribute('label').shape[0]
    def get_epsilon_and_L(self,semantic_label=0,semantic_readings=0):
        epsilon = o3c.Tensor([self.epsilon]).to(self.device)
        L = o3c.Tensor([self.L]).to(self.device)
        return epsilon.reshape((1,1)).to(o3c.float32),L.reshape((1,1)).cpu().numpy()
    def get_weights(self,semantic_label,semantic_readings,v_proj,u_proj,valid_voxel_indices,mask_inlier):
        w = o3c.Tensor(np.ones(shape = valid_voxel_indices.shape)).to(o3c.float32).to(self.device)
        # print(w.shape)
        return w.reshape((-1,1))
    def get_temperatures(self,semantic_label):
        return self.T.view(1,1,-1)
    def update_semantics(self, semantic_label, v_proj, u_proj, valid_voxel_indices, mask_inlier, weight):
        new_size = self.vbg.attribute('label').shape[0]
        if(new_size != self.original_size):
            print('VBG size changed from {} to {}'.format(self.original_size,new_size))
            self.original_size = new_size

        # Scaling Step
        T = self.get_temperatures(semantic_label)

        semantic_label = torch.from_numpy(semantic_label).to(self.torch_device)
        semantic_label = self.sm(semantic_label/T)
        # Laplace smoothing step
        semantic_label += self.miu
        renormalizer = 1+self.miu*self.n_labels
        semantic_label = semantic_label/renormalizer
        semantic_label_l = torch.log(semantic_label)
        semantic_label_l = semantic_label_l.cpu().numpy()
        semantic_label = semantic_label.cpu().numpy()

        # projection step
        semantic_image = o3d.t.geometry.Image(semantic_label).to(self.device)
        semantic_image_l = o3d.t.geometry.Image(semantic_label_l).to(self.device)
        
        semantic_readings = semantic_image.as_tensor()[v_proj,
                                        u_proj].to(o3c.float32)
        semantic_readings_l = semantic_image_l.as_tensor()[v_proj,u_proj].to(o3c.float32)

        semantic = self.vbg.attribute('label').reshape((-1, self.n_labels))
        semantic_l = self.vbg.attribute('log_label').reshape((-1, self.n_labels))
        semantic_weights = self.vbg.attribute('semantic_weight').reshape((-1))

        W = self.get_weights(semantic_label,semantic_readings,v_proj,u_proj,valid_voxel_indices,mask_inlier)
        # epsilon,L = self.get_epsilon_and_L(semantic_label,semantic_readings)
        o3d.core.cuda.synchronize()

        # initializing previously unobserved voxels with uniform prior
        # for log operations
        new_voxels = valid_voxel_indices[weight[valid_voxel_indices].flatten() == 0]
        semantic_l[new_voxels] += o3c.Tensor(np.log(np.array([1.0/self.n_labels])).astype(np.float32)).to(self.device)
        # for averaging
        semantic[new_voxels] += o3c.Tensor(np.array([1.0/self.n_labels]).astype(np.float32)).to(self.device)
        # initializing weights to one:
        semantic_weights[new_voxels] = o3c.Tensor(np.array([1.0])).to(o3c.float32).to(self.device)
        o3d.core.cuda.synchronize()

        #Bayesian update in log space    
        semantic_l[valid_voxel_indices] = semantic_l[valid_voxel_indices]+W*semantic_readings_l[mask_inlier]
        o3d.core.cuda.synchronize()

        #Bayesian update in non-log space
        # print(semantic_readings[mask_inlier].cpu().numpy().isnull())
        semantic[valid_voxel_indices] = semantic[valid_voxel_indices]+W*semantic_readings[mask_inlier]
        o3d.core.cuda.synchronize()

        #Updating the Semantic Weights:
        # print(semantic_weights[valid_voxel_indices])
        semantic_weights[valid_voxel_indices] = semantic_weights[valid_voxel_indices] +  W.reshape((-1))
        o3d.core.cuda.synchronize()
        o3d.core.cuda.release_cache()
        # return super().update_semantics(semantic_label, v_proj, u_proj, valid_voxel_indices, mask_inlier, weight)

    def extract_point_cloud(self,return_raw_logits = False):

        """Returns the current (colored) point cloud and the current probability estimate for each of the points, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.t.geometry.PointCloud, np.array(N_points,n_labels) (or None)
        """
        pcd = self.vbg.extract_point_cloud()
        pcd = pcd.to_legacy()
        sm = nn.Softmax(dim = 1)
        target_points = np.asarray(pcd.points)
        if(self.semantic_integration):
            labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            labels_l,coords = get_properties(self.vbg,target_points,'log_label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            semantic_weights,coords = get_properties(self.vbg,target_points,'semantic_weight',res = self.res,voxel_size = self.voxel_size,device = self.device)
            epsilon,L = self.get_epsilon_and_L()
            alpha = (1-epsilon)/(semantic_weights.reshape((-1,1))) + epsilon
            alpha = alpha.cpu().numpy()
            labels = labels.cpu().numpy().astype(np.float64)
            labels_l = labels_l.cpu().numpy().astype(np.float64)   
            if labels is not None:
                if(return_raw_logits):
                    return pcd,labels,labels_l,semantic_weights.cpu().numpy()
                else:
                    labels = alpha*labels
                    labels_l = alpha*labels
                    l_probs = sm(torch.from_numpy(labels_l)).numpy()
                    probs = labels/labels.sum(axis = 1,keepdims = True)
                    final_probs = L*l_probs+(1-L)*probs
                    final_probs = final_probs/final_probs.sum(axis =1,keepdims = True)

                    return pcd,final_probs
            else:
                return None,None
        else:
            return pcd,None

    def extract_triangle_mesh(self):
        """Returns the current (colored) mesh and the current probability for each class estimate for each of the vertices, if performing metric-semantic reconstruction

        Returns:
            open3d.cpu.pybind.geometry.TriangleMesh, np.array(N_vertices,n_labels) (or None)
        """
        mesh = self.vbg.extract_triangle_mesh()
        mesh = mesh.to_legacy()
        sm = nn.Softmax(dim =1)
        if(self.semantic_integration):
            target_points = np.asarray(mesh.vertices)
            labels,coords = get_properties(self.vbg,target_points,'label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            log_labels,coords = get_properties(self.vbg,target_points,'log_label',res = self.res,voxel_size = self.voxel_size,device = self.device)
            semantic_weights,coords = get_properties(self.vbg,target_points,'semantic_weight',res = self.res,voxel_size = self.voxel_size,device = self.device)
            epsilon,L = self.get_epsilon_and_L()
            alpha = (1-epsilon)/(semantic_weights) + epsilon
            alpha = alpha.cpu().numpy()
            labels = labels.cpu().numpy().astype(np.float64)
            labels_l = labels_l.cpu().numpy().astype(np.float64)   

            if labels is not None:
                labels = alpha*labels
                labels_l = alpha*labels
                l_probs = sm(torch.from_numpy(labels_l)).numpy()
                probs = labels/labels.sum(axis = 1,keepdims = True)
                final_probs = L*l_probs+(1-L)*probs
                final_probs = final_probs/final_probs.sum(axis =1,keepdims = True)

                return mesh,final_probs
        else:
            return mesh,None

class LearnedGeneralizedIntegration(GeneralizedIntegration):
    def __init__(self, depth_scale=1000, depth_max=5, res=8, voxel_size=0.025, trunc_multiplier=8, n_labels=None, integrate_color=True, device=o3d.core.Device('CUDA:0'),
                  miu=0.001, epsilon=0, L=0, torch_device='cuda:0', T=np.array(1),weights = np.array(1),depth_ranges = np.arange(0.0,5.1,0.5),angle_ranges =  np.arange(0,90.1,30)):
        super().__init__(depth_scale, depth_max, res, voxel_size, trunc_multiplier, n_labels, integrate_color, device, miu, epsilon, L, torch_device, T)
        self.weights = weights
        self.weights[self.weights <0] = 0
        self.depth_ranges = depth_ranges
        self.angle_ranges = angle_ranges

    def compute_and_digitize(self,rendered_depth_1,n1):
        with torch.no_grad():
            device = self.torch_device
            this_rays = self.rays
            dr = torch.from_numpy(self.depth_ranges).to(device)
            ar = torch.from_numpy(self.angle_ranges).to(device)
            n = torch.from_numpy(n1).to(device)
            # pdb.set_trace()

            rendered_depth = torch.from_numpy(rendered_depth_1).to(device)
            n = torch.clamp(n,-1,1)
            n[torch.all(n == 0,dim = 2)] = torch.Tensor([0,0,1.0]).to(device)
            digitized_depth = torch.clamp(torch.bucketize(rendered_depth[:,:,0].float()/1000,dr),0,dr.shape[0]-2)
            p = (n.view(-1,3)*this_rays).sum(axis = 1)
            p = torch.clamp(p/(torch.linalg.norm(n.view(-1,3),dim =1)*torch.linalg.norm(this_rays,dim =1)),-1,1)
            projective_angle = torch.arccos((torch.abs(p)))*180/np.pi
            # print('this is ar shape = {} and ar = {}'.format(ar.shape[0]-1,ar))
            angle_proj = torch.clamp(torch.bucketize(projective_angle,ar).reshape(digitized_depth.shape),0,ar.shape[0]-2)
            # pdb.set_trace()

            del dr 
            del ar 
            del n
            del p
            del rendered_depth

            return digitized_depth.cpu().numpy(),angle_proj.cpu().numpy() 

    def get_weights(self,semantic_label,semantic_readings,v_proj_o3d,u_proj_o3d,valid_voxel_indices,mask_inlier_03d):
        sl = semantic_label.argmax(axis=2)
        v_proj = v_proj_o3d.cpu().numpy()
        u_proj = u_proj_o3d.cpu().numpy()
        mask_inlier = mask_inlier_03d.cpu().numpy()
        rendered_depth,n = render_depth_and_normals(self.vbg,self.depth,self.intrinsic,self.pose,device = self.device,use_depth = True)
        digitized_depth,digitized_angle = self.compute_and_digitize(rendered_depth,n)
        # pdb.set_trace()
        selected_weights = self.weights[sl[v_proj,u_proj],digitized_depth[v_proj,u_proj],digitized_angle[v_proj,u_proj]]
        return o3c.Tensor(selected_weights).to(self.device).reshape((-1,1)).to(o3c.float32)[mask_inlier]


def main():
    voxel_size = 0.025
    trunc = voxel_size * 8
    res = 8
    depth_scale = 1000.0
    depth_max = 5.0
    root_dir = "/home/motion/data/scannet_v2"
    color = False
    data = pickle.load(open('example_data_reconstruction.p','rb'))

    ### Performing metric reconstruction only
    rec = Reconstruction(depth_scale = 1000.0,depth_max=5.0,res = 8,voxel_size = 0.025,n_labels = None,integrate_color = False,
        device = o3d.core.Device('CUDA:0'),miu = 0.001)
    
    for data_dict in tqdm(data):
        depth = data_dict['depth']
        rec.update_vbg(data_dict['depth'],data_dict['intrinsics_depth'][:3,:3],data_dict['pose'],data_dict['color'],data_dict['semantic_label'])
        # pcd,_ = rec.extract_point_cloud()
    mesh,label = rec.extract_triangle_mesh()
    o3d.visualization.draw_geometries([mesh])



    ### Performing Colored Metric reconstruction
    rec = Reconstruction(depth_scale = 1000.0,depth_max=5.0,res = 8,voxel_size = 0.025,n_labels = None,integrate_color = True,
        device = o3d.core.Device('CUDA:0'),miu = 0.001)
    
    total_len = len(data)

    for data_dict in tqdm(data):
        depth = data_dict['depth']
        rec.update_vbg(data_dict['depth'],data_dict['intrinsics_depth'][:3,:3],data_dict['pose'],data_dict['color'],data_dict['semantic_label'])
    mesh,label = rec.extract_triangle_mesh()
    o3d.visualization.draw_geometries([mesh])




    del rec
    ### performing colored metric-semantic reconstruction
    rec = Reconstruction(depth_scale = 1000.0,depth_max=5.0,res = 8,voxel_size = 0.025,n_labels = 21,integrate_color = True,
        device = o3d.core.Device('CUDA:0'),miu = 0.001)
    
    for data_dict in tqdm(data):
        depth = data_dict['depth']
        rec.update_vbg(data_dict['depth'],data_dict['intrinsics_depth'][:3,:3],data_dict['pose'],data_dict['color'],data_dict['semantic_label'])
    mesh,label = rec.extract_triangle_mesh()
    o3d.visualization.draw_geometries([mesh])
    print(label,label.shape)



if __name__ == '__main__':
    main()



