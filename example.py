import pickle
from reconstruction import Reconstruction
import open3d as o3d
from tqdm import tqdm

def main():
    voxel_size = 0.025
    trunc = voxel_size * 8
    res = 8
    depth_scale = 1000.0
    depth_max = 5.0
    n_labels = 21
    data = pickle.load(open('example_data_reconstruction.p','rb'))

    ### Performing metric reconstruction only
    rec = Reconstruction(depth_scale = depth_scale,depth_max=depth_max,res = res,voxel_size = voxel_size,n_labels = None,integrate_color = False,
        device = o3d.core.Device('CUDA:0'),miu = 0.001)
    
    for data_dict in tqdm(data):
        rec.update_vbg(data_dict['depth'],data_dict['intrinsics_depth'][:3,:3],data_dict['pose'],data_dict['color'],data_dict['semantic_label'])
        # pcd,_ = rec.extract_point_cloud()
    mesh,label = rec.extract_triangle_mesh()
    o3d.visualization.draw_geometries([mesh])



    ### Performing Colored Metric reconstruction
    rec = Reconstruction(depth_scale = depth_scale,depth_max=depth_max,res = res,voxel_size = voxel_size,n_labels = None,integrate_color = True,
        device = o3d.core.Device('CUDA:0'),miu = 0.001)
    
    for data_dict in tqdm(data):
        rec.update_vbg(data_dict['depth'],data_dict['intrinsics_depth'][:3,:3],data_dict['pose'],data_dict['color'],data_dict['semantic_label'])
    mesh,label = rec.extract_triangle_mesh()
    o3d.visualization.draw_geometries([mesh])




    del rec
    ### performing colored metric-semantic reconstruction
    rec = Reconstruction(depth_scale = depth_scale,depth_max=depth_max,res = res,voxel_size = voxel_size,n_labels = n_labels,integrate_color = True,
        device = o3d.core.Device('CUDA:0'),miu = 0.001)
    
    for data_dict in tqdm(data):
        rec.update_vbg(data_dict['depth'],data_dict['intrinsics_depth'][:3,:3],data_dict['pose'],data_dict['color'],data_dict['semantic_label'])
    mesh,label = rec.extract_triangle_mesh()
    o3d.visualization.draw_geometries([mesh])
    print(label,label.shape)



if __name__ == '__main__':
    main()