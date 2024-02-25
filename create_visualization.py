import os
import numpy as np
import open3d as o3d
import open3d.core as o3c
import time
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from tqdm import tqdm
import pickle
import pickle
from glob import glob
from copy import deepcopy
from scipy.stats import entropy
import time
import torch.nn as nn
import torch
import pdb
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scene_definitions import get_larger_test_and_validation_scenes,get_filenames
import torch.nn as nn
from pcd_utils import read_alignment
import matplotlib.cm as cm
import json
import open3d.visualization.rendering as rendering

val_scenes,test_scenes = get_larger_test_and_validation_scenes()
selected_scenes = test_scenes
fnames = get_filenames()
a = json.load(open('./experiments_and_short_names.json','r'))
results_dir = fnames['results_dir']

experiments = a['experiments']
short_names = a['short_names']


interactive = False
visualize = False
save = True

COLORS = np.array([
    [14,14,63],[151,226,173],[174,198,232],[31,120,180],[255,188,120],[188,189,35],[140,86,74],[255,152,151],[213,39,40],[196,176,213],[148,103,188],[196,156,148],
    [23,190,208],[247,183,210],[218,219,141],[254,127,14],[227,119,194],[158,218,229],[43,160,45],[112,128,144],[82,83,163]
])

if(not interactive):
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    pass
def align_pcds(pcd_1,pcd_2):
    a  = o3d.geometry.OrientedBoundingBox()
    for i in range(5):
        tmp = a.create_from_points(pcd_1.points)
        R = tmp.R
        center = tmp.get_center()
        pcd_1 = pcd_1.rotate(R,-center)
        pcd_2 = pcd_2.rotate(R,center)
    return pcd_1,pcd_2

render = rendering.OffscreenRenderer(1024, 1024)

def to_color_map(array):
    a = np.array([1,0,0])
    b = np.array([0,0,0])
    # color = array.reshape(-1,1)*a + (1-array).reshape(-1,1)*b
    SM = plt.cm.ScalarMappable(cmap='coolwarm', norm=mynorm)
    SM.set_array([0,1])
    SM.set_clim(-0.01,1.01)

    SM.autoscale()
    color = SM.to_rgba(array.flatten())[:,:3]
    # print(color)
    return color

for experiment,exp_name in zip(experiments,short_names):
    print(experiment,exp_name)
    sm = nn.Softmax(dim = 1)
    pcds_template = '{}/{}/{}/*.pcd'
    labels_template = '{}/{}/{}/*.p'
    max_e = 3.0
    limiter = -1
    pcd_colors = []
    entropies = []
    root_dir = "/scratch/bbuq/jcorreiamarques/3d_calibration/scannet_v2/"
    save_dir = '{}/pred_vis/'.format(results_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    exp_save_dir = '{}{}'.format(save_dir,exp_name)
    if not os.path.exists(exp_save_dir):
        os.mkdir(exp_save_dir)


    

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    for index,scene in enumerate(selected_scenes):
        pcd_colors = []
        gt_pcd_file = '{}/sanity_checks/gt_pcd_{}.pcd'.format(results_dir,scene)
        gt_labels_file = '{}/sanity_checks/gt_labels_{}.p'.format(results_dir,scene)
        gt_pcd = o3d.io.read_point_cloud(gt_pcd_file)
        gt_labels= pickle.load(open(gt_labels_file,'rb'))
        counts = gt_labels.sum(axis = 1).reshape(-1,1)
        bin_counts = np.digitize(counts,np.quantile(counts,np.linspace(0,1,100,endpoint = False)),right = True)/100
        pcd_files = sorted(glob(pcds_template.format(results_dir,experiment,scene)))
        label_files = sorted(glob(labels_template.format(results_dir,experiment,scene)))
        pcd_file = pcd_files[-1]
        labels_file = label_files[-1]
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd2 = o3d.io.read_point_cloud(pcd_file)
        pcd3 = o3d.io.read_point_cloud(pcd_file)
        pcd4 = o3d.io.read_point_cloud(pcd_file)
        pcd5 = o3d.io.read_point_cloud(pcd_file)
        # if(index == 0):
        #     vis.add_geometry(gt_pcd)
        #     vis.add_geometry(pcd)
        #     vis.add_geometry(pcd2)
        #     vis.add_geometry(pcd3)

        labels= pickle.load(open(labels_file,'rb')).astype(np.float64)
        # pdb.set_trace
        # pdb.set_trace()
        if(np.any(labels.sum(axis = 1)<0)):
            labels = sm(torch.from_numpy(labels)).cpu().numpy()
        # we then unscramble the labels:
        if(np.any(labels.sum(axis =1)>0)):
            labels = labels/labels.sum(axis =1,keepdims = True)
        # print(labels)

        pcd_tree = o3d.geometry.KDTreeFlann(gt_pcd)
        points = np.asarray(pcd.points)
        gt_points = np.asarray(gt_pcd.points)
        unscrambler = np.zeros(points.shape[0]).astype(np.int64)
        scrambler = np.zeros(points.shape[0]).astype(np.int64)
        for i in tqdm(range(points.shape[0])):
            [k, idx, dist] = pcd_tree.search_knn_vector_3d(points[i],1)
        #     print(points[i]-gt_points[idx],i,idx)
            unscrambler[idx[0]] = i
            scrambler[i] = idx[0]
        diff_labels = labels[unscrambler]
        # cm = confusion_matrix(np.argmax(gt_labels,axis = 1),np.argmax(diff_labels,axis=1),normalize = 'true',labels = list(range(21)))
        a = np.zeros((labels.shape[0],3))

        classes = np.argmax(labels,axis = 1)
        # print(np.unique(classes,return_counts = True))
        colors = COLORS[classes].astype(float)/255
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # setting up the colormap for all the plots        
        mynorm = plt.Normalize(vmin=-0.05, vmax=1.01)

        # 255*sm.to_rgba([1.0])
        # sm.set_clim(-0.01,1.01)
        a = np.zeros((gt_labels.shape[0],3))

        classes = np.argmax(gt_labels,axis = 1)
        # print(np.unique(classes,return_counts = True))
        colors = COLORS[classes].astype(float)/255
        gt_pcd.colors = o3d.utility.Vector3dVector(colors)
        print(scene)
        # o3d.visualization.draw([gt_pcd,pcd_translated])

        # now showing differences:
        a = np.zeros((labels.shape[0],3))
        classes = np.zeros(gt_labels.shape[0]).astype(np.int64)
        classes[unscrambler] = (np.argmax(diff_labels,axis = 1) == np.argmax(gt_labels,axis = 1)).astype(int)
        # print(np.unique(classes,return_counts = True))
        differences_coloring = np.array([[255,0,0],[0,0,0]])
        # colors = differences_coloring[classes].astype(float)/255
        colors = to_color_map(1-classes)
        # colors[np.argmax(gt_labels[scrambler],axis = 1) == 0,:] = [0,0,0]

        # pdb.set_trace()
        pcd2.colors = o3d.utility.Vector3dVector(colors)
        # this_entropy = entropy(labels,axis = 1,base = 2)
        this_entropy = 1-np.max(labels,axis = 1)
        this_correlation = np.clip(np.abs(classes - this_entropy),0,1)
        # pdb.set_trace()

 

        # pdb.set_trace()
        max_ent = 1
        # this_color = np.array([255,0,0])*(this_entropy/max_ent).reshape(-1,1) + (1-this_entropy/max_ent).reshape(-1,1)*np.array([0,255,0])
        # this_color = this_color.astype(np.float64)/255
        #this_color = #sm.to_rgba(this_entropy)[:,:3]
        this_color = to_color_map(np.clip(this_entropy/0.5,0,1.0))
        pcd3.colors = o3d.utility.Vector3dVector(this_color)

        # count_color = np.array([255,0,0])*(1-bin_counts).reshape(-1,1) + (bin_counts).reshape(-1,1)*np.array([0,255,0])
        # count_color = count_color.astype(np.float64)/255.0
        count_color = to_color_map(bin_counts.flatten())[:,:3]
        # pdb.set_trace()
        count_color = count_color[scrambler]
        pcd4.colors = o3d.utility.Vector3dVector(count_color)
        # bucketized_correlation = np.digitize(this_correlation,np.linspace(0,1,256,endpoint = True)).astype(np.float32)/256
        # pdb.set_trace()





        correlation_colors =to_color_map((1-this_correlation.flatten()))[:,:3]

        pcd5.colors = o3d.utility.Vector3dVector(correlation_colors)

        a  = o3d.geometry.OrientedBoundingBox()
        tmp = a.create_from_points(gt_pcd.points)
        R = tmp.R
        tmp.R
        center = tmp.get_center()

        tmp2 = tmp.get_axis_aligned_bounding_box()
        res = tmp2.get_max_bound()-tmp2.get_min_bound()
        res = res[:2]    
        translation = np.zeros(3)
        translation[res.argmin()] = res.min() + 0.1
        non_null = gt_labels.argmax(axis = 1) != 0
        scrambled_non_null = deepcopy(non_null)
        scrambled_non_null[unscrambler] = non_null
        to_see = np.arange(np.asarray(gt_labels.shape[0]))[non_null]

        to_see_scrambled = np.arange(np.asarray(gt_labels.shape[0]))[scrambled_non_null]
        gt_pcd = gt_pcd.select_by_index(to_see)
        pcd = pcd.select_by_index(to_see_scrambled)
        pcd2 = pcd2.select_by_index(to_see_scrambled)
        pcd3 = pcd3.select_by_index(to_see_scrambled)
        pcd4 = pcd4.select_by_index(to_see_scrambled)
        pcd5 = pcd5.select_by_index(to_see_scrambled)
        if(interactive):
            pcd_diff = pcd2.translate(2*translation)
            pcd_translated = pcd.translate(translation)
            pcd_entropy = pcd3.translate(3*translation)
            pcd_counts = pcd4.translate(4*translation)

            classes = ['wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','desk','curtain','refridgerator','shower','curtain','toilet','sink','bathtub',
            'otherfurniture']
            o3d.visualization.draw([gt_pcd,pcd_diff,pcd_translated,pcd_entropy,pcd_counts],bg_color=(1.0, 1.0, 1.0, 1.0),show_skybox = False)



        alignment_file = '{}/scans/{}/{}.txt'.format(root_dir,scene,scene)
        tf = read_alignment(alignment_file)
        img_save_dir = '{}/{}'.format(exp_save_dir,scene)
        if(not os.path.exists(img_save_dir)):
            os.mkdir(img_save_dir)



        def render_image(pcd):
            mtl = rendering.MaterialRecord()
            # mtl.base_color = [1, 1, 1, 1]
            mtl.base_roughness = 1
            mtl.point_size = 2
            mtl.shader = "defaultLit"
            render.scene.set_background([255, 255, 255, 255])
            if(render.scene.has_geometry("point cloud")):
                render.scene.clear_geometry()
            render.scene.add_geometry("point cloud", pcd_show, mtl)
            render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (0, 1, 0))
            render.scene.scene.enable_sun_light(False)
            render.scene.camera.look_at([0, 0, 0], [0, 0, 10], [0, 1, 0])
            bev_img = render.render_to_image()
            return np.asarray(bev_img)

        if(not interactive):
            for this_pcd,name in zip([gt_pcd,pcd,pcd2,pcd3,pcd4,pcd5],['gt','pred','diff','conf','obs','error_correlations']):
                # Visualize Point Cloud
                pcd_show = this_pcd.transform(tf)
                # vis.add_geometry(pcd_show)
                # pdb.set_trace()
                # vis.get_render_option().light_on = False

                # # Read camera params
                # param = o3d.io.read_pinhole_camera_parameters('cameraparams.json')
                # ctr = vis.get_view_control()
                # ctr.convert_from_pinhole_camera_parameters(param)




                bev_img = render_image(pcd_show)
                if(visualize):
                    if(name in ['conf','diff','error_correlations']):
                        cv2.imshow(name,cv2.cvtColor(bev_img, cv2.COLOR_RGB2BGR))



                # Capture image
                # time.sleep(0.01)
                if(save):
                    imname = '{}/{}.png'.format(img_save_dir,name)
                    cv2.imwrite(imname,cv2.cvtColor(bev_img, cv2.COLOR_RGB2BGR))
                # vis.capture_screen_image(imname)
                # vis.remove_geometry(pcd_show)
                # image = vis.capture_screen_float_buffer()

        if(visualize):
            cv2.waitKey(1)

                # Close
                # vis.destroy_window()
# if(not interactive):
#     vis.destroy_window()



    # vis.update_geometry(gt_pcd)
    # vis.update_geometry(pcd)
    # vis.update_geometry(pcd2)
    # vis.update_geometry(pcd3)
    # vis.poll_events()
    # vis.update_renderer()
    # print(np.unique(np.argmax(labels,axis = 1)))
    # print(np.unique(np.max(labels,axis = 1)))
    # plt.figure()
    # sns.heatmap(cm,vmin = 0,vmax = 1,xticklabels= classes,yticklabels = classes)
    # plt.title("Segformer Confusion Matrix for scene {}".format(scene))
    # plt.show()
    # tm = o3d.geometry.TriangleMesh.create_coordinate_frame (size = 2  )
    # tm = tm.translate(translation)