
from reconstruction import Reconstruction,LearnedGeneralizedIntegration
from reconstruction import ProbabilisticAveragedReconstruction,HistogramReconstruction,GeometricBayes,GeneralizedIntegration
from utils.segmentation_model_loader import ESANetClassifier, TSegmenter,FineTunedTSegmenter,FineTunedESANet
import open3d as o3d
import json
import torch
import pickle
import numpy as np
import os
class Experiment_Generator:
    def __init__(self):
        self.voxel_size = 0.025 #3.0 / 512
        self.trunc =self.voxel_size * 8
        self.res = 8
        self.n_labels = 21
        self.depth_scale = 1000.0
        self.depth_max = 5.0
        self.miu = 0.001
        self.optimized_temperatures_template = './scaling_results/{}/{}.json'
    
    def get_reconstruction_and_model(self,experiment,process_id = 0):
        integration = experiment['integration']
        calibration = experiment['calibration']
        oracle = experiment.get('oracle',False)
        L = experiment.get('L',0)
        epsilon = experiment.get('epsilon',1)
        segmentation = experiment.get('segmentation','Segformer')
        learned = experiment.get('learned_params',None)
        self.process_id = process_id
        gpu_to_use = self.process_id%torch.cuda.device_count()
        self.o3d_device = 'CUDA:{}'.format(gpu_to_use)
        self.torch_device = 'cuda:{}'.format(gpu_to_use)


        rec = self.get_reconstruction(calibration,integration,segmentation,oracle,epsilon,L,learned)
        model = self.get_model(segmentation,experiment,calibration)
        return rec,model
    
    def get_reconstruction(self,calibration,integration,segmentation,oracle,epsilon,L,learned):
        assert integration in ['Bayesian Update','Naive Bayesian','Naive Averaging','Averaging','Geometric Mean','Histogram','Generalized'],"Integration choice {} is not yet a valid choice".format(integration)
        assert calibration in ['None','2D Temperature Scaling','3D Temperature Scaling','2D Vector Scaling','3D Vector Scaling','VEDE','Informed VEDE','Learned'],"Calibration choice {} is not yet a valid choice".format(calibration)
        if(learned is not None):
            temperature_file = learned['temperature']
            weights_file = learned['weights']

        if(integration in ['Bayesian Update','Naive Bayesian']):
            rec = Reconstruction(depth_scale =self.depth_scale,depth_max=self.depth_max,res =self.res,voxel_size =self.voxel_size,n_labels =self.n_labels,integrate_color = False,
            device = o3d.core.Device(self.o3d_device),miu =self.miu)
        elif(integration in ['Averaging','Naive Averaging']):
            rec = ProbabilisticAveragedReconstruction(depth_scale =self.depth_scale,depth_max=self.depth_max,res =self.res,voxel_size =self.voxel_size,n_labels =self.n_labels,integrate_color = False,
            device = o3d.core.Device(self.o3d_device),miu =self.miu)
        elif(integration == 'Histogram'):
            rec = HistogramReconstruction(depth_scale =self.depth_scale,depth_max=self.depth_max,res =self.res,voxel_size =self.voxel_size,n_labels =self.n_labels,integrate_color = False,
            device = o3d.core.Device(self.o3d_device),miu =self.miu)
        elif(integration == 'Geometric Mean'):
            rec = GeometricBayes(depth_scale =self.depth_scale,depth_max=self.depth_max,res =self.res,voxel_size =self.voxel_size,n_labels =self.n_labels,integrate_color = False,
            device = o3d.core.Device(self.o3d_device),miu =self.miu)
        elif(integration == 'Generalized'):
            if(calibration == 'None'):
                rec = GeneralizedIntegration(depth_scale =self.depth_scale,depth_max=self.depth_max,res =self.res,voxel_size =self.voxel_size,n_labels =self.n_labels,integrate_color = False,
            device = o3d.core.Device(self.o3d_device),miu =self.miu,epsilon = epsilon,L = L,torch_device = self.torch_device)
            else:
                T = pickle.load(open(temperature_file,'rb'))
                weights=  pickle.load(open(weights_file,'rb')) 
                angle_spacings = learned.get('angle_spacings',30)
                depth_spacings = learned.get('depth_spacings',0.5)
                angle_ranges = np.arange(0,90.1,angle_spacings)
                depth_ranges = np.arange(0,5.1,depth_spacings)
                rec = LearnedGeneralizedIntegration(depth_scale =self.depth_scale,depth_max=self.depth_max,res =self.res,voxel_size =self.voxel_size,n_labels =self.n_labels,integrate_color = False,
            device = o3d.core.Device(self.o3d_device),miu =self.miu,epsilon = epsilon,L = L,torch_device = self.torch_device,T=T,weights= weights,angle_ranges=angle_ranges,depth_ranges=depth_ranges)

        return rec

    def get_model(self,segmentation,experiment,calibration):
        assert segmentation in ['Segformer','ESANet'],"Segmentation Model {} is not yet a valid choice"
        if(segmentation == 'Segformer'):
            model = FineTunedTSegmenter()
        elif(segmentation == 'ESANet'):
            model = FineTunedESANet()
        else:
            raise ValueError("Selected model {} is not a valid selection".format(segmentation))
        if((calibration != 'None') and (calibration != 'Learned')):
            T = self.get_correct_temperature(segmentation = segmentation,experiment = experiment)
            model.set_temperature(temperature = T)
        return model
    def get_correct_temperature(self,segmentation,experiment):
        target_file = self.optimized_temperatures_template.format(segmentation,experiment['experiment_name'])
        assert os.path.exists(target_file),"\n The file {} does not exist! \n".format(target_file)
        results = json.load(open(target_file,'rb'))
        params = results['params']
        if(len(params.keys()) == 1):
            T = torch.Tensor([params['T']]).to(self.torch_device).view(1,-1,1,1)
        else:
            T = []
            for i in range(self.n_labels):
                T.append(params['T{}'.format(i)])
            T = torch.Tensor(T).view(1,-1,1,1).to(self.torch_device)
        return T

            

