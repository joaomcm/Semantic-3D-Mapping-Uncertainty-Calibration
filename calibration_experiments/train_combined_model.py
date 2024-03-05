import os
import sys
from glob import glob

os.environ["OMP_NUM_THREADS"] = "8" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "8" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "8" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "8" # export NUMEXPR_NUM_THREADS=6

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir)

import argparse
import gc
import pdb
import pickle

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm


from learned_calibrators import DifferentiableVectorCalibrator
from external_dependencies.meta_calibration.Losses.dece import DECE, mDECE
from utils.my_calibration import Cumulative_mIoU
from utils.my_calibration import (
    mECE_Calibration_calc_3D_fix as mECE_Calibration_calc_3D,
)
from utils.scene_definitions import get_filenames

torch.set_float32_matmul_precision('medium')

fnames = get_filenames()
nll_weight_type = 'log'
assert nll_weight_type in ['sqrt','log'], "Invalid choice of Weight Type {}".format(nll_weight_type)
# log of the inverse frequency small dataset
# class_weights = torch.from_numpy(np.array(([0.90080415, 1.57968535, 2.17625471, 3.54496332, 3.64892668,
#        3.92682999, 3.97699309, 4.357388  , 3.34383647, 3.71111319,
#        3.62912815, 6.76405612, 5.9988827 , 3.98192895, 4.04597801,
#        6.49766603, 5.56734288, 6.16151051, 6.14521167, 5.18696208,
#        3.61694632])).astype(np.float32)).to('cuda:0')

# #log of the inverse frequency large dataset
if(nll_weight_type == 'log'):
    class_weights = torch.from_numpy(np.array([0.8529185 , 1.71680491, 1.81150784, 3.79800334, 4.09265521,
        2.97619434, 4.18588315, 3.63256882, 3.65050605, 4.38539509,
        4.84077071, 5.28943268, 5.45705945, 4.87534981, 4.77233873,
        5.65772243, 6.72807301, 6.39965977, 6.64812876, 6.31896787,
        4.02620683]).astype(np.float32)).to('cuda:0')



#sqrt of the inverse frequency for the small dataset
# class_weights = torch.Tensor([ 1.56894289,  2.20304981,  2.96870952,  5.88544092,  6.19946713,
#         7.12361267,  7.30454346,  8.83476052,  5.32236758,  6.39525684,
#         6.1383997 , 29.43039725, 20.07431926,  7.32259281,  7.56089072,
#        25.76026053, 16.17830971, 21.77484177, 21.59811055, 13.37625392,
#         6.10112487]).to('cuda:0')

#sqrt of the inverse frequency for the large dataset
if(nll_weight_type == 'sqrt'):
    class_weights = torch.Tensor([ 1.5318241 ,  2.35938846,  2.47379626,  6.67922305,  7.73942662,
            4.42866051,  8.10873257,  6.14896891,  6.20436467,  8.95934878,
        11.25019377, 14.07945073, 15.31036006, 11.44639583, 10.87176828,
        16.92617456, 28.9056335 , 24.52835725, 27.77300144, 23.55843507,
            7.48651511]).to('cuda:0')


#uniform class weights
# class_weights = torch.from_numpy(np.ones(21)).to('cuda:0').float()


# wandb.init(project = 'unified integration optimization',name = 'streamlined - non biased sampling, weighed loss',
#            config = {'learning_rate':0.001,'L':L,'epsilon':epsilon,
#                      'class_weights':class_weights.tolist(),
#                      'batch_size':batch_size,'epochs':epochs})


class StreamlinedWeightEstimator(nn.Module):
    def __init__(self,depth_ranges = np.arange(0.0,5.1,1),angle_ranges =  np.arange(0,90.1,30),classes = 21,init_weights = None,requires_grad = True):
        super(StreamlinedWeightEstimator,self).__init__()
        self.classes = 21
        self.rl = nn.ReLU()
        self.torch_device = 'cuda:0'
        self.depth_ranges = torch.Tensor(depth_ranges).to(self.torch_device)
        self.angle_ranges = torch.Tensor(angle_ranges).to(self.torch_device)
        self.requires_grad = requires_grad
        if(requires_grad == True):
            if(init_weights is None):
                self.weights = nn.Parameter(torch.tensor(np.ones((21,self.depth_ranges.shape[0],
                                                self.angle_ranges.shape[0])),requires_grad = self.requires_grad).to(self.torch_device))
            else:
                self.weights = nn.Parameter(torch.tensor(init_weights),requires_grad = self.requires_grad).to(self.torch_device)
        else:
            self.weights = torch.tensor(np.ones((21,self.depth_ranges.shape[0],
                                                self.angle_ranges.shape[0]))).to(self.torch_device)
        self.rays = None

    def get_weights(self,depth,angle,logits):
        semantic_map = logits.argmax(dim = 2)
        dd,ap = self.digitize(depth,angle)
        w = self.rl(self.weights[semantic_map,dd,ap]) + 0.0001
        del dd 
        del ap
#         del semantic_map
        return w
    def digitize(self,rendered_depth_1,angle):
        with torch.no_grad():
            dr = self.depth_ranges
            ar = self.angle_ranges
            rendered_depth = rendered_depth_1
            angle = torch.clamp(angle,0,90)
            digitized_depth = torch.clamp(torch.bucketize(rendered_depth[:,:].float(),dr),0,dr.shape[0]-1)
            angle_proj = torch.clamp(torch.bucketize(angle,ar).reshape(digitized_depth.shape),0,ar.shape[0]-1)

            return digitized_depth,angle_proj
        
class FullyLearnedWeightEstimator(nn.Module):
    def __init__(self):
        super(FullyLearnedWeightEstimator,self).__init__()
        self.weight_generator = nn.Sequential(
            nn.Linear((21+1+1),64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return 10*self.weight_generator(x) + 0.0001
    
    def get_weights(self,depth,angle,probits):
        
        depth2 = 2*depth/5-1
        angle2 = 2*angle/90-1
        probits2 = 2*probits-1
        cat_input = torch.cat([probits2,depth2.unsqueeze(2),angle2.unsqueeze(2)],dim = 2)
        original_dim = cat_input.shape
        output = self.forward(cat_input.view(-1,23))
        output = output.view(original_dim[:-1])
        return output



class voxel_readings_dataset(Dataset):
    def __init__(self,ds_directory,max_obs = 1000):
        self.ds_directory = ds_directory
        self.files = sorted(glob(self.ds_directory,recursive=True))
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

class voxel_readings_dataset_h5py(Dataset):
    def __init__(self,ds_directory,split = 'train',size = 900000,max_uses = 20000000):
        self.ds_directory = ds_directory
        self.size = size
        self.split = split
        self.uses = 0
        self.max_uses = max_uses
    def h5py_worker_init(self):
        # print('starting a new dataset instance')
        self.uses = 0 
        self.f = h5py.File(self.ds_directory,'r')
        self.logits = self.f['logits']
        self.depth = self.f['depth']
        self.angle = self.f['angle']
        self.label = self.f['label']
        
    def __len__(self):
        return self.size - 1
    def __getitem__(self,idx):
        self.uses += 1
        try:
            sample = {'logits':self.logits[idx,:,:].astype(np.float32),'depth':self.depth[idx,:].astype(np.float32),
                    'angle':self.angle[idx,:].astype(np.float32),'label':self.label[idx]}
        except Exception as e:
            print('There was an error {} loading item {}, retrying with fresh file'.format(e,idx))
            self.f.close()
            self.h5py_worker_init()
            try:
                sample = {'logits':self.logits[idx,:,:].astype(np.float32),'depth':self.depth[idx,:].astype(np.float32),
                    'angle':self.angle[idx,:].astype(np.float32),'label':self.label[idx]}
            except Exception as e:
                print('Retrying failed with error {}, returning dummy values'.format(e))
                sample = {'logits':np.zeros((1000,21)).astype(np.float32),'depth':np.zeros(1000).astype(np.float32),
                    'angle':np.zeros(1000).astype(np.float32),'label':np.array([0]).astype(np.uint8)}
                self.f.close()
                self.h5py_worker_init()
        if(self.uses > self.max_uses):
            self.f.close()
            self.h5py_worker_init()
        return sample
    
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.h5py_worker_init()


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

class Validation_metric_callback(Callback):
    def __init__(self,verbose = True):
        self.verbose = verbose
        pass
    def on_validation_epoch_end(self,*args,**kwargs):
        trainer.model.compute_validation_ECE()
    def on_train_batch_end(self,*args,**kwargs):
        trainer.model.reset_validation_accumulation_buffers()
        pass
    def on_validation_epoch_start(self,*args,**kwargs):
        trainer.model.reset_validation_accumulation_buffers()
        pass

class CombinedTrainer(pl.LightningModule):
    def __init__(self,weight_estimator,scaling,L,epsilon,lr = 0.0001,use_dece = False,both = False,include_grads = False,model_name = 'ESANet'):
        super(CombinedTrainer,self).__init__()
        self.scaling = scaling
        self.weight_estimator = weight_estimator
        self.L = L
        self.epsilon = epsilon
        self.use_dece = use_dece
        self.both = both
        self.DECE = DECE(torch_device,10,t_a = 100,t_b = 0.01)
        self.model_name = model_name
        self.include_grads = include_grads
        if(self.both):
            self.c1 = mDECE(torch_device,10,t_a = 100,t_b = 0.01,ignore_class = 0)
            self.c2 = torch.nn.NLLLoss(weight = class_weights)
            self.criterion = self.both_criterion

        else:
            if(self.use_dece):
                self.criterion = mDECE(torch_device,10,t_a = 100,t_b = 0.01,ignore_class = 0)
            else:    
                self.criterion =  torch.nn.NLLLoss(weight = class_weights)
        self.metric =  MulticlassAccuracy(num_classes = 21,average = 'micro')
        self.lr = lr   
        self.reset_validation_accumulation_buffers()
    
    def both_criterion(self,prob,log_final_prob,gt):
        # self.l1 = self.c1(prob,gt.long())/4
        self.l1 = self.c1(prob,gt.long())*mDECE_loss_weight
        self.l2 = self.c2(log_final_prob,gt.long())

        return self.l1+self.l2
    
    def forward(self,batch):
        smoothing = 0.0001
        angle = batch['angle'].to('cuda:0')
        depth = batch['depth'].to('cuda:0')
        logits = batch['logits'].to('cuda:0')
        padded_logits = torch.logical_not(torch.all(logits==0,dim = 2).float())
        preds = logits.argmax(dim = 2)
        # weights = self.weight_estimator.get_weights(depth,angle,logits)
        # masked_weights = (weights*padded_logits).float()
        unsmooth_probits = sm(self.scaling(logits)).to(torch_device).float()
        # laplace smoothing
        unsmooth_probits2 = unsmooth_probits + smoothing
        probits = unsmooth_probits2/unsmooth_probits2.sum(keepdim = True,axis = 2)
        # pdb.set_trace()
        weights = self.weight_estimator.get_weights(depth,angle,probits)
        masked_weights = (weights*padded_logits).float()
    #     probits = sm(scaling(logits))
        log_probits = torch.log(probits)
        alpha =((1-self.epsilon)/masked_weights.sum(axis = 1)+self.epsilon).view(-1,1)
        sum_probits = (masked_weights.unsqueeze(2)*probits).sum(axis = 1)
        sum_log_probits = (masked_weights.unsqueeze(2)*log_probits).sum(axis = 1)
        semifinal_prob = self.L*sm2((alpha*sum_log_probits))+(1-self.L)*alpha*sum_probits
        final_prob = semifinal_prob/semifinal_prob.sum(axis = 1,keepdims = True)
        log_final_prob = torch.log(final_prob)
        del batch
        del sum_probits
        del semifinal_prob
        del weights
        del preds
        del unsmooth_probits
        del unsmooth_probits2
        del padded_logits
        del angle
        del depth
        del masked_weights
        del sum_log_probits
        gc.collect()
        # pdb.set_trace()
        return final_prob,log_final_prob
    def train_dataloader(self):
        # train_directory = '/home/motion/Instance Retrieval/data/calibration/per_voxel/train/**/*.p'
        # train_ds = voxel_readings_dataset(train_directory)
        # train_dataloader = DataLoader(train_ds,batch_size = batch_size,shuffle = True,collate_fn = debug_collate,num_workers = 5,prefetch_factor = 3)
        if(self.model_name == 'ESANet'):
            train_directory = fnames['singleton_dataset_dir'] + '/ESANet_train.h5py'
        elif(self.model_name == 'Segformer'):
            train_directory = fnames['singleton_dataset_dir'] + '/Segformer_train.h5py'
        tmp = h5py.File(train_directory,'r')
        size = tmp['logits'].shape[0]
        tmp.close()
        del tmp
        train_ds = voxel_readings_dataset_h5py(train_directory,split = 'train',size = size)
        train_dataloader = DataLoader(train_ds,batch_size = batch_size,shuffle = True,num_workers = 8,worker_init_fn = worker_init_fn,prefetch_factor = 1,persistent_workers = True)
        return train_dataloader
    def val_dataloader(self):
        # val_directory = '/home/motion/Instance Retrieval/data/calibration/per_voxel/validation/**/*.p'
        # val_ds = voxel_readings_dataset(val_directory)
        # val_dataloader = DataLoader(val_ds,batch_size = batch_size,shuffle = False,collate_fn = debug_collate,num_workers = 5,prefetch_factor = 3)
        if(self.model_name == 'ESANet'):
            val_directory = fnames['singleton_dataset_dir'] + '/ESANet_validation.h5py'
        elif(self.model_name == 'Segformer'):
            val_directory = fnames['singleton_dataset_dir'] + '/Segformer_validation.h5py'        
        
        tmp = h5py.File(val_directory,'r')
        size = tmp['logits'].shape[0]
        tmp.close()
        del tmp
        val_ds = voxel_readings_dataset_h5py(val_directory,split = 'val',size = size)
        val_dataloader = DataLoader(val_ds,batch_size = val_batch_size,shuffle = False,num_workers = 8,worker_init_fn = worker_init_fn,prefetch_factor = 1,persistent_workers = True,drop_last = False)
        
        return val_dataloader
    def reset_validation_accumulation_buffers(self):
        self.val_pred = []
        self.val_gt = []
        self.cal = mECE_Calibration_calc_3D(no_void = True,one_hot = False)
        self.IoU_metric = Cumulative_mIoU(n_classes =21)
    def compute_validation_ECE(self):
        # self.val_pred = np.array(self.val_pred)
        # self.val_gt = np.array(self.val_gt)
        # pdb.set_trace()
        # cal = mECE_Calibration_calc_3D(no_void = False,one_hot = False)
        # cal.update_bins(semantic_label=self.val_pred,semantic_label_gt = self.val_gt)
        eces = self.cal.get_ECEs()
        # IoU_metric = Cumulative_mIoU(n_classes =21)
        # IoU_metric.update_counts(self.val_pred.argmax(axis = 1),self.val_gt)
        mIoU = np.mean(self.IoU_metric.get_IoUs()[1:])

        # comparing with DECE and mDECE
        # self.val_pred = torch.Tensor(self.val_pred)
        # self.val_gt = torch.Tensor(self.val_gt).long()
        # with torch.no_grad():
        #     self.DECE = self.DECE.to('cpu')
        #     self.DECE.device = 'cpu'
        #     this_DECE = self.DECE(self.val_pred,self.val_gt)
        #     self.c1 = self.c1.to('cpu')
        #     self.c1.device = 'cpu'
        #     this_mDECE = self.c1(self.val_pred,self.val_gt)
        #     self.c1 = self.c1.to(torch_device)
        #     self.DECE.device = torch_device
        #     self.c1.device = torch_device


        self.log_dict({'val_mECE':self.cal.get_mECE(),'val_ECE':eces[-2],'val_mIoU':mIoU})#,'val_mDECE':this_mDECE,'val_DECE':this_DECE)
        
        # del this_DECE
        # del this_mDECE
        del self.cal
        del self.IoU_metric
        self.val_pred = []
        self.val_gt = []
        gc.collect()
        torch.cuda.empty_cache()


        self.reset_validation_accumulation_buffers()

    def training_step(self,train_batch,batch_idx):
        final_prob,log_final_prob = self.forward(train_batch)
        gt = train_batch['label'].to('cuda:0').view(-1)
        if(self.both):
            loss = self.criterion(final_prob,log_final_prob,gt)
            self.log_dict({'train mDECE':self.l1,'train NLL':self.l2},on_epoch = True, on_step = False)
            if(self.include_grads):
                grads_mDECE = torch.autograd.grad(self.l1, self.parameters(), retain_graph=True)
                grads_NLL = torch.autograd.grad(self.l2, self.parameters(), retain_graph=True)
                mDECE_temp_grad_norms = torch.linalg.norm(grads_mDECE[0])
                NLL_temp_grad_norms = torch.linalg.norm(grads_NLL[0])
                mDECE_weights_grad_norms = torch.linalg.norm(grads_mDECE[1])
                NLL_weights_grad_norms = torch.linalg.norm(grads_NLL[1])
                self.log_dict({'mDECE-Temp-Gradient':mDECE_temp_grad_norms,'mDECE-Weights-Gradient':mDECE_weights_grad_norms,
                               'NLL-Temp-Gradient':NLL_temp_grad_norms,'NLL-Weights-Gradient':NLL_weights_grad_norms})
        else:    
            if(self.use_dece):
                loss = self.criterion(final_prob,gt.long())
            else:
                loss = self.criterion(log_final_prob,gt) 
        acc = self.metric(final_prob.argmax(axis = 1).detach().cpu(),gt.cpu())
        self.log_dict({'Batch_Loss':loss,'Batch Accuracy':acc},on_step = True)
        self.log_dict({'Epoch Loss':loss,'Epoch Accuracy':acc},on_epoch = True, on_step = False)
        gc.collect()
        return loss
    def validation_step(self,val_batch,batch_idx):
        with torch.no_grad():        
            final_prob,log_final_prob = self.forward(val_batch)
            gt = val_batch['label'].to('cuda:0').view(-1)
            # self.val_gt.extend(gt.detach().cpu().numpy())
            # self.val_pred.extend(final_prob.detach().cpu().numpy())        
            if(self.both):
                loss = self.criterion(final_prob,log_final_prob,gt)
                # self.log_dict({'val mDECE':self.l1,'val NLL':self.l2},on_epoch = True, on_step = False)

            else:
                if(self.use_dece):
                    loss = self.criterion(final_prob,gt.long())
                else:
                    loss = self.criterion(log_final_prob,gt)             
            acc = self.metric(final_prob.argmax(axis = 1).detach().cpu(),gt.detach().cpu())
            self.cal.update_bins(final_prob.detach().cpu().numpy(),gt.detach().cpu().numpy())
            self.IoU_metric.update_counts(final_prob.argmax(axis = 1).cpu().detach(),gt.detach().cpu().numpy())
            self.log_dict({'val_loss':loss,'val_accuracy':acc},on_epoch = True)
        return loss
    def configure_optimizers(self):
        # the lightningModule HAS the parameters (remember that we had the __init__ and forward method but we're just not showing it here)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = optim.SGD(self.parameters(),lr = self.lr,momentum = 0.9)
#         scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.scheduler_gamma)
        return optimizer



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Model training command line settings')
    parser.add_argument('--model_to_train', type=str, default="ESANet",
                    help='which model we are calibrating')
    parser.add_argument('--use_mDECE', type=int, default=1,
                    help='Use mDECE in the loss')
    parser.add_argument('--use_NLL', type=int, default=1,
                    help='Use NLL in the loss')
    parser.add_argument('--exp_name',type=str,
                    help='how to log this model to WandB')
    parser.add_argument('--learned_weights',type = int,default = 1,help = 'whether to use learned sample weights')
    parser.add_argument('--angle_spacing',type = float,default = 30,help = 'spacing between angle bins')
    parser.add_argument('--depth_spacing',type = float,default = 1,help = 'spacing between angle bins')
    parser.add_argument('--loss_weight_mDECE',type = float,default = 0.5,help = 'the weight given to the mDECE loss in the total loss')
    parser.add_argument('--lr',type = float,default = 0.00001,help = 'the weight given to the mDECE loss in the total loss')
    parser.add_argument('--record_gradients',type = int,default = 0,help = 'whether to record the gradients')

    args = parser.parse_args()


    epsilon = 0
    L = 0
    criterion = torch.nn.NLLLoss(weight = class_weights)
    accumulated_losses = []
    epochs = 1000
    metric = MulticlassAccuracy(num_classes = 21)
    batch_size = 36000
    val_batch_size = 36000
    # batch_size = 16000
    smoothing = 0.0001
    lr = args.lr
    use_mDECE = args.use_mDECE > 0
    patience = 5
    angle_spacing = args.angle_spacing
    depth_spacing = args.depth_spacing
    learned_sample_weights = args.learned_weights > 0
    weight_mlp = False
    large_dataset = True
    use_both = (use_mDECE and(args.use_NLL > 0))
    record_grads = args.record_gradients > 0
    trained_scaling = True
    linear_weights = False
    mDECE_loss_weight = args.loss_weight_mDECE


    model_name = args.model_to_train
    if(model_name == 'ESANet'):
        project_name = 'ESANET Learned Calibration'
    else:
        project_name = 'Segformer Learned Calibration'

    wandb.login(key = 'eb2c2b155df626514fee504f43abe5fb32cb170d')
    wandb_logger = WandbLogger(name= args.exp_name,
                            project=project_name,log_model = 'all',checkpoint_name = 'Try1')
    wandb_logger.log_hyperparams({'NLL_weights':class_weights,'lr':lr,'use_mDECE':use_mDECE,'patience':patience,
                                  'angle_spacing':angle_spacing,'depth_spacing':depth_spacing,'learned_weights':learned_sample_weights,
                                  'mlp_weights':weight_mlp,'batch_size':batch_size,
                                  'trained_scaling':trained_scaling,'linear_weights':linear_weights,
                                  'nll_weight_type':nll_weight_type,'mDece_loss_weight':mDECE_loss_weight
                            })

    torch_device = 'cuda:0'
    T = np.ones(21)
    T[:] = 0.4
    scaling = DifferentiableVectorCalibrator(T0 = T,requires_grad= trained_scaling,linear = linear_weights).to(torch_device)
    scaling.to(torch_device)
    if(not weight_mlp):
        weight_estimator = StreamlinedWeightEstimator(requires_grad= learned_sample_weights,depth_ranges = np.arange(0.0,5.1,depth_spacing),angle_ranges =  np.arange(0,90.1,angle_spacing)).to(torch_device)
    else:
        weight_estimator = FullyLearnedWeightEstimator().to(torch_device)
    sm = nn.Softmax(dim = 2)
    sm2 = nn.Softmax(dim = 1)

    early_stop_callback = EarlyStopping(monitor='val_mIoU',mode = 'max',patience = patience)
    checkpoint_callback = ModelCheckpoint(monitor="val_mIoU", mode="max")
    validation_callback = Validation_metric_callback()
    model = CombinedTrainer(weight_estimator,scaling,L,epsilon,lr = lr,use_dece = use_mDECE,both = use_both,include_grads= record_grads,model_name = model_name)
    model.to('cuda:0')
    # model = CombinedTrainer.load_from_checkpoint('/home/motion/Instance Retrieval/unified integration optimization lightning/try1/model.ckpt')
    tl = model.train_dataloader()
    vl = model.val_dataloader()
    trainer = pl.Trainer(max_epochs = 300,logger= wandb_logger, accelerator = 'gpu',devices = 1,
                        log_every_n_steps=5,enable_checkpointing = True,num_sanity_val_steps = 0,
                        callbacks = [early_stop_callback,checkpoint_callback,validation_callback],
                        val_check_interval = 1/3,gradient_clip_val = 1,precision = 16
                        )
    # trainer = pl.Trainer(resume_from_checkpoint = '/home/motion/Instance Retrieval/unified integration optimization lightning/try1/model.ckpt', accelerator = 'gpu',devices = 1,logger = wandb_logger)

    def train():
        trainer.validate(model,vl)
        trainer.fit(model,tl,vl)

    train()


    print(trainer.model.scaling.get_T())
    print(trainer.model.weight_estimator.weights)
