import albumentations as A
import gc
import sys
sys.path.append('./ESANet')
from utils.my_calibration import Cumulative_mIoU_torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.optim as optim
from src.models.model import Upsample
from utils.segmentation_model_loader import ESANetClassifier, TSegmenter
import torch.nn as nn
import torch
import cv2
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import cv2
from datasets import Dataset
import albumentations as A
import traceback
from collections import OrderedDict
import pickle
from utils.segmentation_model_loader import ESANetClassifier

val_batch_size = 128
train_batch_size = 64
# class_weights = torch.from_numpy(np.array(([ 0.86863202,  1.        ,  1.26482577,  4.97661045,  6.21128435,
#         4.0068586 ,  8.72477767,  4.93037224,  5.65326448, 16.44580194,
#        18.8649601 , 55.24242013, 29.60985561, 11.04643569, 20.82360894,
#        30.38149462, 45.75461865, 32.74486949, 50.70553433, 30.23161118,
#         7.48407616])).astype(np.float32)).float()


# if(not os.path.exists('./hm3d_train_weights.p')):
#     total_pixels = np.zeros(10)
#     for i in tqdm(train_ds):
#         a,b = np.unique(np.asarray(i['labels']),return_counts = True)
#         total_pixels[a] += b

#     weights = total_pixels/total_pixels.sum()
#     weights = np.sqrt(1/weights)
#     # weights = weights/weights[1:].min()
#     pickle.dump(weights,open('./hm3d_train_weights.p','wb'))
# else:
class_weights = torch.from_numpy(pickle.load(open('./hm3d_train_weights.p','rb'))).to('cuda:0').float()


lr = 0.000005
patience = 10
warm_start = False
# checkpoint_to_load = 'jmc12_team/Finetuning ESANet/Try1:v27'
val_dataset_dir = '/tmp/new/val/{}'
train_dataset_dir = '/tmp/new/train/{}'
# train_dataset_dir = '/scratch/bbuq/jcorreiamarques/hm3d_seg/new/train/{}'
# val_dataset_dir = '/scratch/bbuq/jcorreiamarques/hm3d_seg/new/val/{}'

def get_clean_model_params(wandbdirectory,logger):
    artifact = logger.use_artifact(wandbdirectory)
    artifact_dir = artifact.download()
    params_dict = torch.load(artifact_dir+'/model.ckpt')
    state = params_dict['state_dict']
    new_state_dict = OrderedDict()
    for param in state.keys():
        prefix,new_param = param.split('.',1)
        if(prefix != 'criterion'):
            new_state_dict.update({new_param:state[param]})
    return new_state_dict


COLORS = np.array([
    [0, 0, 0], [151, 226, 173], [174, 198, 232], [31, 120, 180], [255, 188, 120], [188, 189, 35], [
        140, 86, 74], [255, 152, 151], [213, 39, 40], [196, 176, 213], [148, 103, 188], [196, 156, 148],
    [23, 190, 208], [247, 183, 210], [218, 219, 141], [254, 127, 14], [
        227, 119, 194], [158, 218, 229], [43, 160, 45], [112, 128, 144], [82, 83, 163]
]).astype(np.uint8)



class RGBD_Dataset(Dataset):
    def __init__(self, dataset_dir, preprocessor):
        super()
        self.dataset_dir = dataset_dir
        # print(dataset_dir)
        rgb_dir = self.dataset_dir.format('rgb')+'/*.png'
        label_dir = self.dataset_dir.format('semseg')+'/*.png'
        depth_dir = self.dataset_dir.format('depth')+'/*.npy'
        self.rgb = sorted(glob(rgb_dir, recursive=True))
        self.labels = sorted(glob(label_dir, recursive=True))
        self.depths = sorted(glob(depth_dir, recursive=True))

        print('\n\n\n\n this is the shape of the datasets: rgb: {} | labels {} | depths | {}'.format(len(self.rgb),len(self.labels),len(self.depths)))


        self.preprocessor = preprocessor
        self.ds_transform = A.Compose([A.RandomCrop(width=640, height=480),A.HorizontalFlip(p=0.5),A.RandomBrightnessContrast(p=0.2)])
        self.augment = False
        # if(split == 'train'):
        #     self.rgb = self.rgb[:int(len(self.rgb)*0.75)]
        #     self.labels = self.labels[:int(len(self.labels)*0.75)]
        #     self.depths = self.depths[:int(len(self.depths)*0.75)]
        # elif(split == 'val'):
        #     self.rgb = self.rgb[int(len(self.rgb)*0.75):]
        #     self.labels = self.labels[int(len(self.labels)*0.75):]
        #     self.depths = self.depths[int(len(self.depths)*0.75):]


    def __len__(self):
        return len(self.rgb)

    def enable_augmentation(self):
        self.augment = True
    def disable_augmentation(self):
        self.augment = False
    def augment_sample(self,rgb,depth,label):
        # print(rgb.shape,depth.shape,label.shape)
        # pdb.set_trace()
        mixed_gt_and_mask = np.concatenate((depth[:,:,np.newaxis],label[:,:,np.newaxis]),axis = 2)
        
        im = self.ds_transform(image = rgb, mask = mixed_gt_and_mask)

        new_rgb = im['image']
        new_depth = im['mask'][:,:,0]
        new_label = im['mask'][:,:,1].astype(np.uint8)
        # print(new_rgb.dtype,new_depth.dtype,new_label.dtype)

        return new_rgb,new_depth,new_label
    def __getitem__(self, indices):
            # pdb.set_trace()
        rgbs = []
        labels = []
        depths = []
        for idx in indices:
            try:

                rgb = cv2.imread(self.rgb[idx], cv2.IMREAD_UNCHANGED)
                label = cv2.imread(self.labels[idx], cv2.IMREAD_UNCHANGED)[:,:,0]
                depth = np.load(self.depths[idx])[:,:,0]
                depth[depth == 0] = 0
                depth[depth ==1] = 0
                invalid_mask = depth == 0
                depth = 0.5 + depth*4.5
                depth[invalid_mask] = 100
                depth = (depth*1000).astype(np.uint16)
                depth[invalid_mask] = 0

                if(self.augment):
                    rgb,depth,label = self.augment_sample(rgb,depth,label)
                sample = self.preprocessor({'image': rgb, 'depth': depth})
                rgb = sample['image']
                depth = sample['depth']
                rgbs.append(rgb)
                depths.append(depth)
                labels.append(label)
            except Exception as e:
                traceback.print_exc()
                print('\n\n\n\n\n There WAS A AN ERROR!!!! \n\n\n\nn\ ')
                print('This is idx {}'.format(idx))
                rgb = np.zeros((480,640,3)).astype(np.uint8)
                depth = np.zeros((480,640)).astype(np.uint16)
                label = np.zeros((480,640)).astype(np.uint8)
                sample = self.preprocessor({'image': rgb, 'depth': depth})
        # pdb.set_trace()

        rgbs = np.array(rgbs)
        depths = np.array(depths)
        labels = np.array(labels)

            # print(rgb,depth,label)


        return {'rgb': rgbs, 'depth': depths, 'label': labels}

a = ESANetClassifier()
model = a.model
preprocessor = a.preprocessor

model.train()

# train_set = RGBD_Dataset(train_dataset_dir, preprocessor)
# train_set.enable_augmentation()

# a = train_set[[1,2,3,4]]
# train_dataloader = DataLoader(train_set, batch_size=train_batch_size,
#                                       shuffle=True, num_workers=1, prefetch_factor=1, persistent_workers=True)

# for i in train_dataloader:
#     print(i)
# a = train_set[0]

for param in model.parameters():
    param.requires_grad = True
# decoder_head = model.decode_head
model.decoder.conv_out = nn.Conv2d(
    128, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
model.decoder.conv_out.requires_grad = True
model.decoder.upsample1 = Upsample(mode='nearest', channels=10)
model.decoder.upsample2 = Upsample(mode='nearest', channels=10)

# for param in decoder_head.parameters():
#     param.requires_grad = True


torch.set_float32_matmul_precision('medium')


class CombinedTrainer(pl.LightningModule):
    def __init__(self, seg_model, preprocessor, lr=0.0001):
        super(CombinedTrainer, self).__init__()
        self.seg_model = seg_model
        self.lr = lr
        self.preprocessor = preprocessor
        self.train_set = RGBD_Dataset(train_dataset_dir, self.preprocessor)
        self.train_set.enable_augmentation()
        self.val_set = RGBD_Dataset(val_dataset_dir, self.preprocessor)
        self.val_set.disable_augmentation()
        self.softmax = nn.Softmax(dim=1)
        self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.metric = MulticlassAccuracy(num_classes=10, average='micro')
        self.mIoU_calc = Cumulative_mIoU_torch(n_classes=10)

    def forward(self, batch):
        image = batch['rgb']
        depth = batch['depth']
        pred = self.seg_model(image, depth)
#         pdb.set_trace()
        return pred

    def train_dataloader(self):

        train_dataloader = DataLoader(self.train_set, batch_size=train_batch_size,
                                      shuffle=True, num_workers=10, prefetch_factor=2, persistent_workers=True,drop_last = False)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_set, batch_size=val_batch_size, shuffle=False,
                                    num_workers=10, prefetch_factor=2, persistent_workers=True, drop_last=False)
        return val_dataloader

    def reset_validation_accumulation_buffers(self):
        self.val_pred = []
        self.val_gt = []
        self.mIoU_calc = Cumulative_mIoU_torch(n_classes=10)

    def compute_validation_mIoU(self):
        self.val_pred = np.array(self.val_pred)
        self.val_gt = np.array(self.val_gt)
        # pdb.set_trace()
#         cal = mECE_Calibration_calc_3D(no_void = False,one_hot = False)
#         cal.update_bins(semantic_label=self.val_pred,semantic_label_gt = self.val_gt)
#         eces = cal.get_ECEs()
        mIoU = np.mean(self.mIoU_calc.get_IoUs())
        # ,'val_mDECE':this_mDECE,'val_DECE':this_DECE)
        self.log_dict({'val_mIoU': mIoU})
        # del this_DECE
        # del this_mDECE
        self.val_pred = []
        self.val_gt = []
        gc.collect()
        torch.cuda.empty_cache()

        self.reset_validation_accumulation_buffers()

    def training_step(self, train_batch, batch_idx):
        pred = self.forward(train_batch)
        pred = pred[0]
        gt = train_batch['label'].long().to('cuda:0')
#         if(self.both):
#             loss = self.criterion(final_prob,log_final_prob,gt)
#             self.log_dict({'train mDECE':self.l1,'train NLL':self.l2},on_epoch = True, on_step = False)
#             if(self.include_grads):
#                 grads_mDECE = torch.autograd.grad(self.l1, self.parameters(), retain_graph=True)
#                 grads_NLL = torch.autograd.grad(self.l2, self.parameters(), retain_graph=True)
#                 mDECE_temp_grad_norms = torch.linalg.norm(grads_mDECE[0])
#                 NLL_temp_grad_norms = torch.linalg.norm(grads_NLL[0])
#                 mDECE_weights_grad_norms = torch.linalg.norm(grads_mDECE[1])
#                 NLL_weights_grad_norms = torch.linalg.norm(grads_NLL[1])
#                 self.log_dict({'mDECE-Temp-Gradient':mDECE_temp_grad_norms,'mDECE-Weights-Gradient':mDECE_weights_grad_norms,
#                                'NLL-Temp-Gradient':NLL_temp_grad_norms,'NLL-Weights-Gradient':NLL_weights_grad_norms})
#         else:
#             if(self.use_dece):
#                 loss = self.criterion(final_prob,gt.long())
#             else:
        loss = self.criterion(pred, gt)
        acc = self.metric(pred.argmax(axis=1).detach().cpu(), gt.cpu())
        self.log_dict(
            {'Batch_Loss': loss, 'Batch Accuracy': acc}, on_step=True)
        self.log_dict({'Epoch Loss': loss, 'Epoch Accuracy': acc},
                      on_epoch=True, on_step=False)
        gc.collect()
        return loss

    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            pred = self.forward(val_batch)
            gt = val_batch['label'].long().to('cuda:0')
            loss = self.criterion(pred, gt)
            acc = self.metric(pred.argmax(axis=1).detach().cpu(), gt.cpu())
            self.mIoU_calc.update_counts(pred.argmax(axis=1),gt)
            self.log_dict(
                {'val_loss': loss, 'val_accuracy': acc}, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # the lightningModule HAS the parameters (remember that we had the __init__ and forward method but we're just not showing it here)
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
#         scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.scheduler_gamma)
        return optimizer




class Validation_metric_callback(Callback):
    def __init__(self, verbose=True):
        self.verbose = verbose
        pass

    def on_validation_epoch_end(self, *args, **kwargs):
        trainer.model.compute_validation_mIoU()

    def on_train_batch_end(self, *args, **kwargs):
        trainer.model.reset_validation_accumulation_buffers()

    def on_validation_epoch_start(self, *args, **kwargs):
        trainer.model.reset_validation_accumulation_buffers()


wandb_logger = WandbLogger(name='Faster Run',
                           project='Objectgoalnav Finetuning ESANet', log_model='all', checkpoint_name='Try3')
wandb_logger.log_hyperparams({'NLL_weights': class_weights, 'lr': lr, 'patience': patience,
                              'batch_size': train_batch_size})


# if(warm_start):
#     # pdb.set_trace()
#     new_state_dict = get_clean_model_params(checkpoint_to_load,wandb_logger)
#     model.load_state_dict(new_state_dict)
#     for param in model.parameters():
#         param.requires_grad = True

torch_device = 'cuda:0'
early_stop_callback = EarlyStopping(
    monitor='val_mIoU', mode='max', patience=patience)
checkpoint_callback = ModelCheckpoint(monitor="val_mIoU", mode="max")
validation_callback = Validation_metric_callback()
pl_model = CombinedTrainer(model, preprocessor, lr=lr)
pl_model.to('cuda:0')
tl = pl_model.train_dataloader()
vl = pl_model.val_dataloader()
trainer = pl.Trainer(max_epochs=300, logger=wandb_logger, accelerator='gpu', devices=1,
                     log_every_n_steps=5, enable_checkpointing=True, num_sanity_val_steps=0,
                     callbacks=[early_stop_callback,
                                checkpoint_callback, validation_callback],
                     val_check_interval=1/3, gradient_clip_val=1, precision=16
                     )


def train():
    if(warm_start):
        trainer.validate(pl_model, vl)
    trainer.fit(pl_model, tl, vl)


train()
