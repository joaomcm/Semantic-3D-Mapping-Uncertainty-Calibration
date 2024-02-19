import numpy as np
from ESANet_loader import TSegmenter,FineTunedTSegmenter
import torch.nn as nn
import torch
from datasets import Dataset,Image
import cv2
from glob import glob
import numpy as np
from torchvision.transforms import ColorJitter
from glob import glob
import pandas as pd
import albumentations as A
from my_calibration import Cumulative_mIoU
from tqdm import tqdm
import wandb

wandb.login(key = 'eb2c2b155df626514fee504f43abe5fb32cb170d')
full_finetune = True
# from my_calibration import 
if(full_finetune):
    modelwrapper = FineTunedTSegmenter(temperature = 1,model_ckpt = "./objectGoalNavFinedTunedSegFormer2_full_finetune")
else:
    modelwrapper = TSegmenter(temperature = 1)
model = modelwrapper.model


COLORS = np.array([
    [0,0,0],[151,226,173],[174,198,232],[31,120,180],[255,188,120],[188,189,35],[140,86,74],[255,152,151],[213,39,40],[196,176,213],[148,103,188],[196,156,148],
    [23,190,208],[247,183,210],[218,219,141],[254,127,14],[227,119,194],[158,218,229],[43,160,45],[112,128,144],[82,83,163]
]).astype(np.uint8)

# dataset_dir = '/tmp/new/{}/{}'
dataset_dir = '/scratch/bbuq/jcorreiamarques/hm3d_seg/new/{}/{}'

rgb_dir = dataset_dir.format('train','rgb')+'/*.png'
label_dir = dataset_dir.format('train','semseg')+'/*.png'
images = sorted(glob(rgb_dir))
labels = sorted(glob(label_dir))
existing_gts = pd.Series(labels).str.split('/',expand = True).iloc[:,-1]
existing_rgb = pd.Series(images).str.split('/',expand = True).iloc[:,-1]
in_common = existing_rgb.isin(existing_gts).tolist()
images = (pd.Series(images)[in_common]).tolist()

train_ds_0 = Dataset.from_dict({'pixel_values':images,'label':labels}).cast_column("pixel_values", Image()).cast_column('label',Image())
train_ds = train_ds_0.shuffle(seed = 1)

rgb_dir = dataset_dir.format('val','rgb')+'/*.png'
label_dir = dataset_dir.format('val','semseg')+'/*.png'
images = sorted(glob(rgb_dir))
labels = sorted(glob(label_dir))
existing_gts = pd.Series(labels).str.split('/',expand = True).iloc[:,-1]
existing_rgb = pd.Series(images).str.split('/',expand = True).iloc[:,-1]
in_common = existing_rgb.isin(existing_gts).tolist()
images = (pd.Series(images)[in_common]).tolist()

val_ds = Dataset.from_dict({'pixel_values':images,'label':labels}).cast_column("pixel_values", Image()).cast_column('label',Image())

# og_ds = og_ds.shuffle(seed = 1)
# og_ds = og_ds.train_test_split(test_size=0.4)
# val_ds = og_ds["train"]
# test_ds = og_ds["test"]
model.train()
# model = model.from_pretrained('best_model')

for param in model.parameters():
    param.requires_grad = full_finetune
if(not full_finetune):
    decoder_head = model.decode_head
    decoder_head.classifier = nn.Conv2d(768,10,kernel_size=(1, 1), stride=(1, 1))
    for param in decoder_head.parameters():         
        param.requires_grad = True

classes = ['chair', 'sofa', 'plant', 'bed', 'toilet', 'tv_monitor',
              'fireplace', 'bathtub', 'mirror','other']
id2label = {}
label2id = {}
print("{")
for i in range(len(classes)):
    comma = ''
    if(i!= len(classes)-1):
        comma = ','
    print('\"{}\":\"{}\"{}'.format(classes[i],i,comma))
    id2label.update({i:classes[i]})
    label2id.update({classes[i]:i})
print('}')

feature_extractor = modelwrapper.feature_extractor
feature_extractor.reduce_labels = False

jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) 

transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])


def train_transforms(example_batch):
#     imgs = []
#     for item in example_batch:
#     images = [x for x in example_batch['pixel_values']]
#     labels = [x for x in example_batch['label']]
#     print(np.array(example_batch['image'][0]).shape,np.array(example_batch['label'][0]).shape)
#     print(example_batch)
#     print(example_batch)
    im = transform(image = np.array(example_batch['pixel_values'][0]), mask = np.array(example_batch['label'][0])[:,:,0])
#     pdb.set_trace()

    inputs = feature_extractor(im['image'], im['mask'])
    
    return inputs


def val_transforms(example_batch):
#     images = [x for x in example_batch['pixel_values']]
#     labels = [x for x in example_batch['label']]
#     inputs = feature_extractor(images, labels)
    inputs = feature_extractor(example_batch['pixel_values'][0], np.array(example_batch['label'][0])[:,:,0])
    
    return inputs

# Set transforms
train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)

from tqdm import tqdm
import os
import pickle

if(not os.path.exists('./hm3d_train_weights.p')):
    total_pixels = np.zeros(10)
    for i in tqdm(train_ds):
        a,b = np.unique(np.asarray(i['labels']),return_counts = True)
        total_pixels[a] += b

    weights = total_pixels/total_pixels.sum()
    weights = np.sqrt(1/weights)
    # weights = weights/weights[1:].min()
    pickle.dump(weights,open('./hm3d_train_weights.p','wb'))
else:
    weights = pickle.load(open('./hm3d_train_weights.p','rb'))

from transformers import TrainingArguments

epochs = 100
lr = 0.00001
batch_size = 128

hub_model_id = "finetuned ScanNet2 ObjectGoalNav"

training_args = TrainingArguments(
    "HM3D Finetuned SegFormer - ObjectGoalNav - sqrt weights - Shuffled - full finetune",
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=1111,
    eval_steps=1111,
    logging_steps=1,
    eval_accumulation_steps=1,
    gradient_accumulation_steps = 1,
    load_best_model_at_end=True,
    report_to='wandb',
    metric_for_best_model = 'mIoU',
    dataloader_num_workers = 16,
    save_on_each_node = True,
    fp16 = True
)

import torch
from torch import nn
import evaluate

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
#         print('one_iter')
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
#         print(logits_tensor.size())
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        # currently using _compute instead of compute
        # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
#         metrics = metric._compute(
#                 predictions=pred_labels,
#                 references=labels,
#                 num_labels=len(id2label),
#                 ignore_index=8,
#                 reduce_labels=feature_extractor.reduce_labels,
#             )

        # add per category metrics as individual key-value pairs
        metrics = {}
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

#         metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
        
        return metrics
    

from transformers import Trainer
import torch 

torch.set_float32_matmul_precision('medium')

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels").to(model.device)
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits').to(model.device)
        logits = nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights).float().to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
    def evaluate(self,eval_dataset = None,ignore_keys = None,metric_key_prefix = 'eval'):
        calc_iou = Cumulative_mIoU(n_classes = 10)
        eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
            
        with torch.no_grad():
            self.model.eval()
            for inputs in tqdm(eval_dataloader,desc = 'eval loop'):
                labels = inputs.get("labels").to(self.model.device).cpu().numpy()
                for k in inputs.keys():
#                     print(k)
                    inputs.update({k:inputs[k].to(self.model.device)})
                # forward pass
                outputs = self.model(**inputs)
                logits = outputs.get('logits').to(self.model.device)
                preds = nn.functional.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).argmax(dim=1).cpu().numpy()
                calc_iou.update_counts(preds,labels)
        per_category_iou = calc_iou.get_IoUs()
        metrics = {}
        metrics.update({metric_key_prefix+'_'+f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
        metrics.update({metric_key_prefix+'_'+'mIoU':np.nanmean(per_category_iou)})
        self.log(metrics)
        self.model.train()
#         return metrics
            
    
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

model.config.id2label = id2label
model.config.label2id = label2id

# trainer.save_model("./objectGoalNavFinedTunedSegFormer2")
trainer.save_model("./objectGoalNavFinedTunedSegFormer2_full_finetune")
