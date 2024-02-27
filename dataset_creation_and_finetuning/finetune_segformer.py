import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from torch import nn
from torchvision.transforms import ColorJitter
from transformers import (
    EarlyStoppingCallback,
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation,
    Trainer,
    TrainingArguments,
)

from utils.scene_definitions import get_filenames

fnames = get_filenames()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model =  SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512").to(device)
huggingface_dataset_dir = fnames["ScanNet_huggingface_dataset_dir"]

COLORS = np.array([
    [0,0,0],[151,226,173],[174,198,232],[31,120,180],[255,188,120],[188,189,35],[140,86,74],[255,152,151],[213,39,40],[196,176,213],[148,103,188],[196,156,148],
    [23,190,208],[247,183,210],[218,219,141],[254,127,14],[227,119,194],[158,218,229],[43,160,45],[112,128,144],[82,83,163]
]).astype(np.uint8)

model.train()
# model = model.from_pretrained('best_model')
for param in model.parameters():
    param.requires_grad = False
decoder_head = model.decode_head
decoder_head.classifier = nn.Conv2d(768,21,kernel_size=(1, 1), stride=(1, 1))
for param in decoder_head.parameters():
    param.requires_grad = True

classes = ['irrelevant','wall','floor','cabinet','bed','chair','sofa','table','door','window','bookshelf','picture',
           'counter','desk','curtain','refridgerator','shower curtain','toilet','sink','bathtub','otherfurniture']
id2label = {}
label2id = {}
print("{")
for i in range(len(classes)):
    comma = ''
    if(i!= len(classes)-1):
        comma = ','
    print('\"{}\":\"{}\"{}'.format(classes[i],i,comma))
    id2label.update({i:classes[i]})
print('}')

feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
feature_extractor.reduce_labels = False

ds = Dataset.load_from_disk(huggingface_dataset_dir)
ds = ds
ds = ds.train_test_split(test_size=0.2)
train_ds = ds["train"]
full_test_ds = ds["test"]
test_ds = full_test_ds.shuffle(seed = 1).train_test_split(test_size = 0.1)['test']

jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1) 

def train_transforms(example_batch):
#     images = [x for x in example_batch['pixel_values']]
#     labels = [x for x in example_batch['label']]
    inputs = feature_extractor(example_batch['pixel_values'], example_batch['label'])
    
    return inputs


def val_transforms(example_batch):
#     images = [x for x in example_batch['pixel_values']]
#     labels = [x for x in example_batch['label']]
#     inputs = feature_extractor(images, labels)
    inputs = feature_extractor(example_batch['pixel_values'], example_batch['label'])
    
    return inputs

# Set transforms
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)

# inverse class frequency error weighing
weights = np.array(([ 0.86863202,  1.        ,  1.26482577,  4.97661045,  6.21128435,
        4.0068586 ,  8.72477767,  4.93037224,  5.65326448, 16.44580194,
       18.8649601 , 55.24242013, 29.60985561, 11.04643569, 20.82360894,
       30.38149462, 45.75461865, 32.74486949, 50.70553433, 30.23161118,
        7.48407616])).astype(np.float32)


epochs = 10
lr = 0.01
batch_size = 12

hub_model_id = "finetuned ScanNet2"

training_args = TrainingArguments(
    "ScanNet Finetuned SegFormer - alt",
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=400,
    eval_steps=400,
    logging_steps=1,
    eval_accumulation_steps=1,
    gradient_accumulation_steps = 3,
    load_best_model_at_end=True,
    metric_for_best_model='mean_iou'
)

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
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
        metrics = metric._compute(
                predictions=pred_labels,
                references=labels,
                num_labels=len(id2label),
                ignore_index=0,
                reduce_labels=feature_extractor.reduce_labels,
            )

        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

        return metrics

early_stop = EarlyStoppingCallback(5,0.0005)


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
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
    
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    callbacks=[early_stop]
)


trainer.train()

#saving the trained model to the best_model directory
trainer.save_model("./best_model")


