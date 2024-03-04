import sys

sys.path.append('../external_dependencies/ESANet')
import pickle

# from torch_scatter import scatter
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from src.build_model import build_model
from src.models.model import Upsample
from src.prepare_data import prepare_data
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation


class ESANetClassifier:
    def __init__(self,temperature = 1,NYU = False):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        args = pickle.load(open('./ESANet/args.p','rb'))
        self.model, self.device = build_model(args, n_classes=40)
        args.ckpt_path = '../external_dependencies/ESANet/trained_models/nyuv2/r34_NBt1D_scenenet.pth'
        args.depth_scaling = 0.1
        checkpoint = torch.load(args.ckpt_path,
                                map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.model.to(self.device)
        self.dataset, self.preprocessor = prepare_data(args, with_input_orig=True)
        self.class_mapping = np.array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.,
        0., 13.,  0., 14.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 15.,  0.,
        0.,  0., 16.,  0.,  0.,  0.,  0., 17., 18.,  0., 19.,  0.,  0.,
        20.,  0.]).astype(np.uint8)
        class_matrix = np.zeros((40,21))
        self.temperature = temperature

        for idx,new_class in enumerate(self.class_mapping):
            class_matrix[idx,new_class] = 1

        self.cm = torch.from_numpy(class_matrix.astype(np.float32)).to(self.device)
        self.NYU = NYU

        self.pred_dist = np.zeros((480,640,21))
        self.softmax = nn.Softmax(dim = 1)
    def set_temperature(self,temperature):
        self.temperature = temperature
    def classify(self,img_rgb,depth,x = None,y = None,temperature = None):
        with torch.no_grad():
            # preprocess sample
            sample = self.preprocessor({'image': img_rgb, 'depth': depth})

            # add batch axis and copy to device
            image = sample['image'][None].to(self.device)
            depth = sample['depth'][None].to(self.device)

            # apply network
            pred = self.model(image, depth)

            if(not self.NYU):
                # Condense probabilities for unsupported classes
                pred = torch.tensordot(pred.squeeze(),self.cm,dims = ([0],[0])).permute((2,0,1)).unsqueeze(0)

            if((x is None) or (y is None)):
                pred = F.interpolate(pred, (pred.shape[2],pred.shape[3]),mode='nearest')
            else:
                pred = F.interpolate(pred, (x,y),mode='nearest')

            if(temperature is None):
                pred = pred/self.temperature
            else:
                pred = pred/temperature

            pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy().squeeze().astype(np.uint8)
            # if(not self.NYU):
            #     pred = self.class_mapping[pred]
        return pred
    def get_pred_probs(self,img_rgb,depth,x = None,y = None,temperature = None):
        with torch.no_grad():
            # preprocess sample
            sample = self.preprocessor({'image': img_rgb, 'depth': depth})

            # add batch axis and copy to device
            image = sample['image'][None].to(self.device)
            depth = sample['depth'][None].to(self.device)
            pred = self.model(image, depth)

            if(not self.NYU):
                # Condense probabilities for unsupported classes
                pred = torch.tensordot(pred.squeeze(),self.cm,dims = ([0],[0])).permute((2,0,1)).unsqueeze(0)



            # print(pred.shape)
            if((x is None) or (y is None)):
                pred = F.interpolate(pred, (pred.shape[2],pred.shape[3]),mode='nearest')
            else:
                pred = F.interpolate(pred, (x,y),mode='nearest')

            if(temperature):
                # apply network
                # print('applying temperature scaling')
                pred = self.softmax(pred/temperature)
            else:
                pred = self.softmax(pred/self.temperature)

            # pred = torch.tensordot(pred.cpu(),self.cm,dims = ([0],[0]))
            # pred = pred.cpu().squeeze().permute(1,2,0).detach().numpy()
            # self.pred_dist[:,:,:] = 0
            # for idx,new_class in enumerate(self.class_mapping):
            #     self.pred_dist[:,:,new_class] += pred[:,:,idx]
        return pred.squeeze().detach().permute((1,2,0)).contiguous().cpu().numpy()

    def get_raw_logits(self,img_rgb,depth,x = None,y = None,temperature = None):
        with torch.no_grad():
            sample = self.preprocessor({'image': img_rgb, 'depth': depth})

            # add batch axis and copy to device
            image = sample['image'][None].to(self.device)
            depth = sample['depth'][None].to(self.device)
            # print(pred.shape)
            pred = self.model(image, depth)
            if((x is None) or (y is None)):
                pred = F.interpolate(pred, (pred.shape[2],pred.shape[3]),mode='nearest')
            else:
                pred = F.interpolate(pred, (x,y),mode='nearest')

            if(not self.NYU):
                pred = torch.tensordot(pred.squeeze(),self.cm,dims = ([0],[0])).permute((2,0,1)).unsqueeze(0)
            
            # if(temperature is None):
            #     pred = pred/self.temperature
            # else:
            #     pred = pred/temperature

            if((x is None) or (y is None)):
                pred = F.interpolate(pred, (pred.shape[2],pred.shape[3]),mode='nearest')
            else:
                pred = F.interpolate(pred, (x,y),mode='nearest')
            # pred = pred.cpu().squeeze().permute(1,2,0).detach().numpy()
            # self.pred_dist[:,:,:] = 0
            # for idx,new_class in enumerate(self.class_mapping):
            #     self.pred_dist[:,:,new_class] += pred[:,:,idx]
        return pred.detach().squeeze().permute((1,2,0)).cpu().numpy()

    def get_logits_and_preds(self,img_rgb,depth,x = None,y = None,temperature = None):
        with torch.no_grad():
            sample = self.preprocessor({'image': img_rgb, 'depth': depth})

            # add batch axis and copy to device
            image = sample['image'][None].to(self.device)
            depth = sample['depth'][None].to(self.device)
            # print(pred.shape)
            pred = self.model(image, depth)
            if((x is None) or (y is None)):
                pred = F.interpolate(pred, (pred.shape[2],pred.shape[3]),mode='nearest')
            else:
                pred = F.interpolate(pred, (x,y),mode='nearest')

            if(not self.NYU):
                pred = torch.tensordot(pred.squeeze(),self.cm,dims = ([0],[0])).permute((2,0,1)).unsqueeze(0)

            if(temperature is None):
                pred = pred/self.temperature
            else:
                pred = pred/temperature


            if((x is None) or (y is None)):
                pred = F.interpolate(pred, (pred.shape[2],pred.shape[3]),mode='nearest')
            else:
                pred = F.interpolate(pred, (x,y),mode='nearest')

            pred_classes = torch.argmax(pred, dim=1).detach().squeeze().cpu().numpy()
        # pred = pred.cpu().squeeze().permute(1,2,0).detach().numpy()
        # self.pred_dist[:,:,:] = 0
        # for idx,new_class in enumerate(self.class_mapping):
        #     self.pred_dist[:,:,new_class] += pred[:,:,idx]
        return pred.detach().cpu().numpy().squeeze(),pred_classes

class FineTunedESANet(ESANetClassifier):
    def __init__(self,temperature = 1,checkpoint = '../segmentation_model_checkpoints/ESANet'):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        super().__init__(temperature = temperature,NYU = True)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        # decoder_head = model.decode_head
        self.model.decoder.conv_out = nn.Conv2d(
            128, 21, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.model.decoder.conv_out.requires_grad = False
        self.model.decoder.upsample1 = Upsample(mode='nearest', channels=21)
        self.model.decoder.upsample2 = Upsample(mode='nearest', channels=21)
        self.checkpoint = checkpoint
        self.temperature = temperature
        new_state_dict = self.get_clean_state_dict()
        self.model.load_state_dict(new_state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)
    
    def set_temperature(self,temperature):
        self.temperature = temperature

    def get_clean_state_dict(self):
        params_dict = torch.load(self.checkpoint+'/model.ckpt')
        state = params_dict['state_dict']
        new_state_dict = OrderedDict()
        for param in state.keys():
            prefix,new_param = param.split('.',1)
            if(prefix != 'criterion'):
                new_state_dict.update({new_param:state[param]})
        return new_state_dict


class TSegmenter:
    def __init__(self,temperature = 1):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512").to(self.device)
        self.model.eval()
        df = pd.read_csv("./objectinfo150_2.csv")
        self.class_mapping = df.NewClass.values.astype(np.int64)
        self.cm = torch.from_numpy(self.class_mapping).to(self.device)
        self.temperature = temperature
        
        self.softmax = nn.Softmax(dim = 1)

        # for idx,new_class in enumerate(self.class_mapping):
        #     class_matrix[idx,new_class] = 1

        # self.cm = torch.from_numpy(class_matrix.astype(np.float32)).to(self.device)
    def set_temperature(self,temperature):
        self.temperature = temperature
        
    def classify(self,rgb,depth = None,x=None,y = None,temperature = None):
        with torch.no_grad():
            image = Image.fromarray(np.uint8(rgb))
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            outputs = self.model(pixel_values=inputs['pixel_values'].to(self.device))
            logits = outputs.logits
            # pred = torch.tensordot(logits,self.cm,dims = ([0],[0])).permute((2,0,1)).unsqueeze(0)
            pred = logits

            if((x == None) or( y == None)):
                pred = F.interpolate(pred, (image.height,image.width),mode='nearest')
            else:
                pred = F.interpolate(pred, (x,y),mode='nearest')

            if(temperature):
                # print('applying temperature scaling')
                pred = self.softmax(pred/temperature)
            else:
                pred = self.softmax(pred/self.temperature)

            pred = self.class_mapping[torch.argmax(pred,axis = 1).squeeze().detach().cpu().numpy()]
        return pred
    def get_pred_probs(self,rgb,depth = None,x = None,y = None,temperature = None):
        with torch.no_grad():
            image = Image.fromarray(np.uint8(rgb))
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            # print(inputs['pixel_values'].shape)
            outputs = self.model(pixel_values=inputs['pixel_values'].to(self.device))
            logits = outputs.logits
            # print(logits.shape)
            # pred = torch.tensordot(logits,self.cm,dims = ([0],[0])).permute((2,0,1)).unsqueeze(0)
            pred = self.aggregate_logits(logits)
            # print(pred.shape)
            # pred = logits.unsqueeze(0)
            if((x == None) or( y == None)):
                pred = F.interpolate(pred, (image.height,image.width),mode='nearest')
            else:
                pred = F.interpolate(pred, (x,y),mode='nearest')

            if(temperature):
                # print('applying temperature scaling')
                pred = self.softmax(pred/temperature)
            else:
                pred = self.softmax(pred/self.temperature)
        
        return pred.squeeze().detach().permute((1,2,0)).contiguous().cpu().numpy()


    def get_raw_logits(self,rgb,depth = None,x=None,y = None):
        with torch.no_grad():
            image = Image.fromarray(np.uint8(rgb))
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            # print(inputs['pixel_values'].shape)
            outputs = self.model(pixel_values=inputs['pixel_values'].to(self.device))
            logits = outputs.logits
            if((x == None) or( y == None)):
                pred = F.interpolate(logits, (image.height,image.width),mode='nearest')
            else:
                pred = F.interpolate(logits, (x,y),mode='nearest')
        return pred
    def aggregate_logits(self,logits):
        """Presumes that you still have the batch_size as 0th dimension.

        Args:
            logits (_type_): _description_
        """
        return scatter(logits,self.cm,dim = 1,reduce = 'max')

    def get_aggregate_logits(self,rgb,depth = None,x = None,y = None):
        with torch.no_grad():
            image = Image.fromarray(np.uint8(rgb))
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            # print(inputs['pixel_values'].shape)
            outputs = self.model(pixel_values=inputs['pixel_values'].to(self.device))
            logits = outputs.logits
            # print(logits.shape)
            # pred = torch.tensordot(logits,self.cm,dims = ([0],[0])).permute((2,0,1)).unsqueeze(0)
            pred = self.aggregate_logits(logits)
            if((x == None) or( y == None)):
                pred = F.interpolate(pred, (image.height,image.width),mode='nearest')
            else:
                pred = F.interpolate(pred, (x,y),mode='nearest')
        
        return pred.squeeze().detach().permute((1,2,0)).contiguous().cpu().numpy()


class FineTunedTSegmenter():
    def __init__(self,temperature = 1,model_ckpt = "../segmentation_model_checkpoints/Segformer"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_ckpt).to(self.device)
        self.model.eval()
        self.temperature = temperature
        
        self.softmax = nn.Softmax(dim = 1)

        # for idx,new_class in enumerate(self.class_mapping):
        #     class_matrix[idx,new_class] = 1

        # self.cm = torch.from_numpy(class_matrix.astype(np.float32)).to(self.device)
    def set_temperature(self,temperature):
        self.temperature = temperature
        
    def classify(self,rgb,depth = None,x=None,y = None,temperature = None):
        with torch.no_grad():
            image = Image.fromarray(np.uint8(rgb))
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            outputs = self.model(pixel_values=inputs['pixel_values'].to(self.device))
            logits = outputs.logits


            # pred = torch.tensordot(logits,self.cm,dims = ([0],[0])).permute((2,0,1)).unsqueeze(0)
            if((x == None) or( y == None)):
                pred = F.interpolate(logits, (image.height,image.width),mode='bilinear')
            else:
                pred = F.interpolate(logits, (x,y),mode='bilinear')

            if(temperature):
                # print('applying temperature scaling')
                pred = self.softmax(pred/temperature)
            else:
                pred = self.softmax(pred/self.temperature)

            pred = torch.argmax(pred,axis = 1)

        return pred.squeeze().detach().cpu().numpy()

    def get_pred_probs(self,rgb,depth = None,x = None,y = None,temperature = None):
        with torch.no_grad():
            image = Image.fromarray(np.uint8(rgb))
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            # print(inputs['pixel_values'].shape)
            outputs = self.model(pixel_values=inputs['pixel_values'].to(self.device))
            logits = outputs.logits
            # print(logits.shape)
            # pred = torch.tensordot(logits,self.cm,dims = ([0],[0])).permute((2,0,1)).unsqueeze(0)
            # pred = self.aggregate_logits(logits)
            pred = logits
            # print(pred.shape)
            # pred = logits.unsqueeze(0)
            
            if(temperature):
                # print('applying temperature scaling')
                pred = self.softmax(pred/temperature)
            else:
                pred = self.softmax(pred/self.temperature)
            if((x == None) or( y == None)):
                pred = F.interpolate(pred, (image.height,image.width),mode='nearest')
            else:
                pred = F.interpolate(pred, (x,y),mode='nearest')
        
        return pred.squeeze().detach().permute((1,2,0)).contiguous().cpu().numpy()


    def get_raw_logits(self,rgb,depth = None,x=None,y = None,temperature = 1):
        with torch.no_grad():
            image = Image.fromarray(np.uint8(rgb))
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            # print(inputs['pixel_values'].shape)
            outputs = self.model(pixel_values=inputs['pixel_values'].to(self.device))
            logits = outputs.logits
            if((x == None) or( y == None)):
                pred = F.interpolate(logits, (image.height,image.width),mode='nearest')
            else:
                pred = F.interpolate(logits, (x,y),mode='nearest')
        return pred.squeeze().detach().permute((1,2,0)).contiguous().cpu().numpy()