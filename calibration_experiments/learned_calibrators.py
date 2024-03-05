import numpy as np
import torch
import torch.nn as nn

class DifferentiableVectorCalibrator(nn.Module):
    def __init__(self,device = torch.device('cuda'),classes = 21,bins = 10,T0 = 21*[1/21],requires_grad = True,linear = False):
        super(DifferentiableVectorCalibrator,self).__init__()
        self.linear = linear
        self.T_min = 0.05
        self.relu = nn.ReLU()
        if(not self.linear):
            self.T = nn.Parameter(torch.tensor(np.log(T0).astype(np.float32),requires_grad = requires_grad))
        else:
            self.T = nn.Parameter(torch.tensor(np.array(T0).astype(np.float32),requires_grad = requires_grad))
        self.device = device
    def forward(self,logits,shape = (1,-1)):
        return logits/self.get_T()
    def get_T(self,numpy = False):
        if(not self.linear):
            T = torch.exp(self.T).view(1,-1)
        else:
            T_temp = self.T.view(1,-1)
            T = self.relu(T_temp) + self.T_min
            # T[T<self.T_min] = self.T_min
        if(numpy):
            return T.detach().cpu().numpy()
        else:
            return T