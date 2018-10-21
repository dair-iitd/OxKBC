import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils as tutils
import utils
import numpy as np
import os
import math
import settings

def select_model(args):
    return SelectionModule(args.input_size,args.output_size, args.hidden_unit_list)
    
class SelectionModule(nn.Module):
    def __init__(self,input_size, output_size, hidden_unit_list): 
        super(SelectionModule, self).__init__()
        
        
        module_list =[]

        prev=input_size
        for i in range(len(hidden_unit_list)):
            module_list.append(nn.Sequential(nn.Linear(prev,hidden_unit_list[i]), nn.ReLU()))
            prev=hidden_unit_list[i]

        module_list.append(nn.Sequential(nn.Linear(prev,output_size)))
        self.mlp=nn.ModuleList(module_list)
        self.softmax = nn.Softmax(dim=1)

        if settings.cuda:
            self = self.cuda()
        

    def forward(self,x):
        x = x.float()
        # print(x)
        # exit(0)
        for layer in self.mlp:
            x = layer(x)
        return self.softmax(x)
        # return self.mlp(x)