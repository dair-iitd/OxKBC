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
    if args.mil:
        return SelectionModuleMIL(args.each_input_size,args.num_templates,args.hidden_unit_list)
    else:
        return SelectionModule(args.input_size,args.output_size, args.hidden_unit_list)
    
class SelectionModule(nn.Module):
    def __init__(self,input_size, output_size, hidden_unit_list): 
        super(SelectionModule, self).__init__()
        
        
        module_list =[]

        prev=input_size
        for i in range(len(hidden_unit_list)):
            module_list.append(nn.Sequential(nn.Linear(prev,hidden_unit_list[i]), nn.ReLU()))
            prev=hidden_unit_list[i]

        module_list.append(nn.Sequential(nn.Linear(prev,output_size), nn.Softmax(dim=1)))
        #self.mlp=nn.ModuleList(module_list)
        self.mlp=nn.Sequential(*module_list)
        #self.softmax = nn.Softmax(dim=1)

        if settings.cuda:
            self = self.cuda()
        

    def forward(self,x):
        x = x.float()
        # print(x)
        # exit(0)
        #x = self.mlp(x)
        #for layer in self.mlp:
        #    x = layer(x)

        #return x
        return self.mlp(x)


class SelectionModuleMIL(nn.Module):
    def __init__(self,input_size, num_templates, hidden_unit_list): 
        super(SelectionModuleMIL, self).__init__()
        module_list =[]
        prev=input_size
        for i in range(len(hidden_unit_list)):
            module_list.append(nn.Sequential(nn.Linear(prev,hidden_unit_list[i]), nn.ReLU()))
            prev=hidden_unit_list[i]

        module_list.append(nn.Sequential(nn.Linear(prev,1), nn.Sigmoid()))
        #self.mlp=nn.ModuleList(module_list)
        self.mlp=nn.Sequential(*module_list)
        self.num_templates = num_templates
        self.input_size = input_size 
        #self.softmax = nn.Softmax(dim=1)
        self.others_template_parameters = nn.Parameter(torch.randn(self.input_size))
        if settings.cuda:
            self = self.cuda()
        

    def forward(self,x):
        x = x.float()
        ts = []
        for i in range(self.num_templates):
            ts.append(self.mlp(x[:,i*self.input_size : (i+1)*self.input_size ]))

        others_score = self.mlp(self.others_template_parameters)
        template_scores = torch.cat(ts ,dim=1)
        template_scores = torch.cat((others_score.expand(template_scores.size(0),1), template_scores), dim=1)
        # print(x)
        # exit(0)
        #x = self.mlp(x)
        #for layer in self.mlp:
        #    x = layer(x)

        #return x
        return template_scores