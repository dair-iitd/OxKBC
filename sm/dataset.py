import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.utils as tutils
import utils
from IPython.core.debugger import Pdb 
import pickle
import copy
import numpy as np
# import data_samplers 
import itertools

def get_data_loaders(args):
    return DataLoader(SelectionModuleDataset(args.training_data_path),batch_size=args.batch_size,shuffle=True)


class SelectionModuleDataset(torch.utils.data.Dataset):    
    def __init__(self, input_file_path):
        data = np.loadtxt(input_file_path,delimiter=',',dtype=float)
        means = np.mean(data[:,:-1],axis=0).reshape(1,-1)
        stds = np.std(data[:,:-1],axis=0).reshape(1,-1)
        data[:,:-1] = (data[:,:-1]-means)/stds
        self.data=data
        print("Loaded data" , self.data)
        # exit(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return (self.data[idx][:-1], self.data[idx][-1], idx)
    
        
        
