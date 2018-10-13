import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.utils as tutils
import utils
from IPython.core.debugger import Pdb 
import pickle
import copy
import data_samplers 
import itertools

def get_data_loaders(args):
    return (SelectionModuleDataset(args.training_data_path),)


class SelectionModuleDataset(torch.utils.data.Dataset):    
    def __init__(self, input_file_path):
        self.data  = pickle.load(open(input_file_path,'rb'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return (self.data[idx][:-1], self.data[idx][-1], idx)
    
        
        
