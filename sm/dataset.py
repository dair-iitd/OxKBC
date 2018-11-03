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
    print("batch size -->",args.batch_size)
    sfp = args.training_data_path + '_stats'
    train_loader = DataLoader(SelectionModuleDataset(args.training_data_path,mode = 'train', stats_file_path=sfp),batch_size=args.batch_size,shuffle=True)
    val_loader= None
    if args.val_data_path != '':
        val_loader = DataLoader(SelectionModuleDataset(args.val_data_path,mode='eval', stats_file_path= sfp),batch_size=args.batch_size,shuffle=False)
    
    return (train_loader, val_loader)

class SelectionModuleDataset(torch.utils.data.Dataset):    
    def __init__(self, input_file_path, mode = 'train',stats_file_path = None):
        data = np.loadtxt(input_file_path,delimiter=',',dtype=float)
        if mode != 'train' and (stats_file_path is None or (not os.path.exists(stats_file_path))):
            raise Exception('please provide stats file path for eval / test mode - {}', str(stats_file_path))
            
        if stats_file_path is None or (not os.path.exists(stats_file_path)):
            means = np.mean(data[:,:-1],axis=0).reshape(1,-1)
            stds = np.std(data[:,:-1],axis=0).reshape(1,-1)
            pickle.dump({'mean': means, 'std': stds}, open(input_file_path+'_stats','wb'))
        else:
            d = pickle.load(open(stats_file_path,'rb'))
            means = d['mean']
            stds = d['std']

        data[:,:-1] = (data[:,:-1]-means)/stds
        self.data=data
        print("Loaded data" , self.data)
        # exit(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return (self.data[idx][:-1], self.data[idx][-1], idx)
    
        
        
