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
    print("batch size -->", args.batch_size)
    sfp = args.training_data_path + '_stats'
    train_loader = DataLoader(SelectionModuleDataset(args.training_data_path, base_model_file=args.base_model_file,
                                                     mode='train', stats_file_path=sfp), batch_size=args.batch_size, shuffle=True)
    val_loader = None
    if args.val_data_path != '':
        val_loader = DataLoader(SelectionModuleDataset(args.val_data_path, base_model_file=args.base_model_file,
                                                       mode='eval', stats_file_path=sfp), batch_size=args.batch_size, shuffle=False)

    return (train_loader, val_loader)


class SelectionModuleDataset(torch.utils.data.Dataset):
    def __init__(self, input_file_path, base_model_file, mode='train', stats_file_path=None):
        if '.txt' in input_file_path:
            print("Loading txt file")
            data = np.loadtxt(input_file_path, delimiter=',', dtype=float)
        else:
            print("Loading pkl file")
            data = pickle.load(open(input_file_path, 'rb'))

        if mode != 'train' and (stats_file_path is None or (not os.path.exists(stats_file_path))):
            raise Exception(
                'please provide stats file path for eval / test mode - {}', str(stats_file_path))

        self.start_idx = 0 if base_model_file == '' else 3

        if stats_file_path is None or (not os.path.exists(stats_file_path)):
            means = np.mean(data[:, self.start_idx:-1], axis=0).reshape(1, -1)
            stds = np.std(data[:, self.start_idx:-1], axis=0).reshape(1, -1)
            pickle.dump({'mean': means, 'std': stds},
                        open(input_file_path+'_stats', 'wb'))
        else:
            d = pickle.load(open(stats_file_path, 'rb'))
            means = d['mean']
            stds = d['std']
        self.bm = None
        if(base_model_file != ''):
            with open(base_model_file, 'rb') as f:
                self.bm = pickle.load(f)

        data[:, self.start_idx:-1] = (data[:, self.start_idx:-1]-means)/stds
        self.data = data
        print("Loaded data", self.data)
        # exit(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if(self.bm is not None):
            # print(self.data[idx])
            # print(self.bm['entity_real'][self.data[idx][0]])
            e1_embed = np.concatenate((self.bm['entity_real'][int(self.data[idx][0])], self.bm['entity_type'][int(self.data[idx][0])]))
            e2_embed = np.concatenate((self.bm['entity_real'][int(self.data[idx][2])], self.bm['entity_type'][int(self.data[idx][2])]))
            r_embed = np.concatenate((self.bm['rel_real'][int(self.data[idx][1])], self.bm['head_rel_type'][int(self.data[idx][1])], self.bm['tail_rel_type'][int(self.data[idx][1])]))
            return (np.concatenate((e1_embed, r_embed, e2_embed, self.data[idx][self.start_idx:-1])), self.data[idx][-1], idx)
        else:
            return (self.data[idx][self.start_idx:-1], self.data[idx][-1], idx)
