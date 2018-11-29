import logging
import os
import pickle

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset


def get_data_loaders(args):
    stats_file = args.training_data_path + '_stats'
    train_loader = DataLoader(SelectionModuleDataset(args.training_data_path, base_model_file=args.base_model_file,
                                                     each_input_size=args.each_input_size, use_ids=args.use_ids, mode='train', stats_file_path=stats_file), batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if args.val_data_path != '':
        val_loader = DataLoader(SelectionModuleDataset(args.val_data_path, base_model_file=args.base_model_file, each_input_size=args.each_input_size,
                                                       use_ids=args.use_ids, mode='eval', stats_file_path=stats_file), batch_size=args.batch_size, shuffle=False)

    return (train_loader, val_loader)


class SelectionModuleDataset(torch.utils.data.Dataset):
    def __init__(self, input_file_path, base_model_file, each_input_size=7, use_ids=False, mode='train', stats_file_path=None):
        if '.txt' in input_file_path:
            logging.info("Input file {} is a txt file".format(input_file_path))
            data = np.loadtxt(input_file_path, delimiter=',', dtype=float)
        else:
            logging.info("Input file {} is a pkl file".format(input_file_path))
            data = pickle.load(open(input_file_path, 'rb'))
        logging.info("Loaded Input")

        if mode != 'train' and (stats_file_path is None or (not os.path.exists(stats_file_path))):
            logging.error(
                'Please provide stats file path for eval / test mode - {}'.format(str(stats_file_path)))
            raise Exception(
                'Please provide stats file path for eval / test mode - {}', str(stats_file_path))

        self.use_ids = use_ids
        self.start_idx = 3

        if stats_file_path is None or (not os.path.exists(stats_file_path)):
            means = np.mean(data[:, self.start_idx:-1], axis=0).reshape(1, -1)
            stds = np.std(data[:, self.start_idx:-1], axis=0).reshape(1, -1)
            for i in range((means.shape[1]-self.start_idx)//each_input_size):
                idx = i*each_input_size + self.start_idx
                temp = np.concatenate(
                    (data[:, idx], data[:, idx+1])).reshape(-1)
                new_mean = np.mean(temp)
                new_std = np.std(temp)
                # Normalizing my_score and max_score using same distribution
                means[0, idx] = new_mean
                means[0, idx+1] = new_mean
                stds[0, idx] = new_std
                stds[0, idx+1] = new_std
                # Not normalizing rank and conditional rank
                means[0, idx+3] = 0
                means[0, idx+4] = 0
                stds[0, idx+3] = 1
                stds[0, idx+4] = 1
            pickle.dump({'mean': means, 'std': stds},
                        open(input_file_path+'_stats', 'wb'))
            logging.info("Calculated stats of the given input file, dumped stats to {}".format(
                input_file_path+'_stats'))
        else:
            stats = pickle.load(open(stats_file_path, 'rb'))
            means = stats['mean']
            stds = stats['std']

        if(base_model_file != '' and os.path.exists(base_model_file)):
            with open(base_model_file, 'rb') as f:
                self.bm = pickle.load(f)
            logging.info('Loaded base model from {}'.format(base_model_file))
        else:
            logging.error(
                'Base model file not present at {}'.format(base_model_file))
            raise Exception(
                'Please provide base model file path - {}', str(base_model_file))

        data[:, self.start_idx:-1] = (data[:, self.start_idx:-1]-means)/stds
        self.data = data

        logging.info('Normalized and successfully loaded data. Size of dataset = {}'.format(
            self.data.shape))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if(self.use_ids):
            e1_embed = np.concatenate((self.bm['entity_real'][int(
                self.data[idx][0])], self.bm['entity_type'][int(self.data[idx][0])]))
            e2_embed = np.concatenate((self.bm['entity_real'][int(
                self.data[idx][2])], self.bm['entity_type'][int(self.data[idx][2])]))
            r_embed = np.concatenate((self.bm['rel_real'][int(self.data[idx][1])], self.bm['head_rel_type'][int(
                self.data[idx][1])], self.bm['tail_rel_type'][int(self.data[idx][1])]))
            return (np.concatenate((e1_embed, r_embed, e2_embed, self.data[idx][self.start_idx:-1])), self.data[idx][-1], idx)
        else:
            return (self.data[idx][self.start_idx:-1], self.data[idx][-1], idx)
