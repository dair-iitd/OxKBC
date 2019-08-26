import logging
import os
import pickle

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from IPython.core.debugger import Pdb
import scipy.sparse as sp

def get_data_loaders(args):
    #need to change ys to account for args.exclude_t_ids
    stats_file = os.path.join(args.output_path,'stats')
    #stats_file = args.training_data_path + '_stats' 
    train_loader = None 
    #if args.supervision in ['un','semi']:
    train_ds = SelectionModuleDataset(args.training_data_path, 
                    base_model_file=args.base_model_file,
                    each_input_size=args.each_input_size, 
                    use_ids=args.use_ids, mode='train', 
                    stats_file_path=stats_file, args = args, labels = 0)
                    
    train_loader = DataLoader(train_ds,batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if args.val_data_path != '':
        val_ds = SelectionModuleDataset(args.val_data_path, 
                    base_model_file=args.base_model_file, 
                    each_input_size=args.each_input_size,
                    use_ids=args.use_ids, mode='eval', 
                    labels_file_path=args.val_labels_path,
                    num_labels = args.num_templates + 1,
                    stats_file_path=stats_file, args = args, labels = 1)

        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    train_anot_loader = None
    if args.supervision in ['semi','sup']:
        #create another loader - train annotated
        train_anot_ds = SelectionModuleDataset(args.labelled_training_data_path, 
                    base_model_file=args.base_model_file,
                    each_input_size=args.each_input_size, 
                    use_ids=args.use_ids, mode='train', 
                    labels_file_path = args.train_labels_path,
                    stats_file_path=stats_file, args = args, labels = 1)
        train_anot_loader = DataLoader(train_anot_ds, batch_size = args.batch_size, shuffle=True)


    if args.supervision == 'un':
        return (train_loader, val_loader, None)
    elif args.supervision == 'semi':
        return (train_loader, val_loader, train_anot_loader)
    elif args.supervision == 'sup':
        return (None, val_loader, train_anot_loader)
    else:
        raise

    

class SelectionModuleDataset(torch.utils.data.Dataset):
    def __init__(self, input_file_path, base_model_file, each_input_size=7, use_ids=False, mode='train', stats_file_path=None, labels_file_path=None, num_labels = 0, args = None, labels = 0):
        self.args = args
        self.is_labelled = labels
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

        self.raw_data = data.copy()
        #my_score, max_score, simi, rank, conditional_rank, mean, std
        if stats_file_path is None or (not os.path.exists(stats_file_path)):
            means = np.mean(data[:, self.start_idx:-1], axis=0).reshape(1, -1)
            stds = np.std(data[:, self.start_idx:-1], axis=0).reshape(1, -1)
            for i in range(means.shape[1]//each_input_size):
                idx = i*each_input_size
                #temp = np.concatenate(
                #    (data[:, idx], data[:, idx+1])).reshape(-1)
                temp = data[:,self.start_idx + idx].reshape(-1)
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
            #pickle.dump({'mean': means, 'std': stds},
            #            open(input_file_path+'_stats', 'wb'))
            pickle.dump({'mean': means, 'std': stds},
                        open(stats_file_path, 'wb'))
            logging.info("Calculated stats of the given input file, dumped stats to {}".format(
                input_file_path+'_stats'))
        else:
            stats = pickle.load(open(stats_file_path, 'rb'))
            means = stats['mean']
            stds = stats['std']

        if(base_model_file is not None and os.path.exists(base_model_file)):
            with open(base_model_file, 'rb') as f:
                self.bm = pickle.load(f)
            logging.info('Loaded base model from {}'.format(base_model_file))
        elif self.use_ids:
            logging.error(
                'Base model file not present at {}'.format(base_model_file))
            raise Exception(
                'Please provide base model file path - {}', str(base_model_file))

        logging.debug('means: {} '.format( means))
        logging.debug('stds: {} '.format( stds))
        data[:, self.start_idx:-1] = (data[:, self.start_idx:-1]-means)/stds
        self.data = data

        logging.info('Normalized and successfully loaded data. Size of dataset = {}'.format(
            self.data.shape))

        #Pdb().set_trace()
        if self.is_labelled and self.args.exclude_t_ids is not None and len(self.args.exclude_t_ids) > 0:
            logging.info("Dataset: Excluding following template ids from target and mapping them to 0: {}".format(','.join(map(str,self.args.exclude_t_ids))))
            
            o2n = np.array(args.o2n)
            self.data[:,-1] = o2n[self.data[:,-1].astype(int)]
            #for i, tid in enumerate(self.args.exclude_t_ids):
            #    self.data[:, -1][self.data[:,-1] == tid] = 0



        self.Y = None 
        if labels_file_path is not None:
            exclude_t_ids_set = set(self.args.exclude_t_ids)
            logging.info("Multi-label evaluation on. Labels in : {}".format(labels_file_path))
            fh = open(labels_file_path,'r')
            lines = fh.readlines()
            lines = [list(map(int,line.strip().strip(',').split(','))) for line in lines]
            row_idx, col_idx, val_idx = [], [], []
            for i,l_list in enumerate(lines):
                l_list = [args.o2n[old_tid] for old_tid in l_list]
                l_list = set(l_list) # remove duplicates
                for y in l_list:
                    row_idx.append(i)
                    col_idx.append(y)
                    val_idx.append(1)
            m = max(row_idx) + 1 
            n = max(col_idx) + 1 
            n = max(n,num_labels - len(self.args.exclude_t_ids))
            self.Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n))
            assert(m == len(self.data))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.Y is not None:
            #multilabel evaluation
            y = np.ravel(self.Y[idx].todense())
        else:
            y = self.data[idx][-1]
        #
        if(self.use_ids):
            e1_embed = np.concatenate((self.bm['entity_real'][int(
                self.data[idx][0])], self.bm['entity_type'][int(self.data[idx][0])]))
            e2_embed = np.concatenate((self.bm['entity_real'][int(
                self.data[idx][2])], self.bm['entity_type'][int(self.data[idx][2])]))
            r_embed = np.concatenate((self.bm['rel_real'][int(self.data[idx][1])], self.bm['head_rel_type'][int(
                self.data[idx][1])], self.bm['tail_rel_type'][int(self.data[idx][1])]))
            return (np.concatenate((e1_embed, r_embed, e2_embed, self.data[idx][self.start_idx:-1])), y, idx)
        else:
            return (self.data[idx][self.start_idx:-1], y, idx)
