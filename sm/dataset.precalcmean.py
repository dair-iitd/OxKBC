import logging
import os
import pickle

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from IPython.core.debugger import Pdb
import scipy.sparse as sp
import utils
#DEFAULT_VALUE = 0

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
                    stats_file_path=stats_file, args = args, labels = 0,is_multi_label = None)
                    
    train_loader = DataLoader(train_ds,batch_size=args.batch_size, shuffle=True)

    val_loader = None
    if args.val_data_path != '':
        val_ds = SelectionModuleDataset(args.val_data_path, 
                    base_model_file=args.base_model_file, 
                    each_input_size=args.each_input_size,
                    use_ids=args.use_ids, mode='eval', 
                    labels_file_path=args.val_labels_path,
                    num_labels = args.num_templates + 1,
                    stats_file_path=stats_file, args = args, labels = 1, is_multi_label = args.eval_ml)

        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    train_anot_loader = None
    if args.supervision in ['semi','sup']:
        #create another loader - train annotated
        train_anot_ds = SelectionModuleDataset(args.labelled_training_data_path, 
                    base_model_file=args.base_model_file,
                    each_input_size=args.each_input_size, 
                    use_ids=args.use_ids, mode='train', 
                    num_labels = args.num_templates + 1,
                    labels_file_path = args.train_labels_path,
                    stats_file_path=stats_file, args = args, labels = 1, is_multi_label = args.train_ml)
        train_anot_loader = DataLoader(train_anot_ds, batch_size = args.batch_size, shuffle=True)


    if args.supervision == 'un':
        return (train_loader, val_loader, None)
    elif args.supervision == 'semi':
        return (train_loader, val_loader, train_anot_loader)
    elif args.supervision == 'sup':
        return (None, val_loader, train_anot_loader)
    else:
        raise


def change_default_scores(data, start_idx, each_input_size, args):
    #my_score, max_score, simi, rank, conditional_rank, mean, std
    #only change : my_score
    #when: when my_score == 0
    def SIDX(template_id):
        return (start_idx + template_id*each_input_size)

    num_templates = (data.shape[1] - start_idx -1 )//each_input_size
    all_means = []
    all_stds  = []
    for i in range(num_templates):
        indx = data[:,SIDX(i)] == 0
        data[indx,SIDX(i)] = args.default_value
        if args.exclude_default == 1:
            this_means = data[np.logical_not(indx), SIDX(i): SIDX(i+1)].mean(axis=0).reshape(1,-1)
            this_stds = data[np.logical_not(indx), SIDX(i): SIDX(i+1)].std(axis=0).reshape(1,-1)
            this_max = data[np.logical_not(indx), SIDX(i): SIDX(i+1)].max(axis=0).reshape(1,-1)
        else:
            this_means = data[:, SIDX(i): SIDX(i+1)].mean(axis=0).reshape(1,-1)
            this_stds = data[:, SIDX(i): SIDX(i+1)].std(axis=0).reshape(1,-1)
            this_max = data[:, SIDX(i): SIDX(i+1)].max(axis=0).reshape(1,-1)
        #
        this_means[0,1] = this_means[0,0]
        this_stds[0, 1] = this_stds[0, 0]
        for j in [2,4]: 
        #for j in [3,4]: 
            this_means[0,j] = 0
            this_stds[0,j] = 1
        #
        this_means[0,3] = 0
        this_stds[0,3] = this_max[0,3]
        
        
        this_means[0,5] = this_means[0,0]
        this_stds[0,5] = this_stds[0,0]
        
        this_means[0,6] = 0
        this_stds[0,6] = this_stds[0,0]
        
        all_means.append(this_means)
        all_stds.append(this_stds)
    #
    return data, {'mean': np.concatenate(all_means, axis=1), 'std': np.concatenate(all_stds, axis=1)}


class SelectionModuleDataset(torch.utils.data.Dataset):
    def __init__(self, input_file_path, base_model_file, each_input_size=7, use_ids=False, mode='train', stats_file_path=None, labels_file_path=None, num_labels = 0, args = None, labels = 0,is_multi_label = 1):
        is_multi_label = ((labels_file_path is not None) and (is_multi_label == 1))
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
        data,self.stats = change_default_scores(data, self.start_idx, each_input_size, args)
        #my_score, max_score, simi, rank, conditional_rank, mean, std
        if stats_file_path is None or (not os.path.exists(stats_file_path)):
            pickle.dump(self.stats,
                        open(stats_file_path, 'wb'))
            logging.info("Calculated stats of the given input file, dumped stats to {}".format(
                input_file_path+'_stats'))
        else:
            self.stats = pickle.load(open(stats_file_path, 'rb'))
        #
        means = self.stats['mean']
        stds = self.stats['std']

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
        y_single = []
        
        if labels_file_path is not None:
            logging.info("Reading Labels from an External file: {}".format(labels_file_path))
            exclude_t_ids_set = set(self.args.exclude_t_ids)
            fh = open(labels_file_path,'r')
            lines = fh.readlines()
            lines = [list(map(int,line.strip().strip(',').split(','))) for line in lines]
            row_idx, col_idx, val_idx = [], [], []
            for i,l_list in enumerate(lines):
                l_list = [args.o2n[old_tid] for old_tid in l_list]
                l_list = utils.clean_label_list(l_list)
                this_y = l_list[0]
                y_single.append(this_y)
                #
                for enum_y,y in enumerate(reversed(l_list)):
                    row_idx.append(i)
                    col_idx.append(y)
                    val_idx.append(1)
            m = max(row_idx) + 1 
            n = max(col_idx) + 1 
            n = max(n,num_labels - len(self.args.exclude_t_ids))
            if is_multi_label:  
                logging.info("Multi-label evaluation on. Labels in : {}".format(labels_file_path))
                self.Y = sp.csr_matrix((val_idx, (row_idx, col_idx)), shape=(m, n)).todense()
            else:
                logging.info("Single -label evaluation. Labels in : {}".format(labels_file_path))
                self.data[:,-1] = np.array(y_single)
            assert(m == len(self.data))
            #
            #

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.Y is not None:
            #multilabel evaluation
            y = np.ravel(self.Y[idx])
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
