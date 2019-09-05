
import argparse
import logging
import os
import pickle
import pprint
import time

import numpy as np
import yaml
import utils
import settings
import dataset

parser = argparse.ArgumentParser()
parser.add_argument('--training_data_path',
                    help="Training data path (pkl file)", type=str, 
                    default = '../logs/fb15k/sm_with_id.data.pkl')

parser.add_argument('--labelled_training_data_path',
                    help="Labelled Training data path (pkl file)", type=str,
                    default = '../logs/fb15k/sm_sup_train_with_id.pkl')

parser.add_argument(
    '--val_data_path', help="Validation data path in the same format as training data", type=str, default='../logs/fb15k/test_hits1_single_label_sm.data.pkl.pkl')
parser.add_argument(
    '--val_labels_path', help="Validation data Labels path for multi-label evaluation", type=str, default='../data/fb15k/test/test_hits_1_ordered_y.txt')

parser.add_argument(
    '--train_labels_path', help="Training data Labels path for multi-label training", type=str, default='../logs/fb15k/sm_sup_train_multilabels.txt')

parser.add_argument('--batch_size', help='batch size',
                    type=int, default=256)


parser.add_argument('--exp_name', help='Experiment name',
                    type=str, default='data_analysis')
parser.add_argument(
    '--output_path', help='Output path to store models, and logs', type=str, required=True)

parser.add_argument('--each_input_size',
                    help='Input size of each template', type=int, default=7)

parser.add_argument(
    '--num_templates', help='number of templates excluding other', type=int, default=6)

parser.add_argument('--use_ids', help='Use embeddings of entity and relations while training',
                    action='store_true', default=False)

parser.add_argument('--mil', help='Use MIL model',
                    action='store_true', default=True)

parser.add_argument('--exclude_t_ids', nargs='+', type=int, required=False,default=[], help='List of templates to be excluded while making predictions')
 
parser.add_argument('--base_model_file',
                        help="Base model dump for loading embeddings", type=str, default='')

parser.add_argument('--supervision', help='possible values - un, semi, sup',
                        type=str, default='semi')

args = parser.parse_args()
config = {}
config.update(vars(args))
args = utils.Map(config)
o2n , n2o = utils.get_template_id_maps(args.num_templates, args.exclude_t_ids)
args.o2n = o2n
args.n2o = n2o

for key in ['train_labels_path','val_labels_path']:
    if args[key]  == 'None':
        args[key] = None

    
settings.set_settings(args)
 


train_loader, val_loader, labelled_train_loader = dataset.get_data_loaders(args)

def SIDX(template_id):
    return (3 + (template_id-1)*7)

def EIDX(template_id):
    return (3 + (template_id-1)*7 + 7)

# for i in val_loader.dataset.raw_data[:,SIDX(6)]:
# 	print (i)
# print (val_loader.dataset.raw_data[:,SIDX(2)])

#my_score, max_score, similarity, rank, conditional_rank, mean, std
for ln, loader in zip(['TRAIN UN','VAL','TRAIN LAB'],[train_loader, val_loader, labelled_train_loader]):
    for i in range(1,7):
        print('{}, Temp: {}, Mean: {}, Max Mean: {}'.format(ln,i,loader.dataset.raw_data[:,SIDX(i)].mean(),loader.dataset.raw_data[:,SIDX(i)+1].mean()))


for ln, loader in zip(['TRAIN UN','VAL','TRAIN LAB'],[train_loader, val_loader, labelled_train_loader]):
    for i in range(1,7):
        print('{}, Temp: {}, Mean: {}, Max Mean: {}'.format(ln,i,loader.dataset.data[:,SIDX(i)].mean(),loader.dataset.raw_data[:,SIDX(i)+1].mean()))





