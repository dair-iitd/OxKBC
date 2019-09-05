import tempfile
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
import dataset1 as dataset
import models
from collections import Counter
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

parser.add_argument('--default_value',help='default value of template score when it is undefined?', default = 0, type=float)
parser.add_argument('--exclude_default',help='should default value be excluded while computing stats?', default = 0, type=int)

parser.add_argument('--hidden_unit_list', nargs='*', type=int, required=False,default=[], help='number of hidden neurons in each layer')


parser.add_argument('--train_ml',help='should use multi label loss?', default = 1, type=int)
parser.add_argument('--eval_ml',help='should eval multi label ?', default = 1, type=int)

parser.add_argument('--out_file', help='output file where analysis is written',required = False,
                    type=str, default = None)

parser.add_argument('--expdir', help='load checkpoints and model from this dir',required = True,
                    type=str)

parser.add_argument('--checkpoint_file', help='load checkpoints and model from this file',required = True,
                    type=str)

#train_r0.125_p1_n-2.0_i4_k0.0_best_checkpoint.pth0
#expdir = 'temp/fb15k_ex25_sl_hul0'


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

#args.output_path = tempfile.TemporaryDirectory().name
os.makedirs(args.output_path)


train_loader, val_loader, labelled_train_loader = dataset.get_data_loaders(args)

model = models.select_model(args)

def SIDX(template_id):
    return (3 + (template_id-1)*7)

def EIDX(template_id):
    return (3 + (template_id-1)*7 + 7)

# for i in val_loader.dataset.raw_data[:,SIDX(6)]:
# 	print (i)
# print (val_loader.dataset.raw_data[:,SIDX(2)])

#my_score, max_score, similarity, rank, conditional_rank, mean, std
#tid = 1
#for i in val_loader.dataset.data[:,SIDX(tid):SIDX(tid)+7+1]:
#    for j in range(7):
#        print (str(i[j])+" ",end="")
#    print ("")


# for ln, loader in zip(['TRAIN UN','VAL','TRAIN LAB'],[train_loader, val_loader, labelled_train_loader]):
#     for i in range(1,7):
#         print('{}, Temp: {}, Mean: {}, Max Mean: {}'.format(ln,i,loader.dataset.raw_data[:,SIDX(i)].mean(),loader.dataset.raw_data[:,SIDX(i)+1].mean()))


# for ln, loader in zip(['TRAIN UN','VAL','TRAIN LAB'],[train_loader, val_loader, labelled_train_loader]):
#     for i in range(1,7):
#         print('{}, Temp: {}, Mean: {}, Max Mean: {}'.format(ln,i,loader.dataset.data[:,SIDX(i)].mean(),loader.dataset.raw_data[:,SIDX(i)+1].mean()))

def printmat(mat,file=None):
    print('\n'.join([','.join(map(str,x.ravel())) for x in mat]), file = file)

import os
import torch
d = {}
for i in range(1,6):
    chp = os.path.join(args.expdir,'run_{}'.format(i),'train',args.checkpoint_file)
    d[i] = torch.load(chp)


wt= 'mlp.0.0.weight'
bias = 'mlp.0.0.bias'
dp = 'others_template_parameters'

loglin = d[1]['model'][wt].shape == (1,7)
if args.out_file is None:
    fh = None
else:
    fh = open(args.out_file,'w')
#print model wts if only a log linear model
if loglin:
    print("WEIGHTS in LOG LINEAR MODEL:",file =fh)
    print('My Score, Max Score, Similarity, Rank, Conditional Rank, Mean, Std',file=fh)
    printmat([d[i]['model'][wt].cpu().numpy() for  i in range(1,6)],file =fh )

i = 1
loader = train_loader 
model.load_state_dict(d[1]['model'])

sf = os.path.join(args.expdir,'run_{}'.format(i), 'train','stats')
stats = pickle.load(open(sf,'rb'))
print("",file = fh)
print("STATS in THE STATS FILE", file = fh)
print("MEAN", file = fh)
printmat(stats['mean'][0].reshape(6,7),file =fh)
print("STD", file = fh)
printmat(stats['std'][0].reshape(6,7), file = fh)

meanmat = loader.dataset.stats['mean'][0].reshape(6,7)
stdmat = loader.dataset.stats['std'][0].reshape(6,7)
assert (meanmat == stats['mean'][0].reshape(6,7)).all()
assert (stdmat == stats['std'][0].reshape(6,7)).all()

"""
for ln, loader in zip(['TRAIN UN','VAL','TRAIN LAB'],[train_loader, val_loader, labelled_train_loader]):
    print(ln, loader.dataset.data.mean(axis=0)[3:-1].reshape(6,7))

for ln, loader in zip(['TRAIN UN','TEST','TRAIN LAB'],[train_loader, val_loader, labelled_train_loader]):
    print(ln, (d[1]['model'][wt].expand(6,7)*torch.tensor(loader.dataset.data.mean(axis=0)[3:-1].reshape(6,7)).float().cuda()).sum(dim=1))


data_indx = 10
ln = 'TEST'
loader = val_loader 
print(ln, (d[1]['model'][wt].expand(6,7)*torch.tensor(loader.dataset.data[data_indx][3:-1].reshape(6,7)).float().cuda()).sum(dim=1))
loader.dataset.data[data_indx][3:-1].reshape(6,7)
"""

ln = 'TEST'
loader = val_loader 
### Avg normalized features
avg_features = loader.dataset.data[:,3:-1].mean(axis=0).reshape(args.num_templates,args.each_input_size)
avg_features = np.concatenate((d[1]['model'][dp].cpu().numpy().reshape(1,args.each_input_size), avg_features),axis=0)

print("\nAVG FEATURE", file = fh)

print('My Score, Max Score, Similarity, Rank, Conditional Rank, Mean, Std',file=fh)
printmat(avg_features,file =fh)

if loglin:
    score_of_avg_features = d[1]['model'][wt].expand(avg_features.shape).cpu().numpy()*avg_features
    #score_of_avg_features = np.concatenate((score_of_avg_features,d[1]['model'][dp].cpu().numpy().reshape(1,7)),axis=0)
    score_of_avg_features = np.concatenate((score_of_avg_features,np.repeat(d[1]['model'][bias].cpu().numpy(), score_of_avg_features.shape[0]).reshape(score_of_avg_features.shape[0],1)), axis = 1)
    score_of_avg_features = np.concatenate((score_of_avg_features,score_of_avg_features.sum(axis=1).reshape(-1,1)), axis=1)
    print("For LOG LINEAR MODEL: AVG FEATURE * WT, BIAS, Total", file = fh)
    print('My Score, Max Score, Similarity, Rank, Conditional Rank, Mean, Std, Bias, Total',file=fh)
    printmat(score_of_avg_features, file = fh)

Y = np.array(loader.dataset.Y.todense())

model = model.cuda()
with torch.no_grad():
    model.eval()
    for template_id in range(0,args.num_templates+1):
        if template_id not in args.exclude_t_ids: 
            new_template_id = args.o2n[template_id]
            pos_ind = (Y[:,new_template_id] == 1)
            neg_ind = (Y[:,new_template_id] != 1)
            pos_data = torch.FloatTensor(loader.dataset.data[pos_ind,3:-1]).cuda()
            neg_data = torch.FloatTensor(loader.dataset.data[neg_ind,3:-1]).cuda()
            
            print("TEMPLATE {}".format(template_id), file = fh)
            if pos_data.size(0) > 0: 
                pos_score = model(pos_data)
                pos_prediction = pos_score.max(dim=1)[1]
                pos_counter = Counter(pos_prediction.detach().cpu().numpy())  
                print("Total Positives: {}.  Predicted as: ".format(pos_data.size(0)), file = fh)
                printmat(np.array([[pos_counter[args.o2n[x]] for x in range(0,args.num_templates+1)]]),file =fh)
                avg_pos_score = pos_score.mean(dim=0)
                avg_pos_feature = pos_data.mean(dim=0)
                pos_score_avg_feat = model(avg_pos_feature.unsqueeze(0))
                print("AVG score POSITIVE ", file = fh)
                printmat(avg_pos_score.detach().cpu().numpy()[args.o2n].reshape(1,-1),file = fh)
                print("score At Avg Features POSITIVE ", file = fh)
                printmat(pos_score_avg_feat.detach().cpu().numpy()[0][args.o2n].reshape(1,-1),file = fh)
                print("AVG FEATUREs when POSITIVE", file = fh)
                printmat(avg_pos_feature.reshape(args.num_templates,args.each_input_size).cpu().numpy(), file= fh)
    
            if neg_data.size(0) > 0: 
                neg_score = model(neg_data)
                neg_prediction = neg_score.max(dim=1)[1]
                neg_counter = Counter(neg_prediction.detach().cpu().numpy())
                avg_neg_score = neg_score.mean(dim=0)
                avg_neg_feature = neg_data.mean(dim=0)
                neg_score_avg_feat = model(avg_neg_feature.unsqueeze(0))
                print("Total Negatives: {}.  Predicted as: ".format(neg_data.size(0)), file = fh)
                printmat(np.array([[neg_counter[args.o2n[x]] for x in range(0,args.num_templates+1)]]),file =fh)
                print("AVG score NEGATIVE ", file = fh)
                printmat(avg_neg_score.detach().cpu().numpy()[args.o2n].reshape(1,-1),file = fh)
                print("score At Avg Features NEGATIVE ", file = fh)
                printmat(neg_score_avg_feat.detach().cpu().numpy()[0][args.o2n].reshape(1,-1),file = fh)
                print("AVG FEATUREs when NEGATIVE", file = fh)
                printmat(avg_neg_feature.reshape(args.num_templates, args.each_input_size).cpu().numpy(), file= fh)
    

fh.close()

"""
loader = train_loader

valid_indx = {}
for i in range(1,7):
    valid_indx[i] = np.logical_not(loader.dataset.raw_data[:,SIDX(i)] == 0)




def_stats = []
for i in range(1,7):
    def_stats.append(loader.dataset.raw_data[valid_indx[i],SIDX(i):EIDX(i)].min(axis=0))

printmat(def_stats) 
"""






