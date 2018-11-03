from datetime import datetime as dt
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os
import sys
import time
import argparse
import pickle
import yaml
import torch
import shutil
import utils
from IPython.core.debugger import Pdb
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from sklearn import metrics
import settings
from collections import Counter
import math
import functools

def get_evaluation_function(args):
    return Loss(args) 


class Loss(object):
    def __init__(self,args):
        self.args = args
        self.criterion = nn.NLLLoss(weight = torch.Tensor([1.0,20.0]))


        l = [args.mil_reward for i in range(args.output_size-1)]
        l.insert(0,args.rho)
        self.weights = Variable(torch.Tensor(l).unsqueeze(0),requires_grad=False)
        self.labels = [x+1 for x in range(args.num_templates)]
        self.header = functools.reduce(lambda x,y: x+y, [[y+str(x) for y in ['p','r','f','s']] for x in self.labels])
        self.header.extend(['mip', 'mir', 'mif', 'acc'])
        if settings.cuda:
            self.criterion.cuda()
            self.weights = self.weights.cuda()
        
    #var is a list of variables returned by dataloader.
    #Convention- var[0] - data
    #var[-1] - index
    #var[-2] - y
    #returns - loss, ypred and ground truth 
    def __call__(self,var,model,mode):
        # Pdb.set_trace()
        args = self.args
        x = var[0]
        y = var[1].squeeze().long()
        template_score = model(x)
        #batch_size x (num_templates + 1)
        max_score,ypred = torch.max(template_score,dim=1)
        # loss_tensor  = (y == 1).float()*(arg_max_score == template_score.size(1)-1).float()*max_score*args.rho + (y == 1).float()*(arg_max_score != template_score.size(1)-1).float()*max_score*args.mil_reward + (y == 0).float()*(arg_max_score != template_score.size(1)-1).float()*args.neg_reward + (y == 0).float()*(arg_max_score == template_score.size(1)-1).float()*args.pos_reward
        #reward_tensor  = (y == 1).float()*(arg_max_score == ((template_score.size(1)-1))).float()*max_score*args.rho + (y == 1).float()*(arg_max_score != (template_score.size(1)-1)).float()*max_score*args.mil_reward + (y == 0).float()*(arg_max_score != (template_score.size(1)-1)).float()*args.neg_reward + (y == 0).float()*(arg_max_score == (template_score.size(1)-1)).float()*args.pos_reward
        #template_score[:,-1] = template_score[:,-1]*args.rho
        #rho = Variable(, )
        #template_score = template_score*rho
        loss = 0 
        if mode != 'train':
            template_score = template_score*self.weights
            reward_tensor = args.class_imbalance*y.float()*torch.max(template_score,dim=1)[0] + (1.0-y.float())*( (template_score[:,0]*args.pos_reward/args.rho) +  torch.max(template_score[:,1:],dim=1)[0]*args.neg_reward/args.mil_reward)
            loss = -1.0*reward_tensor.mean()
        
            # print((template_score.size(1)-1))
            # loss = self.criterion(template_score,y)
            # print(loss_tensor)
            # print(loss)
            # print("Loss is ------> ",loss)
            # exit(0)
        
        return loss, ypred, y
        
    
    def calculate_accuracies(self,ycpu, ypred_cpu):
        p,r,f,s = metrics.precision_recall_fscore_support(ycpu,ypred_cpu)
        micro_p = metrics.precision_score(ycpu,ypred_cpu, labels = self.labels, average='micro')
        micro_r = metrics.recall_score(ycpu,ypred_cpu, labels = self.labels, average='micro')
        micro_f = metrics.f1_score(ycpu,ypred_cpu, labels = self.labels, average='micro')
        accuracy = metrics.accuracy_score(ycpu,ypred_cpu)
        metric = functools.reduce(lambda x,y: x+y, map(list,zip(p,r,f,s)))
        metric.extend([accuracy, micro_p, micro_r, micro_f])
        return metric
    


def compute(epoch, model, loader, optimizer, mode, fh, tolog, eval_fn, args):
    
    t1 = time.time()
    
    if mode == 'train':
        model.train()
    else:
        model.eval()

    
    last_print = 0
    count = 0
    cum_loss = 0
    cum_count = 0
    
    if mode == 'train':
        optimizer.zero_grad()
        
    predictions = np.zeros(len(loader.dataset))
    gt = np.zeros(len(loader.dataset))
    
    #var is a list of - sentence, ner , pos and idx
    for var in loader:
        # print(var)
        # print(type(var))
        #Pdb().set_trace()
        var = list(var)
        idx = var[-1]
        count += len(idx)
        # print('IDX: ',len(idx))
        volatile = True
        if mode == 'train':
            volatile = False

        for index in range(len(var)-1):
            var[index] = Variable(var[index], volatile = volatile)
            if settings.cuda:
                var[index] = var[index].cuda()

        loss, arg_max_score, y   = eval_fn(var, model, mode)
        predictions[idx] = arg_max_score.data.cpu().numpy()
        gt[idx] = y.squeeze().data.cpu().numpy()
        if mode == 'train':
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
        # pos_pred = pos_pred.data.cpu().numpy()
        cum_loss = cum_loss + loss.data[0]*y.size(0)
        # cum_count += count
        cum_count = count
        # print("cum_count is -->",cum_count)
        # print("count is -->",count)


        if (count - last_print) >= args.log_after:
            print("Counts--> ", Counter(arg_max_score.data.cpu().numpy())) 

            last_print = count 
            rec = [epoch, mode, 1.0 * cum_loss / cum_count, cum_count, len(loader.dataset), time.time() - t1]
            utils.log(','.join([str(round(x, 6)) if isinstance(
                x, float) else str(x) for x in rec]))

    rec = [epoch, mode, 1.0 * cum_loss / cum_count, cum_count, len(loader.dataset), time.time() - t1] 
    tlh = ','.join(['tolog'+str(ti) for ti in range(len(tolog))])
    
    metric_header = []
    if mode != 'train':
        metric = eval_fn.calculate_accuracies(gt,predictions)
        rec.extend(metric)
        metric_header = eval_fn.header 
    
    utils.log('epoch,mode,loss,count,count1,time,'+tlh+','+metric_header)
    utils.log(','.join([str(round(x, 6)) if isinstance(
        x, float) else str(x) for x in rec]), file=fh)

    #predictions[predictions < 5] = 1
    #predictions[predictions == 5] = 0
    
    #acc = metrics.classification_report(gt,predictions)
    #print("Acc ==>",acc)

    #fh1 = open('predictions.txt','w')
    #print('\n'.join(map(str,predictions)),file=fh1)
    #fh1.close()
    return (rec,-1)
    





