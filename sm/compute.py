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


cuda = settings.cuda


def get_evaluation_function(args):
    return Loss(args) 

class Loss(object):
    def __init__(self,args):
        self.args = args
        
    #var is a list of variables returned by dataloader.
    #Convention- var[0] - data
    #var[-1] - index
    #var[-2] - y
    #returns - loss, ypred and ground truth 
    def __call__(self,var,model,mode):
        # Pdb.set_trace()
        args = self.args
        x = var[0]
        y = var[1].squeeze()
        template_score = model(x)
        #batch_size x (num_templates + 1)
        max_score,arg_max_score = torch.max(template_score,dim=1)
        # loss_tensor  = (y == 1).float()*(arg_max_score == template_score.size(1)-1).float()*max_score*args.rho + (y == 1).float()*(arg_max_score != template_score.size(1)-1).float()*max_score*args.mil_reward + (y == 0).float()*(arg_max_score != template_score.size(1)-1).float()*args.neg_reward + (y == 0).float()*(arg_max_score == template_score.size(1)-1).float()*args.pos_reward
        loss_tensor  = (y == 1).float()*(arg_max_score == ((template_score.size(1)-1))).float()*max_score*args.rho + (y == 1).float()*(arg_max_score != (template_score.size(1)-1)).float()*max_score*args.mil_reward + (y == 0).float()*(arg_max_score != (template_score.size(1)-1)).float()*args.neg_reward + (y == 0).float()*(arg_max_score == (template_score.size(1)-1)).float()*args.pos_reward

        # print((template_score.size(1)-1))

        loss = -1*loss_tensor.mean()
        # print(loss_tensor)
        # print(loss)
        # print("Loss is ------> ",loss)
        # exit(0)
        if(math.isnan(loss)):
            print(loss_tensor)
            exit(0)
        return loss, arg_max_score, y  


def compute(epoch, model, loader, optimizer, mode, fh, tolog, eval_fn, args):
    global cuda
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
        

    #var is a list of - sentence, ner , pos and idx
    for var in loader:
        # print(var)
        # print(type(var))
        #Pdb().set_trace()
        var = list(var)
        idx = var[-1]
        count += len(idx)
        volatile = True
        if mode == 'train':
            volatile = False

        for index in range(len(var)-1):
            var[index] = Variable(var[index], volatile = volatile)
            if cuda:
                var[index] = var[index].cuda()

        loss, arg_max_score, y   = eval_fn(var, model, mode)

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
            print("Counts--> ", Counter(arg_max_score.data.numpy())) 
            last_print = count 
            rec = [epoch, mode, 1.0 * cum_loss / cum_count, cum_count, len(loader.dataset), time.time() - t1]
            utils.log(','.join([str(round(x, 6)) if isinstance(
                x, float) else str(x) for x in rec]))

    rec = [epoch, mode, 1.0 * cum_loss / cum_count, cum_count, len(loader.dataset), time.time() - t1] 
    tlh = ','.join(['tolog'+str(ti) for ti in range(len(tolog))])
    utils.log('epoch,mode,loss,count,count1,time,'+tlh)
    utils.log(','.join([str(round(x, 6)) if isinstance(
        x, float) else str(x) for x in rec]), file=fh)
    return (rec,2)
    





