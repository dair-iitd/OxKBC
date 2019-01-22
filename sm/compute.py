import functools
import logging
import os
import time
from collections import Counter

import numpy as np
from sklearn import metrics

import settings
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F 

def get_evaluation_function(args):
    return Loss(args)


class Loss(object):
    def __init__(self, args):
        self.args = args
        self.criterion = nn.NLLLoss(weight=torch.Tensor([1.0, 20.0]))

        l = [args.mil_reward for i in range(args.num_templates)]
        l.insert(0, args.rho)
        #self.weights = Variable(torch.Tensor(
        #    l).unsqueeze(0), requires_grad=False)
        self.weights =  Variable(torch.Tensor(l), requires_grad=False)
        self.labels = [x+1 for x in range(args.num_templates)]
        self.header = functools.reduce(
            lambda x, y: x+y, [[y+str(x) for y in ['p', 'r', 'f', 's']] for x in self.labels])
        self.header.extend(['acc','mip', 'mir', 'mif'])

        if settings.cuda:
            self.criterion.cuda()
            self.weights = self.weights.cuda()

    def __call__(self, var, model, mode):
        '''
        var is a list of variables returned by dataloader.
        var = (data,y,index)
        returns - loss, ypred and ground truth
        '''
        x = var[0]
        y = var[1].squeeze().long()
        template_score = model(x)

        max_score, ypred = torch.max(template_score, dim=1)
        
        loss = Variable(torch.Tensor([0]))
        if 'train' in mode:
            if mode == 'train_un':
                #convert template score into probabilites
                template_score = F.softmax(template_score, dim=1)
                #template_score = template_score * self.weights
                reward_tensor = (self.args.class_imbalance*y.float()*(template_score[:, 0]*self.weights[0]+torch.max(template_score[:, 1:], dim=1)[0])) + (1.0-y.float())*((template_score[:, 0]*self.args.pos_reward) + torch.sum(template_score[:, 1:], dim=1)*self.args.neg_reward)
                loss = -1.0*reward_tensor.mean()
            elif mode == 'train_sup':
                #template_score = F.softmax(template_score, dim=1)
                #template_score = template_score * self.weights
                loss = F.cross_entropy(template_score,y)
            else:
                raise 
                 

        return loss, ypred, y

    def calculate_accuracies(self, ycpu, ypred_cpu):
        p, r, f, s = metrics.precision_recall_fscore_support(
            ycpu, ypred_cpu, labels=self.labels)
        micro_p = metrics.precision_score(
            ycpu, ypred_cpu, labels=self.labels, average='micro')
        micro_r = metrics.recall_score(
            ycpu, ypred_cpu, labels=self.labels, average='micro')
        micro_f = metrics.f1_score(
            ycpu, ypred_cpu, labels=self.labels, average='micro')
        accuracy = metrics.accuracy_score(ycpu, ypred_cpu)
        metric = functools.reduce(lambda x, y: x+y, map(list, zip(p, r, f, s)))
        metric.extend([accuracy, micro_p, micro_r, micro_f])

        str_ct = str(Counter(ypred_cpu))
        str_ct_y = str(Counter(ycpu))
        logging.info(' Predicted counts val: {}'.format(str_ct))
        logging.info(' True counts val: {}'.format(str_ct_y))
        
        cf = metrics.confusion_matrix(ycpu,ypred_cpu,labels=[0,1,2,3,4,5])
        logging.info('Confusion Matrix:\n {}'.format(cf))
        logging.info('Classification Report\n'+metrics.classification_report(ycpu,ypred_cpu,labels=[0,1,2,3,4,5]))

        return metric


def compute(epoch, model, loader, optimizer, mode, eval_fn, args, labelled_train_loader=None,calc_acc=True):

    start_time = time.time()
    last_print = 0
    count = 0
    cum_loss = 0
    cum_count = 0

    if 'train' in mode:
        if calc_acc:
            logging.info('Setting model mode to train')
        model.train()
        optimizer.zero_grad()
    else:
        logging.info('Setting model mode to eval')
        model.eval()

    predictions = np.zeros(len(loader.dataset))
    ground_truth = np.zeros(len(loader.dataset))

    # var is a list of - data, y and idx
    for var in loader:
        var = list(var)
        idx = var[-1]
        count = len(idx)

        volatile = False if 'train' in mode else True
        for index in range(len(var)-1):
            var[index] = Variable(var[index], volatile=volatile)
            if settings.cuda:
                var[index] = var[index].cuda()

        loss, ypred, ytrue = eval_fn(var, model, mode)
        predictions[idx] = ypred.data.cpu().numpy()
        ground_truth[idx] = ytrue.squeeze().data.cpu().numpy()

        cum_loss = cum_loss + loss.data[0]*ytrue.size(0)
        #cum_loss = cum_loss + loss.data.item()*ytrue.size(0)
        cum_count += count

        if 'train' in mode:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (cum_count - last_print) >= args.log_after:
            str_ct = str(Counter(ypred.data.cpu().numpy()))
            last_print = cum_count
            rec = [epoch, mode, cum_loss/cum_count, cum_count,
                   len(loader.dataset), time.time() - start_time, 'Counts --> '+str_ct]
            logging.info(
                ','.join([str(round(x, 6)) if isinstance(x, float) else str(x) for x in rec]))

        #run sup training here
        if labelled_train_loader is not None:
            _ = compute(
                epoch, model, labelled_train_loader, optimizer, 'train_sup', eval_fn=eval_fn, args=args,calc_acc = False)


    rec = [epoch, mode, cum_loss/cum_count, cum_count,
           len(loader.dataset), time.time() - start_time]

    metric_header = []
    if mode != 'train_un' and calc_acc:
        np.savetxt(os.path.join(args.output_path,
                                'predictions_valid.txt'), predictions)
        if(args.raw):
            loader.save_raw_data(os.path.join(args.output_path,'valid_raw_data.txt'),predictions)
        metric = eval_fn.calculate_accuracies(ground_truth, predictions)
        rec.extend(metric)
        metric_header = eval_fn.header

    header = 'epoch,mode,loss,count,dataset_size,time'.split(',') + metric_header 
    if calc_acc:
        logging.info('epoch,mode,loss,count,dataset_size,time,' +
                 ','.join(metric_header))
        logging.info(
        ','.join([str(round(x, 6)) if isinstance(x, float) else str(x) for x in rec]))

        print('epoch,mode,loss,count,dataset_size,time,' +
                     ','.join(metric_header), file = args.lpf)

        print(','.join([str(round(x, 6)) if isinstance(x, float) else str(x) for x in rec]),
                file=args.lpf)
            
    if mode == 'train_un':
        return (rec, 2, header)
    else:
        # Presently optimizing micro_f score
        return (rec, -1, header)
