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


def get_evaluation_function(args):
    return Loss(args)


class Loss(object):
    def __init__(self, args):
        self.args = args
        self.criterion = nn.NLLLoss(weight=torch.Tensor([1.0, 20.0]))

        l = [args.mil_reward for i in range(args.num_templates)]
        l.insert(0, args.rho)
        self.weights = Variable(torch.Tensor(
            l).unsqueeze(0), requires_grad=False)

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
        if mode == 'train':
            template_score = template_score * self.weights
            reward_tensor = self.args.class_imbalance*y.float()*torch.max(template_score, dim=1)[0] + (1.0-y.float())*(
                (template_score[:, 0]*self.args.pos_reward/self.args.rho) + torch.max(template_score[:, 1:], dim=1)[0]*self.args.neg_reward/self.args.mil_reward)
            loss = -1.0*reward_tensor.mean()

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
        return metric


def compute(epoch, model, loader, optimizer, mode, eval_fn, args):

    start_time = time.time()
    last_print = 0
    count = 0
    cum_loss = 0
    cum_count = 0

    if mode == 'train':
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

        volatile = False if mode == 'train' else True
        for index in range(len(var)-1):
            var[index] = Variable(var[index], volatile=volatile)
            if settings.cuda:
                var[index] = var[index].cuda()

        loss, ypred, ytrue = eval_fn(var, model, mode)
        predictions[idx] = ypred.data.cpu().numpy()
        ground_truth[idx] = ytrue.squeeze().data.cpu().numpy()

        cum_loss = cum_loss + loss.data[0]*ytrue.size(0)
        cum_count += count

        if mode == 'train':
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

    rec = [epoch, mode, cum_loss/cum_count, cum_count,
           len(loader.dataset), time.time() - start_time]

    metric_header = []
    if mode != 'train':
        np.savetxt(os.path.join(args.output_path,
                                'predictions_valid.txt'), predictions)
        metric = eval_fn.calculate_accuracies(ground_truth, predictions)
        rec.extend(metric)
        metric_header = eval_fn.header

    logging.info('epoch,mode,loss,count,dataset_size,time,' +
                 ','.join(metric_header))
    logging.info(
        ','.join([str(round(x, 6)) if isinstance(x, float) else str(x) for x in rec]))

    if mode == 'train':
        return (rec, 3)
    else:
        # Presently optimizing micro_f score
        return (rec, -1)
