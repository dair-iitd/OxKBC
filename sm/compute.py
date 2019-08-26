import yaml
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
from IPython.core.debugger import Pdb


def get_evaluation_function(args):
    return Loss(args)





class Loss(object):
    def __init__(self, args):
        self.args = args

        l = [args.mil_reward for i in range(args.num_templates - len(args.exclude_t_ids))]
        l.insert(0, args.rho)
        #self.weights = Variable(torch.Tensor(
        #    l).unsqueeze(0), requires_grad=False)
        self.weights =  Variable(torch.Tensor(l), requires_grad=False)
        self.labels = [x+1 for x in range(args.num_templates - len(args.exclude_t_ids))]
        self.header = functools.reduce(
            lambda x, y: x+y, [[y+str(args.n2o[x]) for y in ['p', 'r', 'f', 's']] for x in self.labels])
        self.header.extend(['acc','mip', 'mir', 'mif'])
        
        if self.args.kldiv_lambda != 0:
            #Pdb().set_trace()
            odist = yaml.load(open(self.args.label_distribution_file))[0]
            fdist = []
            for i,d in enumerate(odist):
                if i not in args.exclude_t_ids:
                    fdist.append(d)

            self.target_distribution = torch.Tensor(fdist).float()

            self.target_distribution = self.target_distribution/self.target_distribution.sum() 
            self.target_distribution = Variable(self.target_distribution)
        if settings.cuda:
            self.weights = self.weights.cuda()
            if self.args.kldiv_lambda != 0:
                self.target_distribution = self.target_distribution.cuda()
    
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
        kldiv_loss = Variable(torch.Tensor([0]))
        if 'train' in mode:
            if mode == 'train_un':
                #convert template score into probabilites
                template_score = F.softmax(template_score, dim=1)
                #template_score = template_score * self.weights
                reward_tensor = (self.args.class_imbalance*y.float()*(template_score[:, 0]*self.weights[0]+torch.max(template_score[:, 1:], dim=1)[0])) + (1.0-y.float())*((template_score[:, 0]*self.args.pos_reward) + torch.sum(template_score[:, 1:], dim=1)*self.args.neg_reward)
                loss = -1.0*reward_tensor.mean()
                if self.args.kldiv_lambda !=  0:
                    #if y.float().sum().data[0] != 0:
                    if y.float().sum().item() != 0:
                        class_probs = torch.log((template_score*(y.float().unsqueeze(-1).expand_as(template_score))).sum(dim=0)/y.float().sum())
                        kldiv_loss = F.kl_div(class_probs, self.target_distribution)
                        loss += self.args.kldiv_lambda * kldiv_loss

            elif mode == 'train_sup':
                #template_score = F.softmax(template_score, dim=1)
                #template_score = template_score * self.weights
                if len(y.shape) > 1 and y.shape[1] > 1:
                    loss = F.binary_cross_entropy_with_logits(template_score, y.float())
                else:
                    loss = F.cross_entropy(template_score,y)
            else:
                raise 
                 

        return loss, kldiv_loss, ypred, y

    def calculate_accuracies(self, ycpu, ypred_cpu):
        if len(ycpu.shape) > 1 and ycpu.shape[1] > 1:
            #Pdb().set_trace()
            logging.info("Multi task evaluation")
            ypred_cpu = np.ravel(ypred_cpu)
            correct_count = ycpu[np.arange(ycpu.shape[0]),ypred_cpu.astype(int)].sum()
            #return [correct_count*1.0/len(ypred_cpu)]
            total_acc = correct_count*1.0/len(ypred_cpu)

            acc = ycpu[np.arange(ycpu.shape[0]),ypred_cpu.astype(int)]
            ind_color = ycpu[:,1:].sum(axis=1) > 0
            correct_color = acc[ind_color].sum()
            recall = 1.0*correct_color/ind_color.sum()
            pred_color = (ypred_cpu > 0).sum()
            precision = 1.0*correct_color/pred_color
            if recall + precision == 0:
                f = 0
            else:
                f = (2*recall*precision)/(recall + precision)

            return [precision, recall, total_acc, f]

            
            #ygt = ycpu[:, 1:].sum(axis=1)
            #ygt = (ygt > 0).astype(int)
            #ypred = ycpu[np.arange(ycpu.shape[0]),ypred_cpu.astype(int)].astype(int)
            #p,r,f,s = metrics.precision_recall_fscore_support(ygt, ypred,labels=[1])
            #return [correct_count*1.0/len(ypred_cpu),p[0],r[0],s[0],correct_count*1.0/len(ypred_cpu), f[0]]

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
        
        cf = metrics.confusion_matrix(ycpu,ypred_cpu,labels=range(self.args.num_templates+1- len(self.args.exclude_t_ids)))
        logging.info('Confusion Matrix:\n {}'.format(cf))
        logging.info('Classification Report\n'+metrics.classification_report(ycpu,ypred_cpu,labels=range(self.args.num_templates+1-len(self.args.exclude_t_ids))))
        #
        return metric


def compute(epoch, model, loader, optimizer, mode, eval_fn, args, labelled_train_loader=None,calc_acc=True):

    start_time = time.time()
    last_print = 0
    count = 0
    cum_loss = 0
    cum_kl_loss = 0
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
    if loader.dataset.Y is not None:
        ground_truth = np.zeros(loader.dataset.Y.shape)
    else:
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

        loss, kl_div_loss, ypred, ytrue = eval_fn(var, model, mode)
        predictions[idx] = ypred.data.cpu().numpy()
        ground_truth[idx] = ytrue.squeeze().data.cpu().numpy()

        #cum_loss = cum_loss + loss.data[0]*ytrue.size(0)
        #cum_kl_loss = cum_kl_loss + kl_div_loss.data[0]*ytrue.size(0)
        
        cum_loss = cum_loss + loss.data.item()*ytrue.size(0)
        cum_kl_loss = cum_kl_loss + kl_div_loss.data.item()*ytrue.size(0)
        cum_count += count

        if 'train' in mode:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (cum_count - last_print) >= args.log_after:
            str_ct = str(Counter(ypred.data.cpu().numpy()))
            last_print = cum_count
            rec = [epoch, mode, cum_loss/cum_count, cum_count,
                   len(loader.dataset), time.time() - start_time, 'Counts --> '+str_ct, cum_kl_loss/cum_count]
            logging.info(
                ','.join([str(round(x, 6)) if isinstance(x, float) else str(x) for x in rec]))

        #run sup training here
        if labelled_train_loader is not None:
            _ = compute(
                epoch, model, labelled_train_loader, optimizer, 'train_sup', eval_fn=eval_fn, args=args,calc_acc = False)


    rec = [epoch, mode, cum_loss/cum_count, cum_count,
           len(loader.dataset), time.time() - start_time, cum_kl_loss/cum_count]

    metric_header = []
    #Pdb().set_trace()
    if args.pred_file is not None:
        np.savetxt(os.path.join(args.output_path,args.pred_file), np.array(args.n2o)[predictions.astype(int)])
        logging.info("Written Predictions to {}".format(os.path.join(args.output_path,args.pred_file)))
    if mode != 'train_un' and calc_acc:
        metric = eval_fn.calculate_accuracies(ground_truth, predictions)
        rec.extend(metric)
        metric_header = eval_fn.header

    header = 'epoch,mode,loss,count,dataset_size,time,kl_loss'.split(',') + metric_header 
    if calc_acc:
        logging.info('epoch,mode,loss,count,dataset_size,time,kl_loss' +
                 ','.join(metric_header))
        logging.info(
        ','.join([str(round(x, 6)) if isinstance(x, float) else str(x) for x in rec]))

        print('epoch,mode,loss,count,dataset_size,time,kl_loss' +
                     ','.join(metric_header), file = args.lpf,flush=True)

        print(','.join([str(round(x, 6)) if isinstance(x, float) else str(x) for x in rec]),
                file=args.lpf,flush=True)
            
    if mode == 'train_un':
        return (rec, 2, header)
    else:
        # Presently optimizing micro_f score
        return (rec, -1, header)
