from __future__ import print_function
from IPython.core.debugger import Pdb
import torch
import shutil
from datetime import datetime as dt
#from . import settings
import math
import settings

CONSOLE_FILE = 'IPYTHON_CONSOLE'

def log(s, file=None):
    print('{},{}'.format(dt.now().strftime('%Y%m%d%H%M%S'), s), file=file)
    if file is not None:
        print('{},{}'.format(dt.now().strftime('%Y%m%d%H%M%S'), s), file=None)
    #
    print('{},{}'.format(dt.now().strftime('%Y%m%d%H%M%S'), s),
          file=open(CONSOLE_FILE, 'a'))
    if file is not None:
        file.flush()

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(state, epoch, isBest, checkpoint_file, best_file):
    torch.save(state, checkpoint_file)
    if isBest:
        print("isBest True. Epoch: {0}, bestError: {1}".format(
            state['epoch'], state['best_score']))
        best_file = best_file + str(0)
        shutil.copyfile(checkpoint_file,
                        best_file)

def log_sum_exp(x,dim = -1):
    max_score, _ = torch.max(x, dim)
    max_score_broadcast = max_score.unsqueeze(dim).expand_as(x)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast), dim))

def sigmoid(x):
        return 1/(1+math.exp(-x))

class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            # If arg is a dict,add all the elements of that dict to self
            # print(args)
            if isinstance(arg, dict):
                for k, v in arg.items():  # Python2 - for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

def get_rank(array,val):
    count=0

    for elem in array:
        if(elem>val):
            count+=1

    return count
