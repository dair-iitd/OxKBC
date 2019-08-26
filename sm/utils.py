import argparse
import logging
import os
import pickle
import shutil

import settings
import torch
from IPython.core.debugger import Pdb
LOG_FILE = 'log.txt'

_LOG_LEVEL_STRINGS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
EPSILON = 0.0000001



def get_template_id_maps(num_templates, exclude_t_ids):
    old_to_new = [0]*(num_templates+1)
    new_to_old = [0]*(num_templates+1 - len(exclude_t_ids))
    cnt  = 0
    for i in range(num_templates+1):
        if i in exclude_t_ids:
            old_to_new[i] = 0
        else:
            new_to_old[cnt] = i
            old_to_new[i] = cnt
            cnt += 1
    
    #Pdb().set_trace()
    return old_to_new, new_to_old



def _log_level_string_to_int(log_level_string):
    if not log_level_string in _LOG_LEVEL_STRINGS:
        message = 'invalid choice: {0} (choose from {1})'.format(
            log_level_string, _LOG_LEVEL_STRINGS)
        raise argparse.ArgumentTypeError(message)

    log_level_int = getattr(logging, log_level_string, logging.INFO)
    # check the logging log_level_choices have not changed from our expected values
    assert isinstance(log_level_int, int)
    return log_level_int


def get_embed_size(base_model_file, use_ids):
    if not use_ids:
        return 0
    if os.path.exists(base_model_file):
        d = pickle.load(open(base_model_file, 'rb'))
        ent_embed = d['entity_real'].shape[1] + d['entity_type'].shape[1]
        rel_embed = d['head_rel_type'].shape[1] + \
            d['tail_rel_type'].shape[1] + d['rel_real'].shape[1]
        return int(2*ent_embed+rel_embed)
    else:
        logging.error(
            'Base Model file not present at {}'.format(base_model_file))
        raise Exception('Base Model file not present')


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(state, epoch, isBest, checkpoint_file, best_file):
    torch.save(state, checkpoint_file)
    if isBest:
        best_file = best_file + str(0)
        shutil.copyfile(checkpoint_file, best_file)

    logging.info('Saving checkpoint to {}'.format(checkpoint_file))


def log_sum_exp(x, dim=-1):
    max_score, _ = torch.max(x, dim)
    max_score_broadcast = max_score.unsqueeze(dim).expand_as(x)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast), dim))


class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            # If arg is a dict,add all the elements of that dict to self
            if isinstance(arg, dict):
                for k, v in arg.items():  # Python2 - for k, v in arg.iteritems():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
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
