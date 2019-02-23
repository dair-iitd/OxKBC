import logging
import argparse
import math
import pickle


import numpy as np
import pandas as pd


_LOG_LEVEL_STRINGS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
EPSILON=0.0000001
e1_e2_r=None
e2_e1_r=None
r_e2_e1=None

def _log_level_string_to_int(log_level_string):
    if not log_level_string in _LOG_LEVEL_STRINGS:
        message = 'invalid choice: {0} (choose from {1})'.format(log_level_string, _LOG_LEVEL_STRINGS)
        raise argparse.ArgumentTypeError(message)

    log_level_int = getattr(logging, log_level_string, logging.INFO)
    # check the logging log_level_choices have not changed from our expected values
    assert isinstance(log_level_int, int)
    return log_level_int

def read_data(path):
    data = []
    with open(path, "r",errors='ignore',encoding='ascii') as infile:
        lines = infile.readlines()
        data = [line.strip().split() for line in lines]
    return data

def read_pkl(filename):
    with open(filename, "rb") as f:
        pkl_dict = pickle.load(f)
    return pkl_dict

def convert_to_pandas(table, header):
    return pd.DataFrame(table, columns=header)

def sigmoid(x):
    return 1/(1+math.exp(-x))

def get_rank(array,val):
    count=0
    for elem in array:
        if(elem>val):
            count+=1
    return count

def delete_from_list(lis,val):
    new_lis1=[]
    new_lis2=[]
    for elem in lis:
        if(elem[0]==val[0]):
            continue
        if(elem[1]==val[1]):
            continue
        new_lis1.append(elem[0])
        new_lis2.append(elem[1])
    
    return (new_lis1,new_lis2)

def get_inverse_dict(mydict):
    inverse_dict = {}
    for k in mydict.keys():
        if mydict[k] in inverse_dict:
            raise "Cannot Construct inverse dictionary, as function not one-one"
        inverse_dict[mydict[k]] = k
    return inverse_dict

def map_fact(line, mapped_entity = None, mapped_relation=None):
    try:
        e1_mapped = mapped_entity.get(line[0],line[0]) if mapped_entity is not None else line[0]
        r_mapped = mapped_relation.get(line[1],line[1]) if mapped_relation is not None else line[1]
        e2_mapped = mapped_entity.get(line[2],line[2]) if mapped_entity is not None else line[2]
        return [e1_mapped,r_mapped,e2_mapped]
    except KeyError:
        logging.warn('Got Key Error for line %s' % (' '.join(line)))

def map_data(data, mapped_entity = None, mapped_relation=None):
    mapped_data = []
    for line in data:
        mapped_data.append(map_fact(line,mapped_entity,mapped_relation))
    return mapped_data

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


