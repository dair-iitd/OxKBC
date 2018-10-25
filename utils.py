import logging
import argparse
import math

_LOG_LEVEL_STRINGS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']

def _log_level_string_to_int(log_level_string):
    if not log_level_string in _LOG_LEVEL_STRINGS:
        message = 'invalid choice: {0} (choose from {1})'.format(log_level_string, _LOG_LEVEL_STRINGS)
        raise argparse.ArgumentTypeError(message)

    log_level_int = getattr(logging, log_level_string, logging.INFO)
    # check the logging log_level_choices have not changed from our expected values
    assert isinstance(log_level_int, int)
    return log_level_int

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


