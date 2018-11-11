import numpy as np
import pickle
import json
import os
import sys
import copy
import sklearn.preprocessing

MODEL_TYPE='yago_inv'

def normalize(X):
    return sklearn.preprocessing.normalize(X)

dump={}
with open(MODEL_TYPE+'_distmult_dump.pkl',"rb") as f:
    dump=pickle.load(f)

def normalize_dump_distmult(dump):
    dump_norm = {}
    for key in dump:
        if(key == 'relation_to_id' or key=='entity_to_id'):
            dump_norm[key] = dump[key]
        else:
            dump_norm[key] = normalize(dump[key])
    with open(MODEL_TYPE+"_distmult_dump_norm.pkl","wb") as f:
        pickle.dump(dump_norm,f)

normalize_dump_distmult(dump)
