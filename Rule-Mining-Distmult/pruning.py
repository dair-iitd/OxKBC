
# coding: utf-8

# In[2]:


import numpy as np
import pickle
import json
import os
import sys
import copy
import sklearn.preprocessing
import models
from sklearn.neighbors import NearestNeighbors
import utils


# In[18]:


data='fb15k' #INPUT
data_root="../../data/" #INPUT
DATA_DIR = os.path.join(data_root,data)
DUMP_FILE = "../dumps/"+data+"_distmult_dump_norm.pkl" #INPUT
MODEL_TYPE = data 
mining_dir=data+"_low_thresh" #INPUT
os.system("mkdir -p "+mining_dir)

path1="./1_sup=1.pkl" #INPUT
path2="./2_sup=1.pkl" #INPUT
path1=os.path.join(mining_dir,path1)
path2=os.path.join(mining_dir,path2)

conf_1=1 #INPUT
conf_2=1 #INPUT
pruned_1="1_sup=1_conf=1.pkl" #INPUT
pruned_2="2_sup=1_conf=1.pkl" #INPUT


# In[19]:


train_data = utils.read_data(os.path.join(DATA_DIR,"train.txt"))
dump=utils.load_pickle(DUMP_FILE)
print(dump.keys())


# In[20]:


model=models.TypedDM(DUMP_FILE)
mapped_train_data = utils.map_data(train_data,dump)


# In[21]:


def prune_rules(rules_dict,set_len_body,thresh=0.1):
    new_rules=[]
    for body in rules_dict:
        denom=set_len_body[body]
        for r in rules_dict[body]:
            confidence=rules_dict[body][r]*1.0
#             confidence/=denom
            if(confidence>thresh):
                new_rules.append([body,r,denom,confidence])
    return new_rules


# In[ ]:


rules_dict_1=utils.load_pickle(path1)
rules_dict_2=utils.load_pickle(path2)


# ## Prune Length 1 Rules

# In[ ]:


count_r,set_r=utils.get_relation_dict(mapped_train_data)
set_len_r={}
for body in set_r:
    set_len_r[body]=len(set_r[body])


# In[ ]:


confidence=conf_1
pruned_rules_1=prune_rules(rules_dict_1,set_len_r,confidence)
pruned_rules_1=sorted(pruned_rules_1,reverse=True,key = lambda x: x[2])
print(len(pruned_rules_1))


# ## Prune Length 2 Rules

# In[ ]:


path="set_r1_r2.pkl"
path=os.path.join(mining_dir,path)
set_len_r1_r2=utils.load_pickle(path)


# In[ ]:


confidence=conf_2
pruned_rules_2=prune_rules(rules_dict_2,set_len_r1_r2,confidence)
pruned_rules_2=sorted(pruned_rules_2,reverse=True,key = lambda x: x[2])
print(len(pruned_rules_2))


# In[ ]:


pruned_path=pruned_1
pruned_path=os.path.join(mining_dir,pruned_path)
utils.dump_pickle(pruned_rules_1,pruned_path)


# In[14]:


pruned_path=pruned_2
pruned_path=os.path.join(mining_dir,pruned_path)
utils.dump_pickle(pruned_rules_2,pruned_path)

