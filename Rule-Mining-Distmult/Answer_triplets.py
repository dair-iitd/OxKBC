#!/usr/bin/env python
# coding: utf-8

# In[11]:


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
import argparse

# In[12]:


data='fb15k'
DATA_DIR = "../../data/"+data
DUMP_FILE = "../dumps/"+data+"_distmult_dump_norm.pkl"
MODEL_TYPE = data
# mining_dir=data+"_low_thresh"
mining_dir=data+"_rule_mining_tmp"

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output_path',help='output file',
                    required=True, default=None)
parser.add_argument('-i', '--triplet_file', help = 'input file containing all triplets for which explanation has to be generated', 
                    required=True, default=None)

args = parser.parse_args()


triplet_file = args.triplet_file
path = os.path.join(mining_dir, args.output_path)


#triplet_file='../data/fb15k/turk_test/hits10_not_hits1.txt'
#path=os.path.join(mining_dir,"explanation_rule_mining_turk_hits10_not_hits1.pkl")


os.system("mkdir -p "+mining_dir)


# In[13]:


train_data = utils.read_data(os.path.join(DATA_DIR,"train.txt"))
# dev_data = read_data(os.path.join(DATA_DIR,"valid.txt"))
# test_data = read_data(os.path.join(DATA_DIR,"test.txt"))


# In[14]:


dump=utils.load_pickle(DUMP_FILE)
dump.keys()


# In[15]:


model=models.TypedDM(DUMP_FILE)


# In[16]:


mapped_train_data = utils.map_data(train_data,dump)
# mapped_dev_data = map_data(dev_data)
# mapped_test_data = map_data(test_data)


# In[17]:


entity_to_rel=utils.get_ent_to_rel(mapped_train_data)


# In[18]:


rules_1_path=os.path.join(mining_dir,"1_sup=1_conf=0.pkl")
rules_2_path=os.path.join(mining_dir,"2_sup=1_conf=0.pkl")
rules_3_path=os.path.join(mining_dir,"3_sup=4_conf=0.pkl")
rules_1=utils.load_pickle(rules_1_path)
rules_2=utils.load_pickle(rules_2_path)
rules_3=utils.load_pickle(rules_3_path)
print(len(rules_1),len(rules_2),len(rules_3))


# In[ ]:





# In[19]:


def add_relation_body(rules,relation_to_body):
    for rule in rules:
        if rule[1] not in relation_to_body:
            relation_to_body[rule[1]]=[]
        relation_to_body[rule[1]].append(rule[0])
    return relation_to_body


# In[20]:


# relation_to_body={}
# relation_to_body=add_relation_body(rules_1,relation_to_body)
# relation_to_body=add_relation_body(rules_2,relation_to_body)
# relation_to_body=add_relation_body(rules_3,relation_to_body)
# print(len(relation_to_body))


# In[21]:


# rules=rules_1+rules_2+rules_3
# rules.sort(key=lambda x:(x[3]*1.0)/x[2],reverse=True)
# print(len(rules))

_ = [rules.sort(key=lambda x:(x[3]*1.0)/x[2],reverse=True) for rules in [rules_1, rules_2, rules_3]]
rules=rules_1+rules_2+rules_3
print(len(rules))


# In[22]:


relation_to_body={}
relation_to_body=add_relation_body(rules,relation_to_body)
print(len(relation_to_body))


# # Length 1 Rules

# In[23]:


dict_1=utils.get_r_e1e2_dict(mapped_train_data)


# # Length 2 Rules

# In[24]:


index_head=utils.get_head_index(mapped_train_data)
dict_2=utils.get_r1r2_e1e2_dict(mapped_train_data,index_head)


# ## Length 3 Rules

# In[25]:


## Gives entity in path for given relation and body


# In[26]:


re2_e1=utils.get_re2_e1_dict(mapped_train_data)
e1r_e2=utils.get_e1r_e2_dict(mapped_train_data)


# In[27]:


set_mapped_train_data=utils.get_set_mapped_train_data(mapped_train_data)


# In[28]:


def solve_3(fact,body,e1r_e2,re2_e1,set_mapped_train_data):
    r1=body[0]
    r2=body[1]
    r3=body[2]
    e1=fact[0]
    e4=fact[2]
    
    key1=(e1,r1)
    key2=(r3,e4)
    
    if key1 not in e1r_e2:
        return ("",-1)
    if key2 not in re2_e1:
        return ("",-1)
    list1=e1r_e2[key1]
    list2=re2_e1[key2]
    
    for e2 in list1:
        for e3 in list2:
            if (e2,r2,e3) in set_mapped_train_data:
                return (body,(e2,e3))
    
    return ("",-1)
    


# In[29]:


def get_explanation(fact,relation_to_body,dict_1,dict_2,e1r_e2,re2_e1,set_mapped_train_data):
    pair=(fact[0],fact[2])
    rel=fact[1]
    
    r1=0
    r2=0
    r3=0
#     print(r1,r2,r3)
    if rel not in relation_to_body:
        return ("",-1)
#     print("Exists")
    bodies=relation_to_body[rel]
    for body in bodies:
        if isinstance(body,int):
            r1+=1
            if pair not in dict_1[body]:
                continue
            return (body,-1)
        else:
            if len(body)==2:
                r2+=1
                if pair not in dict_2[body]:
                    continue
                return (body,dict_2[body][pair])
            else:
                r3+=1
                temp=solve_3(fact,body,e1r_e2,re2_e1,set_mapped_train_data)
                if(temp[0]!=""):
                    return temp
    
    return ("",-1)


# In[30]:


# get_explanation((453,37,82),relation_to_body,dict_1,dict_2)


# In[31]:


#triplet_file="/home/cse/btech/cs1150210/scratch/BTP/Interpretable-KBC/logs/fb15k/turk_test_hits10_not_hits1/small_id.txt"
#triplet_file="/home/yatin/hpcscratch/Aman_BTP/Interpretable-KBC-tlp/data/fb15k/test/test_hits_1_ordered_x.txt"
data=utils.read_data(triplet_file)
np_arr=np.array(utils.map_data(data,dump=dump)).astype(np.int32)
# np_arr=np.loadtxt(triplet_file)


# In[32]:


arr=[]
coun=0
for line in np_arr:
    fact=(int(line[0]),int(line[1]),int(line[2]))
    arr.append(get_explanation(fact,relation_to_body,dict_1,dict_2,e1r_e2,re2_e1,set_mapped_train_data))
    coun+=1


# In[33]:


# print(arr)
count=0
for x in arr:
    if x[0]!="":
        count+=1
print(count)


# In[34]:


print(arr[0:100])


# In[39]:


# path=os.path.join(mining_dir,"explanation_test_hits10_not_hits1.pkl")
utils.dump_pickle(arr,path)


# In[40]:


print(len(np_arr),len(arr))


# In[41]:


lol=utils.load_pickle(path)


# In[42]:


print(len(lol))
print(lol[0:100])


# In[ ]:




