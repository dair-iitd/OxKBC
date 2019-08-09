
# coding: utf-8

# In[114]:


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


# In[115]:


data='fb15k' #INPUT
data_root="../../data/" #INPUT
DATA_DIR = os.path.join(data_root,data)
DUMP_FILE = "../dumps/"+data+"_distmult_dump_norm.pkl" #INPUT
MODEL_TYPE = data 
mining_dir=data+"_low_thresh" #INPUT
os.system("mkdir -p "+mining_dir)
rules_1="1_sup=1_conf=0.pkl" #INPUT 
rules_2="2_sup=1_conf=0.pkl" #INPUT
rules_3="3_sup=4_conf=0.pkl" #INPUT
triplet_file="/home/cse/btech/cs1150210/scratch/BTP/Interpretable-KBC/logs/fb15k/turk_test_hits10_not_hits1/small_id.txt" #INPUT
mined_answers="explanation_test_hits10_not_hits1.pkl" #INPUT

# In[116]:


train_data = utils.read_data(os.path.join(DATA_DIR,"train.txt"))
# dev_data = read_data(os.path.join(DATA_DIR,"valid.txt"))
# test_data = read_data(os.path.join(DATA_DIR,"test.txt"))


# In[117]:


dump=utils.load_pickle(DUMP_FILE)
dump.keys()


# In[118]:


model=models.TypedDM(DUMP_FILE)


# In[120]:


mapped_train_data = utils.map_data(train_data,dump)
# mapped_dev_data = map_data(dev_data)
# mapped_test_data = map_data(test_data)


# In[121]:


entity_to_rel=utils.get_ent_to_rel(mapped_train_data)


# In[122]:


rules_1_path=os.path.join(mining_dir,rules_1)
rules_2_path=os.path.join(mining_dir,rules_2)
rules_3_path=os.path.join(mining_dir,rules_3)
rules_1=utils.load_pickle(rules_1_path)
rules_2=utils.load_pickle(rules_2_path)
rules_3=utils.load_pickle(rules_3_path)
print(len(rules_1),len(rules_2),len(rules_3))


# In[123]:


def add_relation_body(rules,relation_to_body):
    for rule in rules:
        if rule[1] not in relation_to_body:
            relation_to_body[rule[1]]=[]
        relation_to_body[rule[1]].append(rule[0])
    return relation_to_body


# In[ ]:

'''
relation_to_body={}
relation_to_body=add_relation_body(rules_1,relation_to_body)
relation_to_body=add_relation_body(rules_2,relation_to_body)
relation_to_body=add_relation_body(rules_3,relation_to_body)
print(len(relation_to_body))
'''

# In[124]:

## USE THIS
rules=rules_1+rules_2+rules_3
rules.sort(key=lambda x:(x[3]*1.0)/x[2],reverse=True)
print(len(rules))
relation_to_body={}
relation_to_body=add_relation_body(rules,relation_to_body)
print(len(relation_to_body))


# # Length 1 Rules

# In[126]:


dict_1=utils.get_r_e1e2_dict(mapped_train_data)


# # Length 2 Rules

# In[127]:


index_head=utils.get_head_index(mapped_train_data)
dict_2=utils.get_r1r2_e1e2_dict(mapped_train_data,index_head)


# ## Length 3 Rules

# In[128]:


## Gives entity in path for given relation and body


# In[129]:


re2_e1=utils.get_re2_e1_dict(mapped_train_data)
e1r_e2=utils.get_e1r_e2_dict(mapped_train_data)


# In[130]:


set_mapped_train_data=utils.get_set_mapped_train_data(mapped_train_data)


# In[131]:


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
    


# In[132]:


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


# In[133]:


# get_explanation((453,37,82),relation_to_body,dict_1,dict_2)


# In[147]:


data=utils.read_data(triplet_file)
np_arr=np.array(utils.map_data(data,dump=dump)).astype(np.int32)
# np_arr=np.loadtxt(triplet_file)


# In[148]:


arr=[]
coun=0
for line in np_arr:
    fact=(int(line[0]),int(line[1]),int(line[2]))
    arr.append(get_explanation(fact,relation_to_body,dict_1,dict_2,e1r_e2,re2_e1,set_mapped_train_data))
    coun+=1


# In[149]:


# print(arr)
count=0
for x in arr:
    if x[0]!="":
        count+=1
print(count)


# In[150]:


print(arr[0:100])


# In[151]:


path=os.path.join(mining_dir,mined_answers)
utils.dump_pickle(arr,path)


# In[152]:


print(len(np_arr),len(arr))


# In[34]:


lol=utils.load_pickle(path)


# In[35]:


print(len(lol))
print(lol[0:100])

