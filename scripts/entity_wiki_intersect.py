#!/usr/bin/env python
# coding: utf-8

# ### This script was used to create a wikipedia linkfile for the entities in the FB15k and not the complete Freebase

# In[ ]:


import utils
import argparse
import logging
import os
import pickle
import string
import time
import numpy as np
import pandas as pd


# In[9]:


data_repo_root = "../data/fb15k/"
model_weights = "dumps/fb15k237_distmult_dump_norm.pkl"
wiki_file = os.path.join(data_repo_root, "mid2wikipedia.tsv")
orig_file = os.path.join(data_repo_root, "entity_mid_name_type_typeid.txt")
intersection_file = os.path.join(data_repo_root, "mid2wikipedia_cleaned.tsv")


# In[5]:


distmult_dump = utils.read_pkl(model_weights)


# In[12]:


def read_data(path):
    mapping_name = {}
    mapping_url = {}
    with open(path, "r") as f:
        for line in f:
            line_arr = line.split("\t")
            mapping_name[line_arr[0]] = line_arr[1]
            mapping_url[line_arr[0]] = line_arr[2]
    return mapping_name, mapping_url


# In[13]:


mapping_name, mapping_url = read_data(wiki_file)


# In[15]:


reader = open(orig_file, "r")
writer = open(intersection_file, "w")
for line in reader:
    line_arr = line.split("\t")
    string = ""
    if(line_arr[0] in mapping_name):
        string = "\t".join(
            [line_arr[0], mapping_name[line_arr[0]], mapping_url[line_arr[0]]])
    else:
        string = "\t".join([line_arr[0], line_arr[1], ""])
    print(string, file=writer)
reader.close()
writer.close()
