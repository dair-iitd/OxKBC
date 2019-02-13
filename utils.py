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

def read_entity_names(path,add_wiki=True):
    WIKI_PREFIX_URL = 'https://en.wikipedia.org/wiki/'
    entity_names = {}
    with open(path, "r",errors='ignore',encoding='ascii') as f:
        lines = f.readlines()
        for line in lines:
            content_raw = line.split('\t')
            content = [el.strip() for el in content_raw]
            if content[0] in entity_names:
                logging.warn('Duplicate Entity found %s in line %s' % (content[0],' '.join(line)))
            name = content[1]
            wiki_id = content[2]
            wiki_link = WIKI_PREFIX_URL+wiki_id
            if(add_wiki):
                entity_names[content[0]] = '<a target=\"_blank\" href=\"'+wiki_link+'\">'+name+'</a>'
            else:
                entity_names[content[0]] = name
    return entity_names

def read_relation_names(path):
    relation_names = {}
    with open(path, "r",errors='ignore',encoding='ascii') as f:
        lines = f.readlines()
        for line in lines:
            content = line.split()
            if content[0] in relation_names:
                logging.warn('Duplicate Entity found %s in line %s' % (content[0],' '.join(line)))
            relation_names[content[0]] = ' '.join(content[1:])
    return relation_names


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

def get_ent_ent_rel(data_arr):
    e1_e2_r={}
    e2_e1_r={}
    for data in data_arr:
        e1=data[0]
        r=data[1]
        e2=data[2]
        
        if e1 not in e1_e2_r:
            e1_e2_r[e1]={}
        if e2 not in e2_e1_r:
            e2_e1_r[e2]={}
        
        if e2 not in e1_e2_r[e1]:
            e1_e2_r[e1][e2]=[]
        if e1 not in e2_e1_r[e2]:
            e2_e1_r[e2][e1]=[]
        
        e1_e2_r[e1][e2].append(r)
        e2_e1_r[e2][e1].append(r)
    return e1_e2_r,e2_e1_r

def get_most_freq_ind(data_arr):
    r_e2_e1={}
    e1_e2_r={}

    for data in data_arr:
        e1=data[0]
        r=data[1]
        e2=data[2]
        
        if e1 not in e1_e2_r:
            e1_e2_r[e1]={}
        if r not in r_e2_e1:
            r_e2_e1[r]={}
        
        if e2 not in e1_e2_r[e1]:
            e1_e2_r[e1][e2]=[]
        if e2 not in r_e2_e1[r]:
            r_e2_e1[r][e2]=[]
        
        e1_e2_r[e1][e2].append(r)
        r_e2_e1[r][e2].append(e1)
    return r_e2_e1,e1_e2_r



def hard_simi(lis1,lis2):
    s1=set(lis1)
    s2=set(lis2)
    temp=list(s1&s2)
    temp=[(x,x) for x in temp]
    return temp

def soft_simi(lis1,lis2,model,thresh=0.5):
    list_r1r2=[]
    if (len(lis1)>len(lis2)):
        lis1,lis2=lis2,lis1
    for r1 in lis1:
        best_score=-1
        best_rel=-1
        for r2 in lis2:
            score=model.get_relation_similarity(r1,r2)
            if(score>best_score):
                best_score=score
                best_rel=r2
        if(best_rel!=-1 and best_score>thresh):
            list_r1r2.append((r1,best_rel))
    
    return list_r1r2

def explain_similarity_aux(e1,e2,model,flip,look_up,hard_match=True):
    d1=look_up[e1]
    d2=look_up[e2]
    relevant_tuple=[]
    
    for e in d1:
        if e not in d2:
            continue
        lis1=d1[e]
        lis2=d2[e]
        relevant=[]
        if(hard_match):
            relevant=hard_simi(lis1,lis2)
        else:
            relevant=soft_simi(lis1,lis2,model)
    
        for r1r2 in relevant:
            if flip:
                relevant_tuple.append(([e,r1r2[0],e1],[e,r1r2[1],e2]))
            else:
                relevant_tuple.append(([e1,r1r2[0],e],[e2,r1r2[1],e]))
    
    return relevant_tuple       

def explain_similarity(e1,e2,model,hard_match):
    list1=explain_similarity_aux(e1,e2,model,False,e1_e2_r,hard_match)
    list2=explain_similarity_aux(e1,e2,model,True,e2_e1_r,hard_match)
    
    return list1,list2

def freq_for_relation(r,e2):
    ans=[]
    if r in r_e2_e1 and e2 in r_e2_e1[r]:
        ans=r_e2_e1[r][e2]
    return ans

def freq_for_entity(e1,e2):
    ans=[]
    if e1 in e1_e2_r and e2 in e1_e2_r[e1]:
        ans=e1_e2_r[e1][e2]
    return ans

def get_relation_frequent(fact,enum_to_id, rnum_to_id, eid_to_name, rid_to_name):
    string_frequent = "<div class=\"tooltip\">frequently seen <span class=\"tooltiptext\">"
    
    other_part = freq_for_relation(fact[1],fact[2])
    
    r_name = rid_to_name.get(rnum_to_id[fact[1]],rnum_to_id[fact[1]])   
    e2_name = eid_to_name.get(enum_to_id[fact[2]],enum_to_id[fact[2]])
        
    other_knowledge = []
    for el in other_part:
        other_knowledge.append(eid_to_name.get(enum_to_id[el],enum_to_id[el]))

    cs_string = get_cs_string(other_knowledge)
    string_frequent += "(" + cs_string  + " ) " + r_name + "  " + e2_name + "<br>"
    string_frequent += "</span></div>"

    return string_frequent


def get_entity_frequent(fact,enum_to_id, rnum_to_id, eid_to_name, rid_to_name):
    e1_name = eid_to_name.get(enum_to_id[fact[0]],enum_to_id[fact[0]])
    e2_name = eid_to_name.get(enum_to_id[fact[2]],enum_to_id[fact[2]])
    
    other_part = freq_for_entity(fact[0],fact[2])
    other_knowledge = []
    for el in other_part:
        other_knowledge.append(rid_to_name.get(rnum_to_id[el],rnum_to_id[el]))

    string_frequent = "<div class=\"tooltip\">frequently seen <span class=\"tooltiptext\">"
    cs_string = get_cs_string(other_knowledge)
    string_frequent += e1_name + " ( " + cs_string  + " ) " + e2_name + "<br>"
    string_frequent += "</span></div>"

    return string_frequent


def get_cs_string(l):
    if len(l) <= 2:
        return ' , '.join(l)
    else:
        n = len(l) - 2
        return '{} and {} more...'.format(' , '.join(l[:2]), n)

def get_why_similar(e1,e2,enum_to_id, rnum_to_id, eid_to_name, rid_to_name,base_model):
        tuples_similar_head,tuples_similar_tail = explain_similarity(e1,e2,base_model,hard_match=True)
        e1_name = eid_to_name.get(enum_to_id[e1],enum_to_id[e1])
        e2_name = eid_to_name.get(enum_to_id[e2],enum_to_id[e2])

        rel_dir_head = {}
        for t in tuples_similar_head:
            t1_mapped =  map_fact(t[0], enum_to_id, rnum_to_id)
            t1_mapped_name =  map_fact(t1_mapped, eid_to_name, rid_to_name)
            t2_mapped =  map_fact(t[1], enum_to_id, rnum_to_id)
            t2_mapped_name =  map_fact(t2_mapped, eid_to_name, rid_to_name)
            if t1_mapped_name[1] not in rel_dir_head:
                rel_dir_head[t1_mapped_name[1]] = []
            rel_dir_head[t1_mapped_name[1]].append(t1_mapped_name[2])
        
        rel_dir_tail = {}
        for t in tuples_similar_tail:
            t1_mapped =  map_fact(t[0], enum_to_id, rnum_to_id)
            t1_mapped_name =  map_fact(t1_mapped, eid_to_name, rid_to_name)
            t2_mapped =  map_fact(t[1], enum_to_id, rnum_to_id)
            t2_mapped_name =  map_fact(t2_mapped, eid_to_name, rid_to_name)
            if t1_mapped_name[1] not in rel_dir_tail:
                rel_dir_tail[t1_mapped_name[1]] = []
            rel_dir_tail[t1_mapped_name[1]].append(t1_mapped_name[0])

        string_similar = "<div class=\"tooltip\">similar <span class=\"tooltiptext\">"
        for rel in rel_dir_head:
            cs_string = get_cs_string(rel_dir_head[rel])
            string_similar += e1_name + " and " + e2_name + " " + rel + " (" + cs_string +" )<br>"
        string_similar += "<br>"

        for rel in rel_dir_tail:
            cs_string = get_cs_string(rel_dir_tail[rel])
            string_similar += "(" + cs_string  + " ) " + rel + "  " + e1_name + " and " + e2_name +"<br>"
        string_similar += "<br>"
        string_similar += "</span></div>"

        return string_similar

def heuristic_purge_relations(rel_to_id,rel_to_name):
    for rel_num in rel_to_id:
        rel = rel_to_id[rel_num]
        if rel in rel_to_name:
            continue
        parts=rel.split(".")
        name=""
        if (len(parts)==1):
            parts=parts[0].split("/")
            name=parts[0]+" "+parts[-1]
        else:
            name=parts[0].split("/")[-2]+" "+parts[1].split("/")[-1]
        rel_to_name[rel]=name


with open('css_style.css','r') as css:
    CSS_STYLE = css.read()
