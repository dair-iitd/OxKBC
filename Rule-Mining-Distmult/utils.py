import pickle

def read_data(path):
    data = []
    with open(path, "r") as infile:
        lines = infile.readlines()
        data = [line.strip().split() for line in lines]
    return data

def map_data(data,dump):
    mapped_data = []
    for el in data:
        l = [dump['entity_to_id'][el[0]], dump['relation_to_id'][el[1]], dump['entity_to_id'][el[2]]]
        mapped_data.append(l)
    return mapped_data

def map_triple(triple,dump):
    el = triple
    l = [dump['entity_to_id'][el[0]], dump['relation_to_id'][el[1]], dump['entity_to_id'][el[2]]]
    return l    

# def dump_new_complex(dump,MODEL_TYPE):
#     dump_norm = {}
#     for key in dump:
#         if(key == 'relation_to_id' or key=='entity_to_id'):
#             dump_norm[key] = copy.deepcopy(dump[key])
#         else:
#             dump_norm[key] = copy.deepcopy(normalize(dump[key]))
#     with open(MODEL_TYPE+"_dump_norm.pkl","wb") as f:
#         pickle.dump(dump_norm,f)


def get_head_index(data_arr):
    index_head = {}

    for data in data_arr:
        if data[0] not in index_head:
            index_head[data[0]] = []
        index_head[data[0]].append((data[1],data[2]))
    return index_head

def get_r1r2_count(data_arr,index_head,get_set=False):

    count_r1_r2={}
    set_r1_r2 = {}

    for data in data_arr:
        if(data[2] not in index_head):
            continue
        rel_tail_list = index_head[data[2]]
        for rel_tail in rel_tail_list:
            rel=rel_tail[0]
            tail=rel_tail[1]
            pair = (data[1], rel)
            if(pair not in count_r1_r2):
                count_r1_r2[pair]=0
                if(get_set):
                    set_r1_r2[pair] = set()
            count_r1_r2[pair]+=1
            if(get_set):
                set_r1_r2[pair].add((data[0],tail))
    return count_r1_r2,set_r1_r2


def get_rel_entset(data_arr):
    rel={}
    for data in data_arr:
        if data[1] not in rel:
            rel[data[1]]=set()
        rel[data[1]].add((data[0],data[2]))
    return rel

def get_ent_to_rel(data_arr):
    ent_to_rel={}
    for data in data_arr:
        pair=(data[0],data[2])
        if pair not in ent_to_rel:
            ent_to_rel[pair]=[]
        ent_to_rel[pair].append(data[1])
    return ent_to_rel

def load_pickle(filepath):
	with open(filepath,"rb") as f:
		dump=pickle.load(f)
	return dump

def dump_pickle(dump,filepath):
	with open(filepath,"wb") as f:
		pickle.dump(dump,f)

def get_relation_dict(data_arr):
	relation_dict={}
	count={}

	for data in data_arr:
	    if data[1] not in relation_dict:
	    	relation_dict[data[1]]=[]
	    	count[data[1]]=0
	    relation_dict[data[1]].append((data[0],data[2]))
	    count[data[1]]+=1

	return count,relation_dict    