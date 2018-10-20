import os
import sys
import pickle
import kb
import settings
import numpy as np

def get_input(fact,y,template_obj_list):
    x=[]
    for template in template_obj_list:
        x.extend(template.get_input(fact))
    
    x.append(y)
    return x

#TODO write both txt and pickle file 
def preprocess(kb,template_obj_list,negative_count=10):

    template_obj_list=[]

    # for file in range(settings.templates):
    #     filepath=os.path.join(pickle_file_path,str(file)+'.pkl')
    #     template_list.append(pickle.load(filepath,'rb'))


    new_facts=[]
        
    for facts in kb.facts:
        ns = np.random.randint(0, len(kb.entity_map), negative_count)
        no = np.random.randint(0, len(kb.entity_map), negative_count)

        new_facts.append(get_input(facts,1,template_obj_list))
        
        for neg_facts in range(negative_count):
            new_fact=(ns[neg_facts],facts[1],facts[2])
            new_facts.append(get_input(new_fact,0,template_obj_list))
            new_fact=(facts[0],facts[1],no[neg_facts])
            new_facts.append(get_input(new_fact,0,template_obj_list))
            

    dump=open('selection_module.data','w')

    dump_str='\n'.join(list(map(lambda x: ','.join(list(map(str,x))), new_facts)))
    print(dump_str,file=dump)
    dump.close()

class RandomTemplate():
    def __init__(self,n):
        self.n = n
    
    def get_input(self,fact):
        return (np.random.random(self.n))


template_obj_list = [ RandomTemplate(5) for _ in range(5)]

