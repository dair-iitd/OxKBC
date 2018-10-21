import os
import sys
import pickle
import kb
# import settings
import numpy as np
import argparse

def get_input(fact,y,template_obj_list):
    x=[]
    # print(fact)
    # print(fact)
    for template in template_obj_list:
        # print(template.get_input(fact))
        # print("aman")
        x.extend(template.get_input(fact))
    
    x.append(y)
    # print(x,"**xxx***")
    return x

#TODO write both txt and pickle file 
def preprocess(kb,template_obj_list,negative_count=10):

    # for file in range(settings.templates):
    #     filepath=os.path.join(pickle_file_path,str(file)+'.pkl')
    #     template_list.append(pickle.load(filepath,'rb'))


    new_facts=[]

    ctr = 0
        
    for facts in kb.facts:
        ctr += 1
        if(ctr%100==0):
            print("Processed ",ctr)
        ns = np.random.randint(0, len(kb.entity_map), negative_count)
        no = np.random.randint(0, len(kb.entity_map), negative_count)

        new_facts.append(get_input(facts,1,template_obj_list))
        
        for neg_facts in range(negative_count):
            new_fact=(ns[neg_facts],facts[1],facts[2])
            new_facts.append(get_input(new_fact,0,template_obj_list))
            new_fact=(facts[0],facts[1],no[neg_facts])
            new_facts.append(get_input(new_fact,0,template_obj_list))
    
    # print(new_facts)

    dump=open('selection_module.data','w')

    dump_str='\n'.join(list(map(lambda x: ','.join(list(map(str,x))), new_facts)))
    print(dump_str,file=dump)
    dump.close()


import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', help="Name of the dataset as in data folder", required=True)
    parser.add_argument(
        '-m', '--model_type', help="model name. Can be distmult or complex ", required=True)
    parser.add_argument('-w', '--model_weights',
                        help="Pickle file of model wieghts", required=True)
    parser.add_argument('-l', '--template_load_dir',
                        required=False, default=None)
    parser.add_argument('-v', '--oov_entity', required=False, default=True)
    parser.add_argument('--t_ids', nargs='+', type=int, required=True,
                        help='List of templates to run for')
    parser.add_argument('--data_repository_root',
                        required=False, default='data')
    args = parser.parse_args()

    dataset_root = os.path.join(args.data_repository_root, args.dataset)
    kvalid,template_objs = main.main(dataset_root, args.model_weights, args.template_load_dir,
         None, args.model_type, args.t_ids, args.oov_entity)

    print("Objs is ",template_objs)
    # exit(0)
    preprocess(kvalid,template_objs)
        
