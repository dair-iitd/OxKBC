import pickle

import numpy as np

import utils
import logging
from templates.template import TemplateBaseClass


class Template4(TemplateBaseClass):
    """
    Template: e1~e' ^ e'r e2
    """
 
    def __init__(self,kb,base_model,use_hard_triple_scoring=True,load_table=None,dump_file=None):
        super().__init__()
        self.kb=kb
        self.base_model=base_model
        self.use_hard_triple_scoring=use_hard_triple_scoring

        if(load_table==None):
            logging.info("Load table is None, so beginning process_data")
            self.process_data()
            logging.info("Process_data done")
            logging.info("Begin Build table")
            self.build_table()
            logging.info("END Build table")
            logging.info("Begin dump data")
            self.dump_data(dump_file)
            logging.info("END dump table")

        else:
            self.load_table(load_table)

    def process_data(self):
        """
        maps (r,e2) to all e1 in data
        stores unique e1_r for building table
        """
        self.dict_r_e2={}
        self.unique_e1_r={}

        for facts in self.kb.facts:
            key=(facts[1],facts[2])
            if(key not in self.dict_r_e2):
                self.dict_r_e2[key]=[]
            self.dict_r_e2[key].append(facts[1])

            if((facts[0],facts[1]) not in self.unique_e1_r):
                self.unique_e1_r[(facts[0],facts[1])]=len(self.unique_e1_r)

    def build_table(self):
        """
        a table for each unique (e1,r)
        """
        entities=len(self.kb.entity_map)
        self.table={}
        ctr = 0
        for (e1,r) in self.unique_e1_r.keys():
            if ctr%250==0:
                logging.info("Processed %d"%(ctr))
            score_dict={}
            for u in range(entities):
                sc,be = self.compute_score((e1,r,u))
                if(sc!=0):
                    score_dict[u] = (sc,be)
            if(len(score_dict.keys()) > 0):
                self.table[(e1,r)] = score_dict
            ctr+=1



    def dump_data(self,filename):
        dump_dict={}
        dump_dict['dict_r_e2']=self.dict_r_e2
        dump_dict['unique_e1_r']=self.unique_e1_r
        dump_dict['table']=self.table

        with open(filename,'wb')  as inputfile:
            pickle.dump(dump_dict,inputfile)
    
    def load_table(self,filename):
        with open(filename,"rb") as f:
            dump_dict=pickle.load(f)
        self.dict_r_e2=dump_dict['dict_r_e2']
        self.unique_e1_r=dump_dict['unique_e1_r']
        self.table=dump_dict['table']
    

    def compute_score(self,triple):
        '''
        Returns template score for given triple
        Iterates over all e1,r depending on flag of use_hard_triple_scoring
        '''

        assert (len(triple) == 3), "Triple must contain three elements"

        score=0
        best=-1
        e2=triple[2]
        r=triple[1]

        if(self.use_hard_triple_scoring==False):
            entities=len(self.kb.entity_map)

            for e1 in range(entities):
                if(e1==triple[0]):
                    continue
                entity_simi=self.base_model.get_entity_similarity(e1,triple[0])
                model_score=self.base_model.compute_score(e1,r,e2)
                if(score<entity_simi*model_score):
                    score=entity_simi*model_score
                    best=e1

        else:
            key=(r,e2)
            if(key not in self.dict_r_e2):
                score=0
            else:
                ent_list = list(
                    filter(lambda x: x != triple[0], self.dict_r_e2[key]))
                if(len(ent_list)!=0):
                    sim_scores = self.base_model.get_entity_similarity_list(
                        triple[0], ent_list)
                    idx = np.argmax(sim_scores)
                    score = sim_scores[idx]
                    best = ent_list[idx]

        return (score,best)

    def get_input(self,fact):
        key=(fact[0],fact[1])
        features=[0,0,0,0]

        if(key in self.table.keys()):
            val_list= [ x[0] for x in self.table[key].values()]
            if (len(val_list) != 0):
                max_score=max(val_list)
                my_score = self.table[key].get(fact[2],(0,-1))[0]
                index_max=val_list.index(max_score)
                simi=self.base_model.get_entity_similarity(fact[2],list(self.table[key].keys())[index_max])
                rank=utils.get_rank(val_list,my_score)
                features=[my_score,max_score,simi,rank]

        return features

    def get_explanation(self, fact):
        """
        returns best entity for this fact, score,
        best answer for (e1,r,?), best_score, best_entity for best answer,
        zi_score
        """
        
        key = (fact[0], fact[1])
        features=[-1,-1,-1,-1,-1,-1]

        if(key in self.table.keys()):
            val_list = [x[0] for x in self.table[key].values()]

            if (len(val_list) != 0):
                my_score = self.table[key].get(fact[2], (0, -1))[0]
                my_best = self.table[key].get(fact[2], (0, -1))[1]

                index_max = np.argmax(val_list)
                best_score = val_list[index_max]
                best_answer=list(self.table[key].keys())[index_max]
                best_answer_relation=self.table[key].get(best_answer, (0, -1))[1]

                mean=np.mean(val_list)
                std=np.std(val_list)

                z_score=(my_score-mean)/(std+utils.EPSILON)

                features = [my_score,my_best,best_score,best_answer,best_answer_relation,z_score]

        return features