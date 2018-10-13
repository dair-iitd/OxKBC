from template import TemplateBaseClass
import pickle
import numpy as np
import time
# from joblib import Parallel, delayed

class Template5(TemplateBaseClass):
    """
    Template: e1~e' ^ r~r' ^ e'r'e2
    """
 
    def __init__(self,kb,base_model,use_hard_triple_scoring=True,load_table=None,dump_file=None):
        super().__init__()
        self.kb=kb
        self.base_model=base_model
        self.use_hard_triple_scoring=use_hard_triple_scoring

        if(load_table==None):
            print("Load table is None, so beginning process_data")
            self.process_data()
            print("Process_data done")
            print("Begin Build table")
            self.build_table()
            print("END Build table")
            print("Begin dump data")
            self.dump_data(dump_file)
            print("END dump table")

        else:
            self.load_table(load_table)

    def process_data(self):
        """
        maps e2 to all (e1,r) in data
        stores unique e1_r for building table
        """
        self.dict_e2={}
        self.unique_e1_r={}

        for facts in self.kb.facts:
            if(facts[2] not in self.dict_e2):
                self.dict_e2[facts[2]]= {}
                self.dict_e2[facts[2]]["e1"] = []
                self.dict_e2[facts[2]]["r"] = [] 

            self.dict_e2[facts[2]]["r"].append(facts[1])            
            self.dict_e2[facts[2]]["e1"].append(facts[0])

            if((facts[0],facts[1]) not in self.unique_e1_r):
                self.unique_e1_r[(facts[0],facts[1])]=len(self.unique_e1_r)
        
        entity_similarity_re = np.matmul(self.base_model.dump['entity_real'],np.transpose(self.base_model.dump['entity_real']))
        print("Calculated re")
        entity_similarity_type = np.matmul(self.base_model.dump['entity_type'],np.transpose(self.base_model.dump['entity_type']))
        print("Calculated type")
        self.entity_similarity = np.multiply(entity_similarity_re,entity_similarity_type)
        print("Calculated entity similarity")
        
        rel_similarity_re = np.matmul(self.base_model.dump['rel_real'],np.transpose(self.base_model.dump['rel_real']))
        print("Calculated rel re")
        head_rel_similarity_type = np.matmul(self.base_model.dump['head_rel_type'],np.transpose(self.base_model.dump['head_rel_type']))
        print("Calculated head rel type")
        tail_rel_similarity_type = np.matmul(self.base_model.dump['tail_rel_type'],np.transpose(self.base_model.dump['tail_rel_type']))
        print("Calculated tail rel type")
        self.rel_similarity = np.multiply(np.multiply(rel_similarity_re,head_rel_similarity_type),tail_rel_similarity_type)
        print("Calculated rel similarity")
        

    def build_table(self):
        """
        a table for each unique (e1,r)
        """
        entities=len(self.kb.entity_map)
        self.table={}
        total_els = len(self.unique_e1_r.keys())
        ctr = 0
        # def one_e1_r(t):
        #     e1,r = t
        #     score_dict={}
        #     for u in range(entities):
        #         sc,be = self.compute_score((e1,r,u))
        #         if(sc!=0):
        #             score_dict[u] = (sc,be)
        #     return score_dict

        # results = Parallel(n_jobs=4,verbose=15,batch_size=10)(delayed(one_e1_r)(t) for t in self.unique_e1_r.keys())
        
        # ctr=0
        # for (e1,r) in self.unique_e1_r.keys():
            # self.table[(e1,r)] = results[ctr]
            # ctr+=1

        start_time = time.time()
        for (e1,r) in self.unique_e1_r.keys():
            if ctr%250==0:
                print("Processed %d in %f seconds"%(ctr,time.time()-start_time))
                start_time = time.time()
            score_dict={}
            for u in range(entities):
                sc,be = self.compute_score((e1,r,u))
                if(sc!=0):
                    score_dict[u] = (sc,be)
            self.table[(e1,r)]=score_dict
            ctr+=1


    def dump_data(self,filename):
        dump_dict={}
        dump_dict['dict_e2']=self.dict_e2
        dump_dict['unique_e1_r']=self.unique_e1_r
        dump_dict['table']=self.table

        with open(filename,'wb')  as inputfile:
            pickle.dump(dump_dict,inputfile)
    
    def load_table(self,filename):
        with open(filename,"rb") as f:
            dump_dict=pickle.load(f)
        self.dict_e2=dump_dict['dict_e2']
        self.unique_e1_r=dump_dict['unique_e1_r']
        self.table=dump_dict['table']
    

    def compute_score(self,triple):
        '''
        Returns template score for given triple
        Iterates over all e1,r depending on flag of use_hard_triple_scoring
        '''
        assert (len(triple) == 3), "Triple must contain three elements"

        score=0
        best=(-1,-1)
        
        e2=triple[2]

        if(self.use_hard_triple_scoring==False):
            entities=len(self.kb.entity_map)
            relations=len(self.kb.relation_map)

            for e1 in range(entities):
                for r in range(relations):
                    entity_simi=self.base_model.get_entity_similarity(e1,triple[0])
                    relation_simi=self.base_model.get_relation_similarity(r,triple[1])                    
                    model_score=self.base_model.compute_score(e1,r,e2)
                    new_sc = entity_simi*relation_simi*model_score
                    if(score<new_sc):
                        score=new_sc
                        best=(e1,r)

        else:
            if(e2 not in self.dict_e2):
                score=0
            else:
                # for (e1,r) in self.dict_e2[e2]:
                #     # entity_simi=self.base_model.get_entity_similarity(e1,triple[0])
                #     # relation_simi=self.base_model.get_relation_similarity(r,triple[1])
                #     entity_simi=self.entity_similarity[e1,triple[0]]
                #     relation_simi=self.rel_similarity[r,triple[1]]
                #     new_sc = entity_simi*relation_simi

                #     if(score<new_sc):
                #         score=new_sc
                #         best=(e1,r)
                e_sim_scores = np.take(self.entity_similarity[triple[0]],self.dict_e2[e2]["e1"])
                r_sim_scores = np.take(self.rel_similarity[triple[1]],self.dict_e2[e2]["r"])
                sim_scores = e_sim_scores*r_sim_scores
                idx = np.argmax(sim_scores)
                score = sim_scores[idx]
                best = (self.dict_e2[e2]["e1"][idx],self.dict_e2[e2]["r"][idx])
        return (score,best)
