from template import TemplateBaseClass
import pickle
from sm import utils

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
        total_els = len(self.unique_e1_r.keys())
        ctr = 0
        for (e1,r) in self.unique_e1_r.keys():
            if ctr%250==0:
                print("Processed %d"%(ctr))
            score_dict={}
            for u in range(entities):
                sc,be = self.compute_score((e1,r,u))
                if(sc!=0):
                    score_dict[u] = (sc,be)

            self.table[(e1,r)]=score_dict
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
                for e1 in self.dict_r_e2[key]:
                    entity_simi=self.base_model.get_entity_similarity(e1,triple[0])
                    if(score<entity_simi):
                        score=entity_simi
                        best=e1
        return (score,best)

    def get_input(self,fact):
        key=(fact[1],fact[2])
        features=[0,0,0,0]

        if(key in self.table.keys()):
            val_list=list(self.table[key].values)
            max_score=max(val_list)
            my_score=self.table[key][fact[2]][0]
            index_max=val_list.index(max_score)
            simi=self.base_model.get_entity_similarity(fact[2],self.table[key].keys()[index_max])
            rank=utils.get_rank(val_list,my_score)
            features=[my_score,max_score,simi,rank]

        return features
