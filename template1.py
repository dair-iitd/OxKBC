from template import TemplateBaseClass
import pickle
from collections import Counter

class Template1(TemplateBaseClass):
    """
    Template: Most frequent for this relation
    """
 
    def __init__(self,kb,base_model,use_hard_triple_scoring=True,load_table=None,dump_file=None):
        super().__init__()
        self.kb=kb
        self.base_model=base_model
        assert (use_hard_triple_scoring),"Hard Scoring is necessary for this template"
        self.use_hard_triple_scoring=use_hard_triple_scoring

        if(load_table==None):
            self.process_data()
            self.build_table()
            self.dump_data(dump_file)

        else:
            self.load_table(load_table)

    def process_data(self):
        """
        Stores all tail entities and their counts for a relation
        """
        self.relation_map = {}    
        for fact in self.kb.facts:
            if (fact[1] not in self.relation_map):
                self.relation_map[fact[1]] = {}
                self.relation_map[fact[1]]["len"] = 0
                self.relation_map[fact[1]]["cts"] = []

            self.relation_map[fact[1]]["len"] +=1
            self.relation_map[fact[1]]["cts"].append(fact[2])
        
        for rel in self.relation_map:
            self.relation_map[rel]["cts"] = Counter(self.relation_map[rel]["cts"])
         
    def build_table(self):
        nentities = len(self.kb.entity_map)
        self.table = {}

        for rel in self.relation_map:
            score_lis = []
            for u in range(nentities):
                score_lis.append(self.get_score((None,rel,u)))
            self.table[rel] = score_lis
        
    def dump_data(self,filename):
        dump_dict = {}
        dump_dict['relation_map'] = self.relation_map
        dump_dict['table'] = self.table
        with open(filename,'wb')  as outfile:
            pickle.dump(dump_dict,outfile)
    
    def load_table(self,filename):
        with open(filename,"rb") as f:
            dump_dict = pickle.load(f)
        self.relation_map = dump_dict['relation_map']
        self.table = dump_dict['table']

    def compute_score(self,triple):
        '''
        Returns template score for given triple
        '''
        assert (len(triple)==3),"Triple must contain three elements"
        rel = triple[1]
        e2 = triple[2]
        if rel not in self.relation_map:
            return 0
        else:
            return self.relation_map[rel]["cts"].get(e2,0)*1.0/self.relation_map[rel]["len"] 
