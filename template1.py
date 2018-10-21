from template import TemplateBaseClass
import pickle
from collections import Counter
import numpy as np
from sm import utils

class Template1(TemplateBaseClass):
    """
    Template: Most frequent for this relation
    """

    def __init__(self, kb, base_model, use_hard_triple_scoring=True,
                 load_table=None, dump_file=None):
        super().__init__()
        self.kb = kb
        self.base_model = base_model
        self.use_hard_triple_scoring = True

        if(load_table == None):
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

            self.relation_map[fact[1]]["len"] += 1
            self.relation_map[fact[1]]["cts"].append(fact[2])

        for rel in self.relation_map:
            self.relation_map[rel]["cts"] = Counter(
                self.relation_map[rel]["cts"])

    def build_table(self):
        nentities = len(self.kb.entity_map)
        self.table = {}

        for rel in self.relation_map:
            score_dict = {}
            for u in range(nentities):
                sc = self.compute_score((None, rel, u))
                if(sc!=0):
                    score_dict[u] = sc
            self.table[rel] = score_dict

    def dump_data(self, filename):
        dump_dict = {}
        dump_dict['relation_map'] = self.relation_map
        dump_dict['table'] = self.table
        with open(filename, 'wb') as outfile:
            pickle.dump(dump_dict, outfile)

    def load_table(self, filename):
        with open(filename, "rb") as f:
            dump_dict = pickle.load(f)
        self.relation_map = dump_dict['relation_map']
        self.table = dump_dict['table']

    def compute_score(self, triple):
        '''
        Returns template score for given triple
        '''
        assert (len(triple) == 3), "Triple must contain three elements"
        rel = triple[1]
        e2 = triple[2]
        if rel not in self.relation_map:
            return 0
        else:
            return self.relation_map[rel]["cts"].get(e2, 0) * 1.0 / self.relation_map[rel]["len"]

    def get_input(self,fact):
        key = fact[1]
        features = [0,0,0,0]

        if(key in self.table.keys()):
            index_max = np.argmax(self.table[key].values())
            max_score = list(self.table[key].values())[index_max]
            my_score = self.table[key].get(fact[2],0)
            simi = self.base_model.get_entity_similarity(fact[2],index_max)
            rank = utils.get_rank(self.table[key].values(),my_score)
            features = [my_score,max_score,simi,rank]
        return features
