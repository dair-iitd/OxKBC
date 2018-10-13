from template import TemplateBaseClass
import pickle
from collections import Counter


class Template2(TemplateBaseClass):
    """
    Template: Most frequent for this head entity
    """

    def __init__(self, kb, base_model, use_hard_triple_scoring=True, load_table=None, dump_file=None):
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
        self.entity_map = {}
        for fact in self.kb.facts:
            if (fact[0] not in self.entity_map):
                self.entity_map[fact[0]] = {}
                self.entity_map[fact[0]]["len"] = 0
                self.entity_map[fact[0]]["cts"] = []

            self.entity_map[fact[0]]["len"] += 1
            self.entity_map[fact[0]]["cts"].append(fact[2])

        for e1 in self.entity_map:
            self.entity_map[e1]["cts"] = Counter(self.entity_map[e1]["cts"])

    def build_table(self):
        nentities = len(self.kb.entity_map)
        self.table = {}

        for e1 in self.entity_map:
            score_dict = {}
            for u in range(nentities):
                sc = self.compute_score((e1, None, u))
                if(sc!=0):
                    score_dict[u] = sc
            self.table[e1] = score_dict

    def dump_data(self, filename):
        dump_dict = {}
        dump_dict['entity_map'] = self.entity_map
        dump_dict['table'] = self.table
        with open(filename, 'wb') as outfile:
            pickle.dump(dump_dict, outfile)

    def load_table(self, filename):
        with open(filename, "rb") as f:
            dump_dict = pickle.load(f)
        self.entity_map = dump_dict['entity_map']
        self.table = dump_dict['table']

    def compute_score(self, triple):
        '''
        Returns template score for given triple
        '''
        assert (len(triple) == 3), "Triple must contain three elements"
        e1 = triple[0]
        e2 = triple[2]
        if e1 not in self.entity_map:
            return 0
        else:
            return self.entity_map[e1]["cts"].get(e2, 0)*1.0/self.entity_map[e1]["len"]
