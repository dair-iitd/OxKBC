import logging
import pickle
import string
import time
from collections import Counter

import numpy as np

import utils
from templates.template import TemplateBaseClass


class Template1(TemplateBaseClass):
    """
    Template: Most frequent for this relation
    """

    def __init__(self, kblist, base_model, use_hard_triple_scoring=True,
                 load_table=None, dump_file=None, parts = 1, offset=0):
        super().__init__()
        self.kb = kblist[0]
        self.base_model = base_model
        self.use_hard_triple_scoring = True
        # self.exp_template = '<b>$e2</b> is frequently seen with the relation <b>\"$r\"</b> hence <b>$e1 $r $e2</b>'
        # self.exp_template = 'Since, $e2 is most frequently occuring entity for the relation $r, so I can say ($e1, $r, $e2)'
        # self.exp_template = '<b>$e2</b> is frequently seen with the relation <b>\"$r\"</b>'
        self.exp_template = '<b><font color="blue">$e2</font></b> is $why_frequent with the relation <b><font color="green">\"$r\"</font></b>'

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
        self.stat_table = {}
        start_time = time.time()
        ctr = 0
        for rel in self.relation_map:
            if ctr % 250 == 0:
                logging.info("Processed %d in %f seconds" %
                             (ctr, time.time()-start_time))
                start_time = time.time()
            score_dict = {}
            for u in range(nentities):
                sc = self.compute_score((None, rel, u))
                if(sc != 0):
                    score_dict[u] = sc
            if(len(score_dict.keys()) > 0):
                self.table[rel] = score_dict
                val_list = [x for x in score_dict.values()]
                mean = np.mean(val_list)
                std = np.std(val_list)
                max_score = max(val_list)
                index_max = val_list.index(max_score)
                simi_index = list(score_dict.keys())[index_max]
                stat = {"mean": mean, "std": std, "max_score": max_score,
                        "index_max": index_max, "simi_index": simi_index}
                self.stat_table[rel] = stat
            ctr += 1

    def dump_data(self, filename):
        dump_dict = {}
        dump_dict['relation_map'] = self.relation_map
        dump_dict['table'] = self.table
        dump_dict['stat_table'] = self.stat_table
        with open(filename, 'wb') as outfile:
            pickle.dump(dump_dict, outfile)

    def load_table(self, filename):
        with open(filename, "rb") as f:
            dump_dict = pickle.load(f)
        self.relation_map = dump_dict['relation_map']
        self.table = dump_dict['table']
        self.stat_table = dump_dict['stat_table']

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

    def get_input(self, fact):
        key = fact[1]
        features = [0, 0, 0, 0, 0, 0, 0]

        if(key in self.table.keys()):
            max_score = self.stat_table[key]['max_score']
            my_score = self.table[key].get(fact[2], 0)
            simi = self.base_model.get_entity_similarity(
                fact[2], self.stat_table[key]['simi_index'])
            rank = utils.get_rank(self.table[key].values(), my_score)
            conditional_rank = rank*1.0/len(self.table[key].values())
            mean = self.stat_table[key]['mean']
            std = self.stat_table[key]['std']
            features = [my_score, max_score, simi,
                        rank, conditional_rank, mean, std]
        return features

    def get_explanation(self, fact):
        """
        returns score,
        best answer for (e1,r,?), best_score,
        zi_score
        """
        key = fact[1]
        features = [0, -1, 0, 0]
        if(key in self.table):
            val_list = list(self.table[key].values())
            if (len(val_list) != 0):
                my_score = self.table[key].get(fact[2], 0)
                index_max = np.argmax(val_list)
                best_score = val_list[index_max]
                best_answer = list(self.table[key].keys())[index_max]
                mean = np.mean(val_list)
                std = np.std(val_list)
                z_score = (my_score-mean)/(std+utils.EPSILON)
                features = [my_score, best_score,
                            best_answer, z_score]
        return features

    def get_english_explanation(self, fact, explainer):
        why_frequent = explainer.get_relation_frequent(fact)
        named_fact = explainer.name_fact(fact)
        return string.Template(self.exp_template).substitute(e1=named_fact[0], r=named_fact[1], e2=named_fact[2], why_frequent=why_frequent)
