import pickle

import numpy as np
import logging
import time


import utils
import string
from templates.template import TemplateBaseClass


class Template3(TemplateBaseClass):
    """
    Template: r~r' ^ e1 r' e2
    """

    def __init__(self, kblist, base_model, use_hard_triple_scoring=True, load_table=None, dump_file=None):
        super().__init__()
        self.kb = kblist[0]
        self.kb_val = kblist[1]
        self.kb_test = kblist[2]
        self.base_model = base_model
        self.use_hard_triple_scoring = use_hard_triple_scoring

        # self.exp_template = 'Since, $rprime is similar to the relation $r and I know that ($e1, $rprime, $e2) , so I can say ($e1, $r, $e2)'
        self.exp_template = 'Since, <b>$rprime</b> is similar to the relation <b>$r</b> and AI knows that <b>($e1, $rprime, $e2)</b> , so AI can say <b>($e1, $r, $e2)</b>'


        if(load_table == None):
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
        maps (e1,e2) to all r in data
        stores unique e1_r for building table
        """
        self.dict_e1_e2 = {}
        self.unique_e1_r = {}

        for facts in self.kb.facts:
            key = (facts[0], facts[2])
            if(key not in self.dict_e1_e2):
                self.dict_e1_e2[key] = []
            self.dict_e1_e2[key].append(facts[1])

            if((facts[0], facts[1]) not in self.unique_e1_r):
                self.unique_e1_r[(facts[0], facts[1])] = len(self.unique_e1_r)

        for facts in self.kb_val.facts:
            if((facts[0], facts[1]) not in self.unique_e1_r):
                self.unique_e1_r[(facts[0], facts[1])] = len(self.unique_e1_r)

        for facts in self.kb_test.facts:
            if((facts[0], facts[1]) not in self.unique_e1_r):
                self.unique_e1_r[(facts[0], facts[1])] = len(self.unique_e1_r)

    def build_table(self):
        """
        a table for each unique (e1,r)
        """
        entities = len(self.kb.entity_map)
        self.table = {}
        self.stat_table = {}
        ctr = 0
        start_time = time.time()
        for (e1, r) in self.unique_e1_r.keys():
            if ctr % 250 == 0:
                logging.info("Processed %d in %f seconds" %
                             (ctr, time.time()-start_time))
                start_time = time.time()
            score_dict = {}
            for u in range(entities):
                sc, be = self.compute_score((e1, r, u))
                if(sc != 0):
                    score_dict[u] = (sc, be)

            if(len(score_dict.keys()) > 0):
                self.table[(e1, r)] = score_dict
                val_list = [x[0] for x in self.table[(e1, r)].values()]
                mean = np.mean(val_list)
                std = np.std(val_list)
                max_score = max(val_list)
                index_max = val_list.index(max_score)
                simi_index = list(self.table[(e1, r)].keys())[index_max]
                stat = {"mean": mean, "std": std, "max_score": max_score,
                        "index_max": index_max, "simi_index": simi_index}
                self.stat_table[(e1, r)] = stat
            ctr += 1

    def dump_data(self, filename):
        dump_dict = {}
        dump_dict['dict_e1_e2'] = self.dict_e1_e2
        dump_dict['unique_e1_r'] = self.unique_e1_r
        dump_dict['table'] = self.table
        dump_dict['stat_table'] = self.stat_table

        with open(filename, 'wb') as inputfile:
            pickle.dump(dump_dict, inputfile)

    def load_table(self, filename):
        with open(filename, "rb") as f:
            dump_dict = pickle.load(f)
        self.dict_e1_e2 = dump_dict['dict_e1_e2']
        self.unique_e1_r = dump_dict['unique_e1_r']
        self.table = dump_dict['table']
        self.stat_table = dump_dict['stat_table']

    def compute_score(self, triple):
        '''
        Returns template score for given triple
        Iterates over all e1,r depending on flag of use_hard_triple_scoring
        '''

        assert (len(triple) == 3), "Triple must contain three elements"

        score = 0
        best = -1
        e2 = triple[2]
        e1 = triple[0]

        if(self.use_hard_triple_scoring == False):
            relations = len(self.kb.relation_map)

            for r in range(relations):
                if(r == triple[1]):
                    continue
                relation_simi = self.base_model.get_relation_similarity(
                    r, triple[1])
                model_score = self.base_model.compute_score(e1, r, e2)
                if(score < relation_simi*model_score):
                    score = relation_simi*model_score
                    best = r

        else:
            key = (e1, e2)
            if(key not in self.dict_e1_e2):
                score = 0
            else:
                rel_list = list(
                    filter(lambda x: x != triple[1], self.dict_e1_e2[key]))
                if(len(rel_list) != 0):
                    sim_scores = self.base_model.get_relation_similarity_list(
                        triple[1], rel_list)
                    logging.debug(sim_scores)
                    idx = np.argmax(sim_scores)
                    score = sim_scores[idx]
                    best = rel_list[idx]

        return (score, best)

    def get_input(self, fact):
        key = (fact[0], fact[1])
        features = [0, 0, 0, 0, 0, 0, 0]

        if(key in self.table.keys()):
            val_list = [x[0] for x in self.table[key].values()]
            if (len(val_list) != 0):
                max_score = self.stat_table[key]['max_score']
                my_score = self.table[key].get(fact[2], (0, -1))[0]
                simi = self.base_model.get_entity_similarity(
                    fact[2], self.stat_table[key]['simi_index'])
                rank = utils.get_rank(val_list, my_score)
                conditional_rank = rank*1.0/len(val_list)
                mean = self.stat_table[key]['mean']
                std = self.stat_table[key]['std']
                features = [my_score, max_score, simi,
                            rank, conditional_rank, mean, std]

        return features

    def get_explanation(self, fact):
        """
        returns best relation for this fact, score,
        best answer for (e1,r,?), best_score, best_relation for best answer,
        zi_score
        """
        key = (fact[0], fact[1])
        features = [0, -1, 0, -1, -1, 0]

        if(key in self.table.keys()):
            val_list = [x[0] for x in self.table[key].values()]

            if (len(val_list) != 0):
                my_score = self.table[key].get(fact[2], (0, -1))[0]
                my_best = self.table[key].get(fact[2], (0, -1))[1]

                index_max = np.argmax(val_list)
                best_score = val_list[index_max]
                best_answer = list(self.table[key].keys())[index_max]
                best_answer_relation = self.table[key].get(
                    best_answer, (0, -1))[1]

                mean = np.mean(val_list)
                std = np.std(val_list)

                z_score = (my_score-mean)/(std+utils.EPSILON)

                features = [my_score, my_best, best_score,
                            best_answer, best_answer_relation, z_score]

        return features

    def get_english_explanation(self, fact, enum_to_id, rnum_to_id, eid_to_name, rid_to_name):
        mapped_fact = utils.map_fact(fact, enum_to_id, rnum_to_id)
        mapped_fact_name = utils.map_fact(mapped_fact, eid_to_name, rid_to_name)
        rprime_id = rnum_to_id[self.get_explanation(fact)[1]]
        rprime = rid_to_name.get(rprime_id,rprime_id)
        return string.Template(self.exp_template).substitute(e1=mapped_fact_name[0],r=mapped_fact_name[1],e2=mapped_fact_name[2],rprime=rprime)

