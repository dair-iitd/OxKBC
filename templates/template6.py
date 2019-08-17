# ToDO:

import logging
import pickle
import string
import sys
import time

import numpy as np

import utils
from templates.template import TemplateBaseClass
from tqdm import tqdm
import multiprocessing
from IPython.core.debugger import Pdb
import os

#os.system("taskset -p 0xfffff %d" % os.getpid())


class Template6(TemplateBaseClass):
    """
    (e1 r e2) since (e1 r1 u1) and (u1 r2 e2) and ((r1.r2)~r)
    """

    def __init__(self, kblist, base_model, use_hard_triple_scoring=True, load_table=None, dump_file=None, parts=1, offset=0):
        super().__init__()
        self.kb = kblist[0]
        self.kb_val = kblist[1]
        self.kb_test = kblist[2]
        self.base_model = base_model
        self.use_hard_triple_scoring = use_hard_triple_scoring
        # Pdb().set_trace()
        self.exp_template = '$html_fact_rprime'
        self.parts = parts
        self.offset = offset
        self.dump_file = dump_file
        if(load_table == None):
            logging.info("Load table is None, so beginning process_data")
            self.process_data()
            logging.info("Process_data done")
            logging.info("BEGIN Build table")
            self.build_table()
            logging.info("END Build table")
            logging.info("BEGIN dump data")
            self.dump_data(dump_file)
            logging.info("END dump table")
        else:
            self.load_table(load_table)

    def process_data(self):
        """
        dict_e1_e2: maps (e1,e2) to all r in data [TRAINING]
        unique_e1_r: stores unique e1_r for building table [TRAINING + TEST + VAL]
        dict_e: stores all possible e2 for a given e1 [TRAINING]
        dict_e1_r: stores all possible e2 for given e1 and r [TRAINING]
        cache_r_r1_r2: a cache for similarity(r,hadamard(r1,r2))
        """
        self.dict_e1_e2 = {}
        self.unique_e1_r = {}
        self.dict_e = {}
        #self.dict_e1_r = {}
        self.cache_r_r1_r2 = {}

        for facts in self.kb.facts:
            key = (facts[0], facts[2])
            # dict_e1_e2
            if(key not in self.dict_e1_e2):
                self.dict_e1_e2[key] = []
            self.dict_e1_e2[key].append(facts[1])

            # dict_e1_r
            # if((facts[0], facts[1]) not in self.dict_e1_r):
            #    self.dict_e1_r[(facts[0], facts[1])] = []
            #self.dict_e1_r[(facts[0], facts[1])].append(facts[2])

            # dict_e
            if facts[0] not in self.dict_e:
                self.dict_e[facts[0]] = set()
            self.dict_e[facts[0]].add(facts[2])

            # unique_e1_r being created
            if((facts[0], facts[1]) not in self.unique_e1_r):
                self.unique_e1_r[(facts[0], facts[1])] = len(self.unique_e1_r)

        for facts in self.kb_val.facts:
            if((facts[0], facts[1]) not in self.unique_e1_r):
                self.unique_e1_r[(facts[0], facts[1])] = len(self.unique_e1_r)

        for facts in self.kb_test.facts:
            if((facts[0], facts[1]) not in self.unique_e1_r):
                self.unique_e1_r[(facts[0], facts[1])] = len(self.unique_e1_r)

        #
        self.sorted_e1_r_list = list(self.unique_e1_r.keys())
        self.sorted_e1_r_list.sort()

    def process(self, key):
        e1, r = key
        #global ctr
        #global total_size
        #global start_time
        # if ctr % 250 == 1:
        #    logging.info("Processed %d in %f seconds. Avg size %f" %
        #                 (ctr, time.time()-start_time, total_size/ctr))
        logging.info("Ctr: {}".format(self.ctr))
        score_dict = {}
        for u in range(len(self.kb.entity_map)):
            score, r1, u1, r2 = self.compute_score((e1, r, u))
            if(score != 0):
                score_dict[u] = (score, r1, u1, r2)

        #total_size += len(score_dict)

        if(len(score_dict.keys()) > 0):
            #self.table[(e1, r)] = score_dict
            val_list = [x[0] for x in score_dict.values()]
            mean = np.mean(val_list)
            std = np.std(val_list)
            max_score = max(val_list)
            index_max = val_list.index(max_score)
            simi_index = list(score_dict.keys())[index_max]
            stat = {"mean": mean, "std": std, "max_score": max_score,
                    "index_max": index_max, "simi_index": simi_index}
            #self.stat_table[(e1, r)] = stat
        self.ctr += 1
        return (key, score_dict, stat)

    def build_table(self):
        """
        self.table = {}
        self.stat_table = {}
        self.ctr = 0
        total_size = 0
        start_time = time.time()

        pool = multiprocessing.Pool()
        #Pdb().set_trace() 
        list_of_tables = list(tqdm(pool.imap_unordered(self.process, list(self.unique_e1_r.keys())[:100]), total=100))
        #with tqdm(total= 100) as pbar:
        #    for i, _ in tqdm(enumerate(pool.imap_unordered(self.process, list(self.unique_e1_r.keys())[:100]))):
        #        pbar.update()
        #list_of_tables = pool.map(self.process,list(self.unique_e1_r.keys())[:100])
        #list_of_tables = map(self.process,list(self.unique_e1_r.keys())[:100])
        #Pdb().set_trace() 
        for key, score_dict, stat in list_of_tables:
            self.table[key] = score_dict
            self.stat_table[key] = stat
        #Pdb().set_trace()
        pool.close()



        """
        # a table for each unique (e1,r)
        # Pdb().set_trace()
        entities = len(self.kb.entity_map)
        self.table = {}
        self.stat_table = {}
        ctr = 0
        start_time = time.time()
        total_size = 0

        start_index = self.offset*int(len(self.sorted_e1_r_list)/self.parts)
        end_index = (self.offset+1)*int(len(self.sorted_e1_r_list)/self.parts)
        if self.offset == self.parts-1:
            end_index = len(self.sorted_e1_r_list)

        total_e1_r = end_index - start_index
        last_logged_at = 0

        if os.path.exists(self.dump_file+'.partial'):
            logging.info("Found Partial dump at: {}. Loading from it: ".format(
                self.dump_file+'.partial'))
            self.load_table(self.dump_file+'.partial')

        logging.info("Total (e1,r): {}. Start Index: {} End Index: {}".format(
            len(self.unique_e1_r), start_index, end_index))
        for (e1, r) in tqdm(self.sorted_e1_r_list[start_index: end_index]):
            if (e1, r) in self.table:
                continue
            #
            if ctr % 250 == 1:
                logging.info("Processed %d in %f seconds. Avg size %f" %
                             (ctr, time.time()-start_time, total_size/ctr))
                start_time = time.time()
            score_dict = {}
            for u in range(entities):
                score, r1, u1, r2 = self.compute_score((e1, r, u))
                if(score != 0):
                    score_dict[u] = (score, r1, u1, r2)

            total_size += len(score_dict)

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
            if int(100*(ctr/total_e1_r)) >= (last_logged_at + 20):
                last_logged_at = int(100*(ctr/total_e1_r))
                logging.info(
                    "***START Creating a partial dump at : {} %".format(last_logged_at))
                self.dump_data(self.dump_file+'.partial')
                logging.info(
                    "***END Created a partial dump at : {} %".format(last_logged_at))

    def dump_data(self, filename):
        dump_dict = {}
        dump_dict['dict_e'] = self.dict_e
        dump_dict['dict_e1_e2'] = self.dict_e1_e2
        dump_dict['unique_e1_r'] = self.unique_e1_r
        #dump_dict['dict_e1_r'] = self.dict_e1_r

        dump_dict['table'] = self.table
        dump_dict['stat_table'] = self.stat_table

        with open(filename, 'wb') as inputfile:
            pickle.dump(dump_dict, inputfile)

    def load_table(self, filename):
        with open(filename, "rb") as f:
            dump_dict = pickle.load(f)
        self.dict_e = dump_dict['dict_e']
        self.dict_e1_e2 = dump_dict['dict_e1_e2']
        self.unique_e1_r = dump_dict['unique_e1_r']
        #self.dict_e1_r = dump_dict['dict_e1_r']

        self.table = dump_dict['table']
        self.stat_table = dump_dict['stat_table']

    def compute_score(self, triple):
        '''
        Returns template score for given triple
        Iterates over all e1,r depending on flag of use_hard_triple_scoring
        '''

        assert (len(triple) == 3), "Triple must contain three elements"

        best_score = 0
        best_r1 = -1
        best_u1 = -1
        best_r2 = -1

        e2 = triple[2]
        r = triple[1]
        e1 = triple[0]

        if(self.use_hard_triple_scoring == False):
            raise NotImplementedError
        else:
            entities = self.dict_e.get(e1, [])
            for u1 in entities:
                if u1 != e2:
                    relations1 = self.dict_e1_e2.get((e1, u1), [])
                    relations2 = self.dict_e1_e2.get((u1, e2), [])
                    for r1 in relations1:
                        for r2 in relations2:
                            if((r, r1, r2) in self.cache_r_r1_r2):
                                score = self.cache_r_r1_r2[(r, r1, r2)]
                            else:
                                hadamard_r1_r2 = self.base_model.get_hadamard_product(
                                    r1, r2)
                                r_emb = self.base_model.get_relation_embedding(
                                    r)
                                score = self.base_model.get_relation_similarity_from_embedding(
                                    hadamard_r1_r2, r_emb)
                                self.cache_r_r1_r2[(r, r1, r2)] = score
                            if(score > best_score):
                                best_score = score
                                best_r1 = r1
                                best_r2 = r2
                                best_u1 = u1
        return (best_score, best_r1, best_u1, best_r2)

    def get_input(self, fact):
        key = (fact[0], fact[1])
        features = [0, 0, 0, 0, 0, 0, 0]

        if(key in self.table.keys()):
            val_list = [x[0] for x in self.table[key].values()]
            if (len(val_list) != 0):
                max_score = self.stat_table[key]['max_score']
                my_score = self.table[key].get(fact[2], (0, -1, -1, -1))[0]
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
        returns the best two length path for this fact, score, 
        best answer for (e1,r,?), best_score, best path for best answer, z_score
        we have : (score, r1, u1, r2)
    
        """
        key = (fact[0], fact[1])
        #features = [score, 2lp, top score, top u, top path, z score]
        features = [0,(-1,-1,-1),0,-1,(-1,-1,-1),0] 
        if(key in self.table.keys()):
            my_table_stats = self.stat_table[key]
            if (len(self.table[key]) != 0):
                my_score_and_path = self.table[key].get(fact[2], (0,-1,-1,-1))
                my_score = my_score_and_path[0]
                my_best_path = my_score_and_path[1:]
                best_answer = my_table_stats['simi_index']
                best_score_and_path = self.table[key].get(best_answer)
                best_score = best_score_and_path[0]
                best_answer_path = best_score_and_path[1:]
                z_score = (my_score - my_table_stats['mean'])/(my_table_stats['std'] + utils.EPSILON)
                features = [my_score, my_best_path, best_score, best_answer,best_answer_path, z_score] 

        return features

    def get_english_explanation(self, fact, explainer):
        pass
