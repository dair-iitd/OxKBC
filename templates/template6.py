import numpy as np
from templates.template import TemplateBaseClass
class Template6(TemplateBaseClass):
	"""
	(e1 r e2) since (e1 r1 u1) and (u1 r2 e2) and ((r1.r2)~r)
	"""
	def __init__(self, kblist, base_model, use_hard_triple_scoring=True, load_table=None, dump_file=None):
		super().__init__()
		self.kb = kblist[0]
		self.kb_val = kblist[1]
		self.kb_test = kblist[2]
		self.base_model = base_model
		self.use_hard_triple_scoring = use_hard_triple_scoring

		self.exp_template = @HELP

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
        maps (e1,e2) to all r in data
        stores unique e1_r(in train, test and val data) for building table
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
                score, r1, u1, r2 = self.compute_score((e1, r, u))
                if(sc != 0):
                    score_dict[u] = (score, r1, u1, r2)

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

	def dump_data(self):
		pass

	def compute_score(self, triple):
        '''
        Returns template score for given triple
        Iterates over all e1,r depending on flag of use_hard_triple_scoring
        '''

        assert (len(triple) == 3), "Triple must contain three elements"

        score = 0
        best_r1 = -1
        best_u1 = -1
        best_r2 = -1

        e2 = triple[2]
        e1 = triple[0]

        if(self.use_hard_triple_scoring == False):
            # relations = len(self.kb.relation_map)
            # for r in range(relations):
            #     if(r == triple[1]):
            #         continue
            #     relation_simi = self.base_model.get_relation_similarity(
            #         r, triple[1])
            #     model_score = self.base_model.compute_score(e1, r, e2)
            #     if(score < relation_simi*model_score):
            #         score = relation_simi*model_score
            #         best = r
            entities = len(self.kb.entity_map)
            relations = len(self.kb.relation_map)
        	for u1 in range(entities):
        		MAY WANT TO DELETE SOME ELEMENTS
        		for r1 in relations:
        			for r2 in relations:
                		model_score1 = self.base_model.compute_score(e1, r1, u1)
                		model_score2 = self.base_model.compute_score(u1, r2, e2)
                		relations_product = COMPUTE PRODUCT
        				if(relations_product*model_score1*model_score2>score):
        					score = relations_product*model_score1*model_score2
        					best_r1 = r1
        					best_r2 = r2
        					best_u1 = u1
        else:
            # key = (e1, e2)
            # if(key not in self.dict_e1_e2):
            #     score = 0
            # else:
            #     rel_list = list(
            #         filter(lambda x: x != triple[1], self.dict_e1_e2[key]))
            #     if(len(rel_list) != 0):
            #         sim_scores = self.base_model.get_relation_similarity_list(
            #             triple[1], rel_list)
            #         logging.debug(sim_scores)
            #         idx = np.argmax(sim_scores)
            #         score = sim_scores[idx]
            #         best = rel_list[idx]
        	entities = len(self.kb.entity_map)
        	for u1 in range(entities):
        		MAY WANT TO DELETE SOME ELEMENTS
        		relations1 = self.dict_e1_e2[(e1,u1)]
        		relations2 = self.dict_e1_e2[(u1,e2)]
        		PROCESS TO UPDATE SCORE, BEST_R1, BEST_U1, BEST_R2
        		relations_product = COMPUTE PRODUCT
        return (score, best_r1, best_u1, best_r2)