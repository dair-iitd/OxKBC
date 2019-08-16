#ToDO:

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

			self.exp_template = '$html_fact_rprime'

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
			self.dict_e1_r = {}
			self.cache_r_r1_r2 = {}

			for facts in self.kb.facts:
				key = (facts[0], facts[2])
				# dict_e1_e2
				if(key not in self.dict_e1_e2):
					self.dict_e1_e2[key] = []
				self.dict_e1_e2[key].append(facts[1])

				# dict_e1_r
				if((facts[0], facts[1]) not in self.dict_e1_r):
					self.dict_e1_r[(facts[0], facts[1])] = []
				self.dict_e1_r[(facts[0], facts[1])].append(facts[2])

				# dict_e
				if facts[0] not in self.dict_e:
					self.dict_e[facts[0]] = []
				self.dict_e[facts[0]].append(facts[2])

				# unique_e1_r being created
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

		def dump_data(self, filename):
			dump_dict = {}
			dump_dict['dict_e'] = self.dict_e
			dump_dict['dict_e1_e2'] = self.dict_e1_e2
			dump_dict['unique_e1_r'] = self.unique_e1_r
			dump_dict['dict_e1_r'] = self.dict_e1_r

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
			self.dict_e1_r = dump_dict['dict_e1_r']

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
			r  = triple[1]
			e1 = triple[0]

			if(self.use_hard_triple_scoring == False):
				raise NotImplementedError
			else:
				entities = self.dict_e[e1]
				for u1 in range(entities):
					relations1 = self.dict_e1_e2[(e1,u1)]
					relations2 = self.dict_e1_e2[(u1,e2)]
					for r1 in relations1:
						for r2 in relations2:
							if((r,r1,r2) in self.cache_r_r1_r2):
								score = cache_r_r1_r2[(r,r1,r2)]
							else:
								hadamard_r1_r2 = [0,0,0]
								hadamard_r1_r2[0] = self.base_model.rel_similarity_re[r1]*self.base_model.rel_similarity_re[r2]
								hadamard_r1_r2[1] = self.base_model.head_rel_similarity_type[r1]*self.base_model.head_rel_similarity_type[r2]
								hadamard_r1_r2[2] = self.base_model.tail_rel_similarity_type[r1]*self.base_model.tail_rel_similarity_type[r2]

								similarity_with_r = [0,0,0]
								similarity_with_r[0] = hadamard_r1_r2[0].dot(self.base_model.rel_similarity_re[r])
								similarity_with_r[1] = hadamard_r1_r2[1].dot(self.base_model.head_rel_similarity_re[r])
								similarity_with_r[2] = hadamard_r1_r2[2].dot(self.base_model.tail_rel_similarity_re[r])

								score = similarity_with_r[0] * similarity_with_r[1] * similarity_with_r[2]
								cache_r_r1_r2[(r,r1,r2)] = score
							if(score > best_score):
								best_score = score
								best_r1 = r1
								best_r2 = r2
								best_u1 = u1
			return (score, best_r1, best_u1, best_r2)

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
			pass

		def get_english_explanation(self, fact, explainer):
			pass