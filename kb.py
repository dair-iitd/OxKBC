import logging
import pickle

import numpy as np


class KnowledgeBase():
    """
    Stores a knowledge base as an numpy array. Can be generated from a file.
    Also stores the entity/relation mappings (which is the mapping from entity
    names to entity id)
    """

    def __init__(self, facts_filename, em=None, rm=None, add_unknowns=True,
                 load_map_file=None):
        """
        Initializes an object of the class KnowledgeBase\n
        :param facts_filename: The file name to read the kb from\n
        :param em: Prebuilt entity map to be used. Can be None for a new map to
        be created\n
        :param rm: prebuilt relation map to be used. Can be None for a new map
        to be created\n 
        :param add_unknowns: Whether new entites are to be acknowledged or put
        as <UNK> token.
        :param load_map_file: filename if mappings need to be loaded from file.
        """

        if(load_map_file is not None):
            with open(load_map_file, "rb") as f:
                dump_dict = pickle.load(f)
            em = dump_dict["entity_map"]
            rm = dump_dict["relation_map"]
        self.entity_map = {} if em is None else em
        self.relation_map = {} if rm is None else rm

        self.facts = []
        if (facts_filename is None):
            return
        facts = []
        with open(facts_filename, "r") as f:
            lines = f.readlines()
            lines = [l.split() for l in lines]
            for l in lines:
                if(add_unknowns):
                    if(l[0] not in self.entity_map):
                        self.entity_map[l[0]] = len(self.entity_map)
                    if(l[2] not in self.entity_map):
                        self.entity_map[l[2]] = len(self.entity_map)
                    if(l[1] not in self.relation_map):
                        self.relation_map[l[1]] = len(self.relation_map)
                facts.append([self.entity_map.get(l[0], len(self.entity_map)-1),
                              self.relation_map.get(
                                  l[1], len(self.relation_map)-1),
                              self.entity_map.get(l[2], len(self.entity_map)-1)])
        self.facts = np.array(facts, dtype='int64')
        logging.info("Loaded {0} facts from file {1}".format(len(self.facts), facts_filename))


def union(kb_list):
    """
    Computes a union of multiple knowledge bases\n
    :param kb_list: A list of kb\n
    :return: The union of all kb in kb_list
    """
    l = [k.facts for k in kb_list]
    k = KnowledgeBase(None, kb_list[0].entity_map, kb_list[0].relation_map)
    k.facts = np.concatenate(l, axis=0)
    logging.info("Created a union of {0} kbs. Total facts in union = {1}\n".format(len(kb_list), len(k.facts)))
    return k


def dump_kb_mappings(kb, kb_name):
    """
    Dumps the entity and relation mapping in a kb\n
    :param kb: The kb\n 
    :param kb_name: The file name under which the mappings should be stored.\n
    :return:
    """
    dump_dict = {}
    dump_dict["entity_map"] = kb.entity_map
    dump_dict["relation_map"] = kb.relation_map
    with open(kb_name+".ids.pkl", "wb") as f:
        pickle.dump(dump_dict, f)
    logging.info("Dumped entity and relation maps to {0}\n".format(kb_name))
    