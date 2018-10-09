import pickle
import numpy as np
import math

class TypedDM():
    
    def init(self,pickle_dump_file):
        with open(pickle_dump_file, "rb") as f:
            self.dump = pickle.load(f)
    
    def get_entity_similarity(self,e1,e2):
        entity_similarity=np.dot(self.dump['entity_real'][e1],self.dump['entity_real'][e2])
        type_compatibility = np.dot(self.dump['entity_type'][e1], self.dump['entity_type'][e2])
        return entity_similarity*type_compatibility
    
    def get_relation_similarity(self, r1, r2):
        relation_similarity = np.dot(
            self.dump['rel_real'][r1], self.dump['rel_real'][r2])
        type_compatibility_head = np.dot(
            self.dump['head_rel_type'][r1], self.dump['head_rel_type'][r2])
        type_compatibility_tail = np.dot(
            self.dump['tail_rel_type'][r1], self.dump['tail_rel_type'][r2])
        return relation_similarity*type_compatibility_head*type_compatibility_tail
    
    def sigmoid(self,x):
        return 1/(1+math.exp(-x))

    def compute_score(self,e1,r,e2):
        score=self.dump['entity_real'][e1]*self.dump['rel_real'][r]*self.dump['entity_real'][e2]
        head_type_compatibility = self.dump['entity_type'][e1]*self.dump['head_rel_type']
        tail_type_compatibility = self.dump['entity_type'][e1]*self.dump['tail_rel_type']
        score=np.sum(score)
        head_type_compatibility=np.sum(head_type_compatibility)
        tail_type_compatibility=np.sum(tail_type_compatibility)
        return self.sigmoid(score)*self.sigmoid(head_type_compatibility)*self.sigmoid(tail_type_compatibility)


class TypedComplex():

    def init(self, pickle_dump_file):
        with open(pickle_dump_file, "rb") as f:
            self.dump = pickle.load(f)

    def get_entity_similarity(self, e1, e2):
        entity_similarity = np.dot(self.dump['entity_real'][e1], self.dump['entity_real'][e2])+np.dot(
            self.dump['entity_im'][e1], self.dump['entity_im'][e2])
        type_compatibility = np.dot(
            self.dump['entity_type'][e1], self.dump['entity_type'][e2])
        return entity_similarity*type_compatibility

    def get_relation_similarity(self, r1, r2):
        relation_similarity = np.dot(self.dump['rel_real'][r1], self.dump['rel_real'][r2])+np.dot(
            self.dump['rel_im'][r1], self.dump['rel_im'][r2])
        type_compatibility_head = np.dot(
            self.dump['head_rel_type'][r1], self.dump['head_rel_type'][r2])
        type_compatibility_tail = np.dot(
            self.dump['tail_rel_type'][r1], self.dump['tail_rel_type'][r2])
        return relation_similarity*type_compatibility_head*type_compatibility_tail

    def sigmoid(self, x):
        return 1/(1+math.exp(-x))

    def compute_score(self, e1, r, e2):
        s_re = self.dump['entity_real'][e1]
        s_im = self.dump['entity_im'][e1]
        o_re = self.dump['entity_real'][e2]
        o_im = self.dump['entity_im'][e2]
        r_re = self.dump['rel_real'][r]
        r_im = self.dump['rel_im'][r]

        score = (s_re*o_re+s_im*o_im)*r_re + (s_re*o_im-s_im*o_re)*r_im
        head_type_compatibility = self.dump['entity_type'][e1] * \
            self.dump['head_rel_type']
        tail_type_compatibility = self.dump['entity_type'][e1] * \
            self.dump['tail_rel_type']
        score = np.sum(score)
        head_type_compatibility = np.sum(head_type_compatibility)
        tail_type_compatibility = np.sum(tail_type_compatibility)
        return self.sigmoid(score)*self.sigmoid(head_type_compatibility)*self.sigmoid(tail_type_compatibility)
