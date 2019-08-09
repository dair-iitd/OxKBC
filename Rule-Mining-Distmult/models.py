import pickle
import numpy as np
import logging
import utils

class TypedDM():
    
    def __init__(self,pickle_dump_file):
        logging.info("Creating Base Model")
        logging.info("Loading Model weights from {0}".format(pickle_dump_file))
        with open(pickle_dump_file, "rb") as f:
            self.dump = pickle.load(f)

        entity_similarity_re = np.matmul(self.dump['entity_real'],np.transpose(self.dump['entity_real']))
        logging.info("Calculated entity real similarity matrix")
        entity_similarity_type = np.matmul(self.dump['entity_type'],np.transpose(self.dump['entity_type']))
        logging.info("Calculated entity type similarity matrix")
        self.entity_similarity = np.multiply(entity_similarity_re,entity_similarity_type)
        logging.info("Calculated entity similarity matrix")
        
        rel_similarity_re = np.matmul(self.dump['rel_real'],np.transpose(self.dump['rel_real']))
        logging.info("Calculated relation real similarity matrix")
        head_rel_similarity_type = np.matmul(self.dump['head_rel_type'],np.transpose(self.dump['head_rel_type']))
        logging.info("Calculated head relation type similarity matrix")
        tail_rel_similarity_type = np.matmul(self.dump['tail_rel_type'],np.transpose(self.dump['tail_rel_type']))
        logging.info("Calculated type relation type similarity matrix")
        self.relation_similarity = np.multiply(np.multiply(rel_similarity_re,head_rel_similarity_type),tail_rel_similarity_type)
        logging.info("Calculated relation similarity matrix")
        logging.info("Created Object of Base Model")
        self.relation_matrix=np.concatenate((self.dump['rel_real'],self.dump['head_rel_type'],self.dump['tail_rel_type']),axis=1)
        # self.relation_matrix=np.transpose(self.relation_matrix)
        self.first=self.dump['rel_real'].shape[1]
        self.second=self.first+self.dump['head_rel_type'].shape[1]

    def similarity_relembedding(self, r1, r2):
        # relation_similarity = np.dot(
        #     self.dump['rel_real'][r1], self.dump['rel_real'][r2])
        # type_compatibility_head = np.dot(
        #     self.dump['head_rel_type'][r1], self.dump['head_rel_type'][r2])
        # type_compatibility_tail = np.dot(
        #     self.dump['tail_rel_type'][r1], self.dump['tail_rel_type'][r2])
        # return relation_similarity*type_compatibility_head*type_compatibility_tail
        # r2 = np.concatenate((self.dump['rel_real'][r2], self.dump['head_rel_type'][r2], self.dump['tail_rel_type'][r2]))
        simi = np.vdot(r1[0:self.first], r2[0:self.first])
        simi *= np.vdot(r1[self.first:self.second], r2[self.first:self.second])
        simi *= np.vdot(r1[self.second:], r2[self.second:])
        return -simi


    def get_relation_similarity(self, r1, r2,embed=False):
        # relation_similarity = np.dot(
        #     self.dump['rel_real'][r1], self.dump['rel_real'][r2])
        # type_compatibility_head = np.dot(
        #     self.dump['head_rel_type'][r1], self.dump['head_rel_type'][r2])
        # type_compatibility_tail = np.dot(
        #     self.dump['tail_rel_type'][r1], self.dump['tail_rel_type'][r2])
        # return relation_similarity*type_compatibility_head*type_compatibility_tail
        if(embed):
            r2=np.concatenate((self.dump['rel_real'][r2],self.dump['head_rel_type'][r2],self.dump['tail_rel_type'][r2]))
            return np.vdot(r1,r2)
        return self.relation_similarity[r1,r2]
    
    def dot_relation(self,rel1,rel2):
        #rel1_embedding=np.concatenate((self.dump['rel_real'][rel1],self.dump['head_rel_type'][rel1],self.dump['tail_rel_type'][rel1]))
        #rel2_embedding=np.concatenate((self.dump['rel_real'][rel2],self.dump['head_rel_type'][rel2],self.dump['tail_rel_type'][rel2]))
        #return rel1_embedding*rel2_embedding
        rel_real_dot=self.dump['rel_real'][rel1]*self.dump['rel_real'][rel2]
        return np.concatenate((rel_real_dot,self.dump['head_rel_type'][rel1],self.dump['tail_rel_type'][rel2]))

    # def get_relation_similarity(self,rel1,rel2,):

    def get_entity_similarity_list(self,e1,lis):
        return np.take(self.entity_similarity[e1], lis)

    def get_relation_similarity_list(self,r1,lis):
        return np.take(self.relation_similarity[r1], lis)

    def compute_score(self,e1,r,e2):
        score=self.dump['entity_real'][e1]*self.dump['rel_real'][r]*self.dump['entity_real'][e2]
        head_type_compatibility = self.dump['entity_type'][e1]*self.dump['head_rel_type']
        tail_type_compatibility = self.dump['entity_type'][e1]*self.dump['tail_rel_type']
        score=np.sum(score)
        head_type_compatibility=np.sum(head_type_compatibility)
        tail_type_compatibility=np.sum(tail_type_compatibility)
        return utils.sigmoid(score)*utils.sigmoid(head_type_compatibility)*utils.sigmoid(tail_type_compatibility)

class TypedComplex():

    def init(self, pickle_dump_file):
        with open(pickle_dump_file, "rb") as f:
            self.dump = pickle.load(f)
        raise "Not Implemented"

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
        return utils.sigmoid(score)*utils.sigmoid(head_type_compatibility)*utils.sigmoid(tail_type_compatibility)
