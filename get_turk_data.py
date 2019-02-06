import argparse
import logging
import os
import pickle
import time
from functools import reduce


import numpy as np
import pandas as pd

import template_builder
import utils

NO_EXPLANATION = "Sorry, I have no explanation for the fact"
def english_exp(mapped_data, template_objs, entity_inverse_map, relation_inverse_map, entity_names,relation_names):
    explanations = []
    for fact in mapped_data:
        pred = int(fact[-1])
        if(pred==0):
            explanations.append(NO_EXPLANATION)
        else:
            explanations.append(template_objs[pred-1].get_english_explanation(fact[:-1], entity_inverse_map, relation_inverse_map, entity_names, relation_names))
    return explanations 

def write_english_exps(explanations, output_file):
    with open(output_file,'w') as f:
        for el in explanations:
            f.write(el+'\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', help="Name of the dataset as in data folder", required=True)
    parser.add_argument('-w', '--model_weights',
                        help="Pickle file of model wieghts", required=True)
    parser.add_argument('-l', '--template_load_dir',
                        required=False, default=None)
    parser.add_argument('-o', '--output_filename',
                        required=True, default=None)
    parser.add_argument('-tf', '--test_file',
                        required=True, default=None)
    parser.add_argument('-tp', '--t_pred',
                        required=True, help='List of template predictions of data',default=None)
    parser.add_argument('--data_repo_root',
                        required=False, default='data')
    parser.add_argument('--log_level',
                        default='INFO',
                        dest='log_level',
                        type=utils._log_level_string_to_int,
                        nargs='?',
                        help='Set the logging output level. {0}'.format(utils._LOG_LEVEL_STRINGS))
    
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s :: %(asctime)s - %(message)s',
                        level=args.log_level, datefmt='%d/%m/%Y %I:%M:%S %p')

    data_root = os.path.join(args.data_repo_root, args.dataset)

    distmult_dump = utils.read_pkl(args.model_weights)
    logging.info("Read Model Dump")

    data = utils.read_data(args.test_file)
    mapped_data = np.array(utils.map_data(data, distmult_dump['entity_to_id'], distmult_dump['relation_to_id']))
    logging.info("Loaded test file from %s" %(args.test_file))

    template_predictions = np.loadtxt(args.t_pred)
    logging.info("Loaded test file predictions  %s" %(args.t_pred))

    if(len(template_predictions) != len(mapped_data)):
        logging.error("Unequal length of predictions and data")
        exit(-1)

    mapped_data = np.hstack((mapped_data,template_predictions.reshape(mapped_data.shape[0],1)))
    mapped_data = mapped_data.astype(np.int32)

    entity_names = utils.read_entity_names(os.path.join(data_root, "entity_mid_name_type_typeid.txt"))
    relation_names = utils.read_relation_names(os.path.join(data_root, "relation_name.txt"))

    entity_inverse_map = utils.get_inverse_dict(distmult_dump['entity_to_id'])
    relation_inverse_map = utils.get_inverse_dict(distmult_dump['relation_to_id'])

    template_objs = template_builder.template_obj_builder(data_root, args.model_weights, args.template_load_dir, None, "distmult", [1,2,3,4,5], True)


    word_exps = english_exp(mapped_data, template_objs, entity_inverse_map, relation_inverse_map, entity_names,relation_names)

    logging.info("Generated explanations")
    write_english_exps(word_exps, args.output_filename)
    logging.info("Written explanations to %s" % (args.output_filename))
