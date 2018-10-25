import argparse
import logging
import os
import pickle
import sys

import numpy as np

import kb
import template_builder
import utils


def get_input(fact, y, template_obj_list):
    x = []
    for template in template_obj_list:
        x.extend(template.get_input(fact))
    x.append(y)
    return x


def preprocess(kb, template_obj_list, negative_count=10):

    new_facts = []
    ctr = 0
    for facts in kb.facts:
        ctr += 1
        if(ctr % 100 == 0):
            logging.info("Processed {0} facts out of {1}".format(
                ctr, len(kb.facts)))
        ns = np.random.randint(0, len(kb.entity_map), negative_count)
        no = np.random.randint(0, len(kb.entity_map), negative_count)

        new_facts.append(get_input(facts, 1, template_obj_list))

        for neg_facts in range(negative_count):
            new_fact = (ns[neg_facts], facts[1], facts[2])
            new_facts.append(get_input(new_fact, 0, template_obj_list))
            new_fact = (facts[0], facts[1], no[neg_facts])
            new_facts.append(get_input(new_fact, 0, template_obj_list))

    return np.array(new_facts)


def write_to_file(facts, fileprefix):
    with open(fileprefix+".pkl", "wb") as f:
        pickle.dump(facts, f)
    logging.info("Written data to {0}".format(fileprefix+".txt"))
    np.savetxt(fileprefix+".txt", facts, delimiter=',')
    logging.info("Written data to {0}".format(fileprefix+".pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', help="Name of the dataset as in data folder", required=True)
    parser.add_argument(
        '-m', '--model_type', help="model name. Can be distmult or complex ", required=True)
    parser.add_argument('-s', '--sm_data_write',
                        required=True, default="selection_module.data")
    parser.add_argument('-w', '--model_weights',
                        help="Pickle file of model wieghts", required=True)
    parser.add_argument('-l', '--template_load_dir',
                        required=False, default=None)
    parser.add_argument('-v', '--oov_entity', required=False, default=True)
    parser.add_argument('--t_ids', nargs='+', type=int, required=True,
                        help='List of templates to run for')
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

    dataset_root = os.path.join(args.data_repo_root, args.dataset)
    kvalid, template_objs = template_builder.template_obj_builder(dataset_root, args.model_weights, args.template_load_dir,
                                                                  None, args.model_type, args.t_ids, args.oov_entity)

    new_facts = preprocess(kvalid, template_objs)

    write_to_file(new_facts, args.sm_data_write)
