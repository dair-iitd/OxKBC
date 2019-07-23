import argparse
import logging
import os
import pickle
import sys

import numpy as np

import kb
import template_builder
import utils


def get_input(fact, y, template_obj_list,add_ids):
    if (add_ids):
        x = [fact[0],fact[1],fact[2]]
    else:
        x = []
    for template in template_obj_list:
        x.extend(template.get_input(fact))
    x.append(y)
    return x


def preprocess(kb, template_obj_list, negative_count,add_ids,y_labels):

    new_facts = []
    ctr = 0
    for facts in kb.facts:
        if(ctr % 500 == 0):
            logging.info("Processed {0} facts out of {1}".format(
                ctr, len(kb.facts)))
        ns = np.random.randint(0, len(kb.entity_map), negative_count)
        no = np.random.randint(0, len(kb.entity_map), negative_count)

        new_facts.append(get_input(facts, y_labels[ctr], template_obj_list,add_ids))

        for neg_facts in range(negative_count):
            new_fact = (ns[neg_facts], facts[1], facts[2])
            new_facts.append(get_input(new_fact, 0, template_obj_list,add_ids))
            new_fact = (facts[0], facts[1], no[neg_facts])
            new_facts.append(get_input(new_fact, 0, template_obj_list,add_ids))

        ctr += 1

    return np.array(new_facts)


def write_to_file(facts, fileprefix):
    with open(fileprefix+".pkl", "wb") as f:
        pickle.dump(facts, f)
    logging.info("Written data to {0}".format(fileprefix+".txt"))
    np.savetxt(fileprefix+".txt", facts, delimiter=',',fmt='%.6e')
    logging.info("Written data to {0}".format(fileprefix+".pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', help="Name of the dataset as in data folder", required=True)
    parser.add_argument(
        '-m', '--model_type', help="model name. Can be distmult or complex ", required=True)
    parser.add_argument('-f', '--preprocess_file',
                        required=True, help="Path of the file which is to be preprocessed")
    parser.add_argument('-y', '--y_labels',
                        required=False, help="Path of the y label file, which has same number of lines as preprocess_file. Use it to generate test or valid data, which has y labels instead of 1 and 0 in last column",default='')
    parser.add_argument('-s', '--sm_data_write',
                        required=True, default="selection_module.data")
    parser.add_argument('-w', '--model_weights',
                        help="Pickle file of model wieghts", required=True)
    parser.add_argument('-l', '--template_load_dir',
                        required=False, default=None)
    parser.add_argument('-v', '--oov_entity', required=False, default=True)
    parser.add_argument('--t_ids', nargs='+', type=int, required=True,
                        help='List of templates to run for')
    parser.add_argument('--del_ids', action='store_true', required=False,
                        help='Use the flag to delete entity and relation ids to the start of row\nDefault behaviour is to add ids in front of each record.')
    parser.add_argument('--data_repo_root',
                        required=False, default='data')
    parser.add_argument('--negative_count',
                        required=False, type=int,default=2)
    parser.add_argument('--log_level',
                        default='INFO',
                        dest='log_level',
                        type=utils._log_level_string_to_int,
                        nargs='?',
                        help='Set the logging output level. {0}'.format(utils._LOG_LEVEL_STRINGS))
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s :: %(asctime)s - %(message)s',
                        level=args.log_level, datefmt='%d/%m/%Y %I:%M:%S %p')

    if(args.y_labels != '' and args.negative_count!=0):
        logging.error('Cannot generate random samples with y labels. If using --y_labels use flag --negative_count 0 also')
        exit(-1)

    dataset_root = os.path.join(args.data_repo_root, args.dataset)
    template_objs = template_builder.template_obj_builder(dataset_root, args.model_weights,args.template_load_dir,None, args.model_type, args.t_ids, args.oov_entity)

    ktrain = template_objs[0].kb

    k_preprocess = kb.KnowledgeBase(args.preprocess_file, ktrain.entity_map, ktrain.relation_map,add_unknowns=not args.oov_entity)

    y_labels = [1 for _ in range(k_preprocess.facts.shape[0])]

    if(args.y_labels != ''):
        y_labels = np.loadtxt(args.y_labels)
        if(y_labels.shape[0] != k_preprocess.facts.shape[0]):
            logging.error('Number of facts and their y labels do not match')
            exit(-1)

    new_facts = preprocess(k_preprocess, template_objs, args.negative_count, not args.del_ids, y_labels)

    write_to_file(new_facts, args.sm_data_write)
