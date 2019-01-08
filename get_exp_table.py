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


def read_data(path):
    data = []
    with open(path, "r",errors='ignore',encoding='ascii') as infile:
        lines = infile.readlines()
        data = [line.strip().split() for line in lines]
    return data


def map_data(data, mapped_entity, mapped_relation):
    mapped_data = []
    for line in data:
        try:
            mapped_data.append([mapped_entity[line[0]], mapped_relation[line[1]], mapped_entity[line[2]]])
        except KeyError as e:
            logging.warn('Got Key Error for line %s' % (' '.join(line)))
    return mapped_data


def read_entity_names(path):
    entity_names = {}
    with open(path, "r",errors='ignore',encoding='ascii') as f:
        lines = f.readlines()
        for line in lines:
            content = line.split()
            if content[0] in entity_names:
                logging.warn('Duplicate Entity found %s in line %s' % (content[0],' '.join(line)))
            entity_names[content[0]] = ' '.join(content[1:-2])
    return entity_names


def read_pkl(filename):
    with open(filename, "rb") as f:
        pkl_dict = pickle.load(f)
    return pkl_dict


def get_inverse_dict(mydict):
    inverse_dict = {}
    for k in mydict.keys():
        if mydict[k] in inverse_dict:
            raise "Cannot Construct inverse dictionary, as function not one-one"
        inverse_dict[mydict[k]] = k
    return inverse_dict


def convert_to_pandas(table, header):
    return pd.DataFrame(table, columns=header)


def get_header(t_type):
    if(t_type == 1 or t_type == 2):
        msg = 'most freq for '+('relation' if t_type == 1 else 'entity')
        return ['e1', 'r', 'e2', 'my_score', msg, 'best_score', 'z_score']
    elif(t_type == 3 or t_type == 4):
        msg = 'best_'+('relation' if t_type == 3 else 'entity')
        return ['e1', 'r', 'e2', 'my_score', 'my_'+msg, 'best_score', 'best_answer', 'best_answer_'+msg, 'z_score']
    elif(t_type == 5):
        return ['e1', 'r', 'e2', 'my_score', 'my_best_e', 'my_best_r', 'best_score',
                'best_answer', 'best_answer_best_e', 'best_answer_best_r', 'z_score']
    else:
        raise "Invalid Type"


def get_word_exp(exp, t_type, entity_id_inverse_map, relation_id_inverse_map, entity_name_map):
    exp_word_list = []
    if(t_type == 1 or t_type == 2):
        exp_word_list.append(exp[0])
        exp_word_list.append(exp[1])
        exp_word_list.append(entity_name_map.get(
            entity_id_inverse_map.get(exp[2], "None"), "None"))
        exp_word_list.append(exp[3])
    elif(t_type == 3):
        exp_word_list.append(exp[0])
        exp_word_list.append(relation_inverse_map.get(exp[1], "None"))
        exp_word_list.append(exp[2])
        exp_word_list.append(entity_name_map.get(
            entity_id_inverse_map.get(exp[3], "None"), "None"))
        exp_word_list.append(relation_inverse_map.get(exp[4], "None"))
        exp_word_list.append(exp[5])
    elif(t_type == 4):
        exp_word_list.append(exp[0])
        exp_word_list.append(entity_name_map.get(
            entity_id_inverse_map.get(exp[1], "None"), "None"))
        exp_word_list.append(exp[2])
        exp_word_list.append(entity_name_map.get(
            entity_id_inverse_map.get(exp[3], "None"), "None"))
        exp_word_list.append(entity_name_map.get(
            entity_id_inverse_map.get(exp[4], "None"), "None"))
        exp_word_list.append(exp[5])
    elif(t_type == 5):
        exp_word_list.append(exp[0])
        exp_word_list.append(entity_name_map.get(
            entity_id_inverse_map.get(exp[1][0], "None"), "None"))
        exp_word_list.append(relation_inverse_map.get(exp[1][1], "None"))
        exp_word_list.append(exp[2])
        exp_word_list.append(entity_name_map.get(
            entity_id_inverse_map.get(exp[3], "None"), "None"))
        exp_word_list.append(entity_name_map.get(
            entity_id_inverse_map.get(exp[4][0], "None"), "None"))
        exp_word_list.append(relation_inverse_map.get(exp[4][1], "None"))
        exp_word_list.append(exp[5])
    else:
        raise "Invalid Template ID"

    return exp_word_list


def generate_exp(data, template_obj_list, entity_id_inverse_map, relation_id_inverse_map, entity_name_map):
    word_exps = {}
    for t in template_obj_list:
        t_type = int(type(t).__name__[-1])
        logging.info(
            "Beginning explaination generation for template %d" % (t_type))
        header = get_header(t_type)
        exp_list = []
        ctr = 0
        start_time = time.time()
        for triple in data:
            if ctr % 1000 == 0:
                logging.info("Processed %d in %f seconds" %
                             (ctr, time.time()-start_time))
                start_time = time.time()
            if '__INV' in relation_id_inverse_map[triple[1]]:
                continue
            temp_list = []
            temp_list.append(entity_name_map.get(entity_id_inverse_map[triple[0]],"None"))
            temp_list.append(relation_id_inverse_map[triple[1]])
            temp_list.append(entity_name_map.get(entity_id_inverse_map[triple[2]],"None"))
            exp = t.get_explanation(triple)
            temp_list.extend(get_word_exp(
                exp, t_type, entity_id_inverse_map, relation_id_inverse_map, entity_name_map))
            exp_list.append(temp_list)
            ctr += 1
        exp_df = convert_to_pandas(exp_list, header)
        word_exps[t_type] = exp_df[exp_df.duplicated(
            ['e1', 'r', 'e2']) == False]

    return word_exps


def write_word_exps(word_exps, output_root_path):
    for t_id in word_exps.keys():
        word_exps[t_id].to_csv(os.path.join(output_root_path, str(
            t_id)+".txt"), encoding='utf-8', index=False, sep='\t')
    if(len(word_exps.keys()) > 1):
        result = reduce(lambda left, right: pd.merge(left, right, how='inner', left_on=[
                        'e1', 'r', 'e2'], right_on=['e1', 'r', 'e2']), word_exps.values())
        result.to_csv(os.path.join(output_root_path, "merged.txt"),
                      encoding='utf-8', index=False, sep='\t')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', help="Name of the dataset as in data folder", required=True)
    parser.add_argument('-w', '--model_weights',
                        help="Pickle file of model wieghts", required=True)
    parser.add_argument('-l', '--template_load_dir',
                        required=False, default=None)
    parser.add_argument('-o', '--output_data_dir',
                        required=True, default=None)
    parser.add_argument('-tf', '--test_file',
                        required=True, default=None)
    parser.add_argument('--txt', action='store_true', required=False,
                        help='Use the flag if the data is not mapped with ids.')
    parser.add_argument('--t_ids', nargs='+', type=int, required=True,
                        help='List of templates to build objects for')
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

    distmult_dump = read_pkl(args.model_weights)
    logging.info("Read Model Dump")

    if(args.txt):
        data = read_data(args.test_file)
        mapped_data = map_data(
            data, distmult_dump['entity_to_id'], distmult_dump['relation_to_id'])
    else:
        mapped_data = np.loadtxt(args.test_file).astype(np.int32)

    np.random.shuffle(mapped_data)
    logging.info("Loaded test file from %s and randomly shuffled it" %
                 (args.test_file))
    entity_names = read_entity_names(os.path.join(
        data_root, "entity_mid_name_type_typeid.txt"))

    entity_inverse_map = get_inverse_dict(distmult_dump['entity_to_id'])
    relation_inverse_map = get_inverse_dict(distmult_dump['relation_to_id'])

    template_objs = template_builder.template_obj_builder(
        data_root, args.model_weights, args.template_load_dir, None, "distmult", args.t_ids, True)

    word_exps = generate_exp(mapped_data, template_objs,
                             entity_inverse_map, relation_inverse_map, entity_names)

    logging.info("Generated explanations")
    write_word_exps(word_exps, args.output_data_dir)
    logging.info("Written explanations to %s" % (args.output_data_dir))
