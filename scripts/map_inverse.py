# This script is used to map the entity and relation names to their ids (as present in the fb15k dataset)

import argparse
import logging
import os
import pickle
import time

import numpy as np
import pandas as pd


def read_data(path):
    data = []
    with open(path, "r", errors='ignore', encoding='ascii') as infile:
        lines = infile.readlines()
        data = [line.strip().split() for line in lines]
    return data


def map_data(data, mapped_entity=None, mapped_relation=None):
    mapped_data = []
    for line in data:
        try:
            e1_mapped = mapped_entity[line[0]] if mapped_entity is not None else line[0]
            r_mapped = mapped_relation[line[1]] if mapped_relation is not None else line[1]
            e2_mapped = mapped_entity[line[2]] if mapped_entity is not None else line[2]
            mapped_data.append([e1_mapped, r_mapped, e2_mapped])
        except KeyError as e:
            logging.warn('Got Key Error for line %s' % (' '.join(line)))
    return mapped_data

def read_entity_names(path):
    entity_names = {}
    with open(path, "r", errors='ignore', encoding='ascii') as f:
        lines = f.readlines()
        for line in lines:
            content_raw = line.split('\t')
            content = [el.strip() for el in content_raw]
            if content[0] in entity_names:
                logging.warn('Duplicate Entity found %s in line %s' %
                                (content[0], line))
            name = content[1]
            wiki_id = content[2]
            # entity_names[content[0]] = {"name": name, "wiki_id": wiki_id}
            entity_names[content[0]] = name
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', help="Name of the dataset as in data folder", required=True)
    parser.add_argument('-w', '--model_weights',
                        help="Pickle file of model weights", required=True)
    parser.add_argument('-cf', '--convert_file',
                        required=True, default=None)
    parser.add_argument('-of', '--out_file',
                        required=True, default=None)
    parser.add_argument('--data_repo_root',
                        required=False, default='data')
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s :: %(asctime)s - %(message)s',
                        level=logging.INFO, datefmt='%d/%m/%Y %I:%M:%S %p')

    data_root = os.path.join(args.data_repo_root, args.dataset)

    distmult_dump = read_pkl(args.model_weights)
    logging.info("Read Model Dump")

    mapped_data = np.loadtxt(args.convert_file).astype(np.int32)

    np.random.shuffle(mapped_data)
    logging.info("Loaded test file from %s and randomly shuffled it" %
                 (args.convert_file))
    entity_names = read_entity_names(os.path.join(
        data_root, "mid2wikipedia_cleaned.tsv"))

    entity_inverse_map = get_inverse_dict(distmult_dump['entity_to_id'])
    relation_inverse_map = get_inverse_dict(distmult_dump['relation_to_id'])

    inverse_mapped_data = map_data(
        mapped_data, entity_inverse_map, relation_inverse_map)
    inverse_mapped_data_name = map_data(
        inverse_mapped_data, entity_names, None)

    logging.info('id_data length = {}'.format(len(inverse_mapped_data)))
    logging.info('name_data length = {}'.format(len(inverse_mapped_data_name)))

    fid = open(args.out_file+'_id.txt', 'w')
    fname = open(args.out_file+'_name.txt', 'w')

    for i in range(len(inverse_mapped_data)):
        el_id = inverse_mapped_data[i]
        el_name = inverse_mapped_data_name[i]
        if(all(len(x) >= 3 for x in el_id) and all(len(x) >= 3 for x in el_name)):
            fid.write('\t'.join(el_id)+'\n')
            fname.write('\t'.join(el_name)+'\n')

    fid.close()
    fname.close()

    logging.info("Written inverse mappings to %s" % (args.out_file))
