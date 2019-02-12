import argparse
import logging
import os
import pickle
import string
import time

import numpy as np
import pandas as pd

import template_builder
import utils

NO_EXPLANATION = "No explanation for this fact"


def english_exp_rules(mapped_data, predictions, entity_inverse_map, relation_inverse_map, entity_names, relation_names):

    def english_from_fact(fact, enum_to_id, rnum_to_id, eid_to_name, rid_to_name):
        exp_template = '<b>$e1 $r $e2</b>'
        mapped_fact = utils.map_fact(fact, enum_to_id, rnum_to_id)
        mapped_fact_name = utils.map_fact(
            mapped_fact, eid_to_name, rid_to_name)
        return string.Template(exp_template).substitute(e1=mapped_fact_name[0], r=mapped_fact_name[1], e2=mapped_fact_name[2])

    explanations = []

    def lambda_english_from_fact(x):
        return english_from_fact(x, entity_inverse_map, relation_inverse_map, entity_names, relation_names)

    for itr in range(len(mapped_data)):
        fact = mapped_data[itr]
        pred = predictions[itr]
        if(pred[1] == -1):
            if(pred[0] == ""):
                explanations.append(NO_EXPLANATION)
            else:
                to_explain = lambda_english_from_fact(fact)
                explaining_fact = (
                    mapped_data[itr][0], pred[0], mapped_data[itr][2])
                explaining_fact = lambda_english_from_fact(explaining_fact)
                # explanations.append(to_explain+" because "+explaining_fact)
                explanations.append("because "+explaining_fact)
        else:
            to_explain = lambda_english_from_fact(fact)
            explaining_fact1 = (mapped_data[itr][0], pred[0][0], pred[1])
            explaining_fact2 = (pred[1], pred[0][1], mapped_data[itr][2])
            explaining_fact1 = lambda_english_from_fact(explaining_fact1)
            explaining_fact2 = lambda_english_from_fact(explaining_fact2)
            # explanations.append(to_explain+" because because AI knows " +explaining_fact+" and "+explaining_fact2)
            explanations.append("because " +explaining_fact1+" and "+explaining_fact2)
    return explanations


def english_exp_template(mapped_data, predictions, template_objs, entity_inverse_map, relation_inverse_map, entity_names, relation_names):
    explanations = []
    for fact, pred in zip(mapped_data, predictions):
        if(pred == 0):
            explanations.append(NO_EXPLANATION)
        else:
            explanations.append(template_objs[pred-1].get_english_explanation(
                fact, entity_inverse_map, relation_inverse_map, entity_names, relation_names))
    return explanations


def write_english_exps(named_data, template_exps, rule_exps, output_file,num_per_hit):
    csv_data = []
    for fact, t_exp, r_exp in zip(named_data, template_exps, rule_exps):
        if(t_exp == NO_EXPLANATION and r_exp == NO_EXPLANATION):
            continue
        row = [' '.join(fact), t_exp, r_exp]
        csv_data.append(row)
    
    reqd = int(len(csv_data)/num_per_hit)*num_per_hit
    csv_data = np.array(csv_data[:reqd])
    csv_data = csv_data.reshape((-1, 3*num_per_hit))
    columns = []
    for i in range(1, num_per_hit+1):
        columns.extend(['fact_'+str(i), 'exp_A_'+str(i), 'exp_B_'+str(i)])
    df = pd.DataFrame(csv_data, columns=columns)
    df.to_csv(output_file+".csv", index=False, sep=',')
    pd.set_option('display.max_colwidth', -1)
    df.to_html(output_file+".html", escape=False, justify='center')


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
    parser.add_argument('-tp', '--template_pred',
                        help='List of template predictions of data', default=None)
    parser.add_argument('-rp', '--rule_pred',
                        help='List of rules predicted for data', default=None)
    parser.add_argument('--data_repo_root',
                        required=False, default='data')
    parser.add_argument('--num',help='Number of samples for one HIT',default=5,type=int)
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

    if(len(data)%args.num != 0):
        logging.error('Number of examples per hit is not a factor of length of data')
        exit(-1)

    mapped_data = np.array(utils.map_data(
        data, distmult_dump['entity_to_id'], distmult_dump['relation_to_id'])).astype(np.int32)
    logging.info("Loaded test file from %s" % (args.test_file))

    if(args.template_pred is None and args.rule_pred is None):
        logging.error("Both template_pred and rule_pred cannot be None")

    if(args.template_pred is not None):
        template_predictions = np.loadtxt(args.template_pred).astype(np.int32)
        logging.info("Loaded test file predictions  %s" % (args.template_pred))
        if(len(template_predictions) != len(mapped_data)):
            logging.error("Unequal length of template predictions and data")
            exit(-1)

    if(args.rule_pred is not None):
        rule_predictions = utils.read_pkl(args.rule_pred)
        logging.info("Loaded rule predictions  from %s" % (args.rule_pred))
        if(len(rule_predictions) != len(mapped_data)):
            logging.error("Unequal length of rule predictions and data")
            exit(-1)

    entity_names = utils.read_entity_names(os.path.join(
        data_root, "mid2wikipedia_cleaned.tsv"),add_wiki=True)
    relation_names = utils.read_relation_names(
        os.path.join(data_root, "relation_name.txt"))

    entity_inverse_map = utils.get_inverse_dict(distmult_dump['entity_to_id'])
    relation_inverse_map = utils.get_inverse_dict(
        distmult_dump['relation_to_id'])

    template_objs = template_builder.template_obj_builder(
        data_root, args.model_weights, args.template_load_dir, None, "distmult", [1, 2, 3, 4, 5], True)

    utils.e1_e2_r, utils.e2_e1_r = utils.get_ent_ent_rel(template_objs[0].kb.facts) 

    if(args.template_pred is not None):
        template_exps = english_exp_template(mapped_data, template_predictions, template_objs,
                                             entity_inverse_map, relation_inverse_map, entity_names, relation_names)
    else:
        template_exps = [NO_EXPLANATION for _ in range(len(mapped_data))]

    if(args.rule_pred is not None):
        rule_exps = english_exp_rules(
            mapped_data, rule_predictions, entity_inverse_map, relation_inverse_map, entity_names, relation_names)
    else:
        rule_exps = [NO_EXPLANATION for _ in range(len(mapped_data))]

    logging.info("Generated explanations")

    if(len(rule_exps) != len(template_exps)):
        logging.error("Invalid length of explanations {} and {}".format(len(rule_exps),len(template_exps)))
        exit(-1)


    named_data = utils.map_data(data, entity_names, relation_names)
    write_english_exps(named_data, template_exps,
                       rule_exps, args.output_filename,args.num)
    logging.info("Written explanations to %s" % (args.output_filename))
