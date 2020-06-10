# This code is used to generate two csvs and a book file that is uploaded to mturk for project -
# TexKBC useful? (id=1419750).
# Essentially it generates a book file (which is a html file for easy viewing) and a _exps_hits.csv and _no_exps_hits.csv
# They contain the data for the facts, with explanation and without explanations respectively.
import argparse
import logging
import os
import pickle
import string
import time
import queue
import itertools

import numpy as np
import pandas as pd
import random
import template_builder
import utils
from explainer import Explainer


def english_exp_rules(mapped_data, predictions, explainer):

    explanations = []

    for itr in range(len(mapped_data)):
        fact = mapped_data[itr]
        pred = predictions[itr]
        if(pred[1] == -1):
            if(pred[0] == ""):
                explanations.append(explainer.NO_EXPLANATION)
            else:
                explaining_fact = explainer.html_fact(
                    [fact[0], pred[0], fact[2]])
                explanations.append(explaining_fact)
        else:
            if len(pred[0]) == 2:
                explaining_fact1 = explainer.html_fact(
                    [fact[0], pred[0][0], pred[1]])
                explaining_fact2 = explainer.html_fact(
                    [pred[1], pred[0][1], fact[2]])
                explanations.append(explaining_fact1+" and "+explaining_fact2)
            else:
                explaining_fact1 = explainer.html_fact(
                    [fact[0], pred[0][0], pred[1][0]])
                explaining_fact2 = explainer.html_fact(
                    [pred[1][0], pred[0][1], pred[1][1]])
                explaining_fact3 = explainer.html_fact(
                    [pred[1][1], pred[0][2], fact[2]])
                explanations.append(
                    explaining_fact1+" and "+explaining_fact2+" and "+explaining_fact3)
    return explanations


def english_exp_template(mapped_data, predictions, template_objs, explainer):
    explanations = []
    for fact, pred in zip(mapped_data, predictions):
        if(pred == 0):
            explanations.append(explainer.NO_EXPLANATION)
        else:
            explanations.append(
                template_objs[pred-1].get_english_explanation(fact, explainer))
    return explanations


def write_english_exps(mapped_data, exps, output_path, num_per_hit, explainer, y_labels):
    html_data = []
    total_exp_data = list(zip(mapped_data, exps, y_labels))
    random.shuffle(total_exp_data)
    for fact, exp, y in total_exp_data:
        if(exp == explainer.NO_EXPLANATION):
            continue
        htmled_fact = explainer.html_fact(fact)
        row = [htmled_fact, exp, y]
        html_data.append(row)

    df_html = pd.DataFrame(html_data, columns=['fact', 'explanation', 'true?'])
    pd.set_option('display.max_colwidth', -1)
    last_out_part = os.path.basename(os.path.normpath(output_path))
    with open(os.path.join(output_path, last_out_part+"_book.html"), 'w') as html_file:
        html_file.write(explainer.CSS_STYLE+'\n')
        df_html.to_html(html_file, escape=False, justify='center')

    logging.info('Total Facts = {}'.format(len(mapped_data)))
    logging.info('Final facts written = {}'.format(len(html_data)))

    csv_data_exps = [[x[0], x[1]] for x in html_data]
    csv_data_no_exps = [[x[0], explainer.NO_EXPLANATION] for x in html_data]

    columns = []
    for i in range(num_per_hit):
        columns.extend(['fact_'+str(i), 'exp_'+str(i)])

    print('Generated columns {}'.format(columns))
    print('Generated csv_data {} and {}'.format(
        len(csv_data_exps), len(csv_data_no_exps)))
    reqd = int(len(html_data)/num_per_hit)*num_per_hit

    csv_data_exps = np.array(csv_data_exps[:reqd])
    csv_data_exps = csv_data_exps.reshape((-1, len(columns)))
    df = pd.DataFrame(csv_data_exps, columns=columns)
    df.to_csv(os.path.join(output_path, last_out_part +
                           "_exps_hits.csv"), index=False, sep=',')

    csv_data_no_exps = np.array(csv_data_no_exps[:reqd])
    csv_data_no_exps = csv_data_no_exps.reshape((-1, len(columns)))
    df = pd.DataFrame(csv_data_no_exps, columns=columns)
    df.to_csv(os.path.join(output_path, last_out_part +
                           "_no_exps_hits.csv"), index=False, sep=',')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset', help="Name of the dataset as in data folder", required=True)
    parser.add_argument('-w', '--model_weights',
                        help="Pickle file of model wieghts", required=True)
    parser.add_argument('-l', '--template_load_dir',
                        required=False, default=None)
    parser.add_argument('-o', '--output_path',
                        required=True, default=None)
    parser.add_argument('-tf', '--test_file',
                        required=True, default=None)
    parser.add_argument('-tp', '--template_pred',
                        help='List of template predictions of data', default=None)
    parser.add_argument('-rp', '--rule_pred',
                        help='List of rules predicted for data', default=None)
    parser.add_argument('--data_repo_root',
                        required=False, default='data')
    parser.add_argument(
        '--num', help='Number of samples for one HIT', default=5, type=int)
    parser.add_argument('--log_level',
                        default='INFO',
                        dest='log_level',
                        type=utils._log_level_string_to_int,
                        nargs='?',
                        help='Set the logging output level. {0}'.format(utils._LOG_LEVEL_STRINGS))
    parser.add_argument(
        '-y', '--y_label', help="Optional y_label specifying if the fact is true or false", required=False, default=None)

    parser.add_argument('--t_ids', nargs='+', type=int, required=True,
                        help='List of templates to build objects for')
    
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s :: %(asctime)s - %(message)s',
                        level=args.log_level, datefmt='%d/%m/%Y %I:%M:%S %p')

    data_root = os.path.join(args.data_repo_root, args.dataset)

    distmult_dump = utils.read_pkl(args.model_weights)
    logging.info("Read Model Dump")

    data = utils.read_data(args.test_file)

    mapped_data = np.array(utils.map_data(
        data, distmult_dump['entity_to_id'], distmult_dump['relation_to_id'])).astype(np.int32)
    logging.info("Loaded test file from %s" % (args.test_file))

    if(args.template_pred is None and args.rule_pred is None):
        logging.error("Both template_pred and rule_pred cannot be None")
    elif(args.template_pred is not None and args.rule_pred is not None):
        logging.error("Both template_pred and rule_pred cannot be non None")

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

    entity_inverse_map = utils.get_inverse_dict(distmult_dump['entity_to_id'])
    relation_inverse_map = utils.get_inverse_dict(
        distmult_dump['relation_to_id'])

    template_objs = template_builder.template_obj_builder(
        data_root, args.model_weights, args.template_load_dir, None, "distmult", args.t_ids, True)

    explainer = Explainer(
        data_root, template_objs[0].kb, template_objs[0].base_model, entity_inverse_map, relation_inverse_map)

    if(args.template_pred is not None):
        exps = english_exp_template(
            mapped_data, template_predictions, template_objs, explainer)
    elif(args.rule_pred is not None):
        exps = english_exp_rules(mapped_data, rule_predictions, explainer)

    logging.info("Generated explanations")

    os.makedirs(args.output_path, exist_ok=True)

    y_labels = ['na' for _ in range(len(mapped_data))]
    if(args.y_label is not None):
        y_labels = np.loadtxt(args.y_label)
        if(len(y_labels) != len(mapped_data)):
            logging.error('Length of y_labels not same as len of mapped data')
            exit(-1)

    write_english_exps(mapped_data, exps, args.output_path,
                       args.num, explainer, y_labels)
    logging.info("Written explanations to %s" % (args.output_path))
