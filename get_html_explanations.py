# This code is used to generate a csv and a book file that is uploaded to mturk for project -
# Explainable KBC (id=1365457).
# Essentially it generates a book file (which is a html file for easy viewing) and a hits.csv which is to be uploaded while creating a batch.
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


def get_options(iter_id, our_is_A, one_is_no):
    iter_id = iter_id % 5
    opt1_id = ('our_' if our_is_A else 'other_') + str(iter_id)
    opt2_id = ('other_' if our_is_A else 'our_') + str(iter_id)
    opt3_id = 'both_' + str(iter_id)
    opt4_id = 'none_' + str(iter_id)

    opt1_str = '<crowd-radio-button name=\"' + opt1_id + \
        '\"> A is better than B </crowd-radio-button>'
    opt2_str = '<crowd-radio-button name=\"' + opt2_id + \
        '\"> B is better than A </crowd-radio-button>'
    opt3_str = '<crowd-radio-button name=\"' + opt3_id + \
        '\"> Both A and B are equally good </crowd-radio-button>'
    opt4_str = '<crowd-radio-button name=\"' + opt4_id + \
        '\"> Both A and B are bad </crowd-radio-button>'

    if one_is_no:
        opt3_str = ''
        opt4_str = ''

    return [opt1_str, opt2_str, opt3_str, opt4_str]


def write_english_exps(mapped_data, template_exps, rule_exps, output_path, num_per_hit, explainer):
    raw_data = queue.Queue(0)
    both_no = 0
    both_same = 0
    qlty_ctrl = queue.Queue(0)
    total_exp_data = list(zip(mapped_data, template_exps, rule_exps))
    random.shuffle(total_exp_data)
    for fact, t_exp, r_exp in total_exp_data:
        if(t_exp == explainer.NO_EXPLANATION and r_exp == explainer.NO_EXPLANATION):
            both_no += 1
            continue
        htmled_fact = explainer.html_fact(fact)
        row = [htmled_fact, t_exp, r_exp]
        if(t_exp == r_exp):
            both_same += 1
            qlty_ctrl.put(row)
            continue
        raw_data.put(row)

    html_data = []
    while(not raw_data.empty()):
        if(len(html_data) % 5 == 4 and not qlty_ctrl.empty()):
            html_data.append(qlty_ctrl.get())
            qlty_ctrl.put(html_data[-1]) ## Need this so that every HIT has a quality control fact.
        else:
            html_data.append(raw_data.get())

    html_data_chunked = list(utils.chunks(html_data, 5))
    _ = [random.shuffle(el) for el in html_data_chunked]
    html_data = list(itertools.chain(*html_data_chunked))

    df_html = pd.DataFrame(html_data, columns=['fact', 'our', 'other'])
    pd.set_option('display.max_colwidth', -1)
    last_out_part = os.path.basename(os.path.normpath(output_path))
    with open(os.path.join(output_path, last_out_part+"_book.html"), 'w') as html_file:
        html_file.write(explainer.CSS_STYLE+'\n')
        df_html.to_html(html_file, escape=False, justify='center')

    logging.info('Total Facts = {}'.format(len(mapped_data)))
    logging.info('Both No explanations = {}'.format(both_no))
    logging.info('Both Same explanations = {}'.format(both_same))

    csv_data = []
    iter_id = 0
    for htmled_fact, t_exp, r_exp in html_data:
        r = random.uniform(0, 1)
        row = [htmled_fact]
        one_is_no = (t_exp == explainer.NO_EXPLANATION or r_exp ==
                     explainer.NO_EXPLANATION)
        if(r <= 0.5):
            row.extend([t_exp, r_exp])
            row.extend(get_options(iter_id, True, one_is_no))
        else:
            row.extend([r_exp, t_exp])
            row.extend(get_options(iter_id, False, one_is_no))
        csv_data.append(row)
        iter_id += 1
    columns = []
    for i in range(num_per_hit):
        columns.extend(['fact_'+str(i), 'exp_A_'+str(i), 'exp_B_'+str(i)])
        for j in range(4):
            columns.append('opt_'+str(i)+'_'+str(j))
    print('Generated columns {}'.format(columns))
    print('Generated csv_data {}'.format(len(csv_data)))
    reqd = int(len(html_data)/num_per_hit)*num_per_hit
    csv_data = np.array(csv_data[:reqd])
    csv_data = csv_data.reshape((-1, len(columns)))
    df = pd.DataFrame(csv_data, columns=columns)
    df.to_csv(os.path.join(output_path, last_out_part +
                           "_hits.csv"), index=False, sep=',')


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
        data_root, args.model_weights, args.template_load_dir, None, "distmult", [1, 2, 3, 4, 5, 6], True)

    explainer = Explainer(
        data_root, template_objs[0].kb, template_objs[0].base_model, entity_inverse_map, relation_inverse_map)

    if(args.template_pred is not None):
        template_exps = english_exp_template(
            mapped_data, template_predictions, template_objs, explainer)
    else:
        template_exps = [
            explainer.NO_EXPLANATION for _ in range(len(mapped_data))]

    if(args.rule_pred is not None):
        rule_exps = english_exp_rules(mapped_data, rule_predictions, explainer)
    else:
        rule_exps = [explainer.NO_EXPLANATION for _ in range(len(mapped_data))]

    logging.info("Generated explanations")

    if(len(rule_exps) != len(template_exps)):
        logging.error("Invalid length of explanations {} and {}".format(
            len(rule_exps), len(template_exps)))
        exit(-1)

    os.makedirs(args.output_path, exist_ok=True)
    write_english_exps(mapped_data, template_exps, rule_exps,
                       args.output_path, args.num, explainer)
    logging.info("Written explanations to %s" % (args.output_path))
