"""
This script is a modified version of cross_val_grid_search.py (21/08/2019)
To support the updated: <YATIN_SCRIPT_FOR_TRAINING>
New Directory structure assumed:
	cross_val_base:
				fb15k:
					supervision_type:	(assumed this point is given in input)
									run1:
										exp_NR_rho_KL_arch:
														exp_1(for folds)
This generates a similar summary.csv for every (KL, arch) pair
summary.csv: tables for rho, NR for each DATA_PREFIX and AGG_METRIC
"""
import os
import pandas as pd
import pprint
import pickle
import numpy as np
from sklearn import metrics
import argparse
import logging
import itertools

DATA_PREFIX = ['train', 'val', 'test']
AGG_METRIC = ['mean', 'std', 'min', 'max']


def load_data(base_dir, folds):
    data = {'train': [], 'val': [], 'test': []}
    for i in range(folds):
        for el in DATA_PREFIX:
            ys = pickle.load(
                open(os.path.join(base_dir, el+'_'+str(i)+".pkl"), 'rb'))[:, -1]
            data[el].append(ys)
    return data


def check_data(base_dir, folds):
    for i in range(folds):
        for el in DATA_PREFIX:
            if not os.path.isfile(os.path.join(base_dir, el+'_'+str(i)+".pkl")):
                logging.error("File not present at {}".format(
                    os.path.join(base_dir, el+'_'+str(i)+".pkl")))
                return False
    return True


def get_params(directory):
    exps = os.listdir(directory)
    neg_rewards = set()
    rhos = set()
    
    kls = set()
    archs = set()

    for exp in exps:
        if os.path.isdir(os.path.join(directory, exp)):
        	exp_split = exp.split('_')
            neg_r, rho, kl, arch = exp_split[1], exp_split[2], exp_split[3], exp_split[4]
            neg_rewards.add(neg_r)
            rhos.add(rho)
            kls.add(kl)
            archs.add(arch)
    return (list(sorted(neg_rewards, key=float)), list(sorted(rhos, key=float)), list(sorted(kls, key=float)), list(sorted(archs, key=float)))


def check_exp(directory, data, folds):
    for i in range(folds):
        for el in DATA_PREFIX[1:]:
            if el == 'val':
                preds_fname = os.path.join(
                    directory, 'exp_'+str(i), 'valid_preds.txt')
            else:
                preds_fname = os.path.join(
                    directory, 'exp_'+str(i), 'test_preds.txt')
            if os.path.isfile(preds_fname):
                pred_data = np.loadtxt(preds_fname).tolist()
                if len(data[el][i]) != len(pred_data):
                    logging.error(
                        "Length of {} does not match with ground truth data".format(preds_fname))
                    return False
            else:
                logging.error("File {} does not exist".format(preds_fname))
                return False
    return True


def begin_checks(base_dir, folds, runs):
    data_present = check_data(base_dir, folds)
    if not data_present:
        logging.error("Data Check failed")
        return (False, [])
    data = load_data(base_dir, folds)
    logging.info("Data Check Passed")

    cols, rows, kls, archs = get_params(os.path.join(base_dir, "run_1"))
    for run in range(2, runs+1):
        this_cols, this_rows, this_kls, this_archs = get_params(
            os.path.join(base_dir, "run_"+str(run)))
        if not (set(cols) == set(this_cols) and set(rows) == set(this_rows) and set(kls)==set(this_kls) and set(archs)==set(this_archs)):
            logging.error(
                "The parameters in the folder run_1 and run_"+str(run)+" are different")
            return (False, [])
    logging.info(
        "Param Check Successfull\nRows are {}\ncolumns are {}".format(rows, cols))

    invalid_exps = []
    for run in range(1, runs+1):
        for col in cols:
            for row in rows:
            	for kl in kls:
            		for arch in archs:
		                directory = os.path.join(
		                    base_dir, "run_"+str(run), "exp_"+col+"_"+row+"_"+kl+"_"+arch)
		                if not check_exp(directory, data, folds):
		                    invalid_exps.append(directory)

    if len(invalid_exps) == 0:
        logging.info("All experiments are ok")
        return (True, [])
    else:
        logging.error("Following experiments have error\n{}".format(
            str(pprint.pformat(invalid_exps))))
        logging.error("Length is {}".format(len(invalid_exps)))
        return (False, invalid_exps)


def write_invalid(invalid_exps, fname):
    if fname is not None:
        with open(fname, 'w') as f:
            for exp in invalid_exps:
                f.write('/'.join(exp.split('/')[-2:])+'\n')
        logging.error("Written invalid experiments to {}".format(fname))


def calc_exp(directory, data, folds):
    predictions = {'val': [], 'test': []}
    true = {'val': [], 'test': []}
    logging.info("Calculating results for experiment {}".format(directory))
    for i in range(folds):
        for el in DATA_PREFIX[1:]:
            if el == 'val':
                preds_fname = os.path.join(
                    directory, 'exp_'+str(i), 'valid_preds.txt')
            else:
                preds_fname = os.path.join(
                    directory, 'exp_'+str(i), 'test_preds.txt')
            pred_data = np.loadtxt(preds_fname).tolist()
            predictions[el].extend(pred_data)
            true[el].extend(data[el][i])
    mif_val = metrics.f1_score(true['val'], predictions['val'], labels=[
                               1, 2, 3, 4, 5], average='micro')
    mif_test = metrics.f1_score(true['test'], predictions['test'], labels=[
                                1, 2, 3, 4, 5], average='micro')
    return (mif_val, mif_test)


def calc_run_results(directory, rows, cols, KLs, Archs, data, folds):
    results_one_run = {'neg_re': [], 'rho': [], 'val': [], 'test': [], 'kl':[], 'arch':[]}
    for col in cols:
        for row in rows:
        	for kl in KLs:
        		for arch in Archs:
		            exp_directory = os.path.join(directory, "exp_"+col+"_"+row+"_"+kl+"_"+arch)
		            val, test = calc_exp(exp_directory, data, folds)
		            results_one_run['neg_re'].append(col)
		            results_one_run['rho'].append(row)
		            results_one_run['val'].append(val)
		            results_one_run['test'].append(test)
		            results_one_run['kl'].append(kl)
		            results_one_run['arch'].append(arch)
    df = pd.DataFrame(data=results_one_run)
    return df


def write_to_file(table, header, f):
    f.write(header+'\n')
    table.to_csv(f, float_format='%.4f')
    f.write('\n')


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s :: %(asctime)s - %(message)s',
                        level=logging.INFO, datefmt='%d/%m/%Y %I:%M:%S %p')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dir', help="Path of the base directory", required=True)
    parser.add_argument('--folds', type=int, default=5, required=True)
    parser.add_argument('--runs', type=int, default=5, required=True)
    parser.add_argument(
        '--ifile', help='File to write the experiment names which failed', type=str, default=None)
    args = parser.parse_args()

    ok, invalid_exps = begin_checks(args.dir, args.folds, args.runs)

    if not ok and len(invalid_exps) > 0:
        write_invalid(invalid_exps, args.ifile)
        exit(0)

    logging.info("All Checks passed")
    cols, rows, KLs, Archs = get_params(os.path.join(args.dir, "run_1"))
    data = load_data(args.dir, args.folds)
    all_run_results = []
    for run in range(1, args.runs+1):
        logging.info("Calculating results for run {}".format(run))
        df = calc_run_results(os.path.join(
            args.dir, "run_"+str(run)), rows, cols, KLs, Archs data, args.folds)
        all_run_results.append(df)

    final_result = pd.concat(all_run_results)
    agg_table = final_result.groupby(['neg_re', 'rho', 'kl', 'arch']).agg(AGG_METRIC)
    agg_table.columns = [d+"_"+a for d,
                         a in itertools.product(DATA_PREFIX[1:], AGG_METRIC)]
    agg_table = agg_table.reset_index()

    fh = open(os.path.join(args.dir, 'summary.csv'), 'w')

    for d, a, kl, arch in itertools.product(DATA_PREFIX[1:], AGG_METRIC, KLs, Archs):
        tbl = agg_table[agg_table["kl"]==kl & agg_table["arch"]==arch].pivot_table(
            columns='neg_re', index='rho', values=d+'_'+a, fill_value=-1)
        print(tbl)
        write_to_file(tbl, (d+'_'+a+'_'+kl+'_'+arch).upper(), fh)

    logging.info("Written results to {}".format(
        os.path.join(args.dir, 'summary.csv')))
