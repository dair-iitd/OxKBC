import os
import re
import pandas as pd
import pprint
import sys
import pickle
import numpy as np
from sklearn import metrics
import argparse
import logging

DATA_PREFIX = ['train','val','test']

def load_data(base_dir,folds):
    data = {'train':[],'val':[],'test':[]}
    for i in range(folds):
        for el in DATA_PREFIX:
            ys = pickle.load(open(os.path.join(base_dir,el+'_'+str(i)+".pkl"),'rb'))[:,-1]
            data[el].append(ys)
    return data

def check_data(base_dir,folds):
    for i in range(folds):
        for el in DATA_PREFIX:
            if not os.path.isfile(os.path.join(base_dir,el+'_'+str(i)+".pkl")):
                logging.error("File not present at {}".format(os.path.join(base_dir,el+'_'+str(i)+".pkl")))
                return False
    return True

def get_params(directory):
    exps = os.listdir(directory)
    neg_rewards = set()
    rhos = set()
    for exp in exps:
        if os.path.isdir(os.path.join(directory,exp)):
            neg_r,rho = exp.split('_')[1],exp.split('_')[2]
            neg_rewards.add(neg_r)
            rhos.add(rho)
    return (list(sorted(neg_rewards)),list(sorted(rhos)))

def check_exp(directory,data,folds):
    for i in range(folds):
        for el in DATA_PREFIX[1:]:
            if el == 'val':
                preds_fname = os.path.join(directory,'exp_'+str(i),'valid_preds.txt')
            else:
                preds_fname = os.path.join(directory,'exp_'+str(i),'test_preds.txt')                
            if os.path.isfile(preds_fname):
                pred_data = np.loadtxt(preds_fname).tolist()
                if len(data[el][i])  != len(pred_data):
                    logging.error("Length of {} does not match with ground truth data".format(preds_fname))
                    return False
            else:
                logging.error("File {} does not exist".format(preds_fname))
                return False
    return True

def begin_checks(base_dir,folds,runs):
    data_present = check_data(base_dir,folds)
    if not data_present:
        logging.error("Data Check failed")
        return (False,[])

    data = load_data(base_dir,folds)
    
    cols,rows = get_params(os.path.join(base_dir,"run_1"))
    print("Rows are {}\ncolumns are {}".format(rows,cols))
    for run in range(2,runs+1):
        this_cols, this_rows = get_params(os.path.join(base_dir,"run_"+str(run)))
        if not (set(cols) == set(this_cols) and set(rows) == set(this_rows)):
            logging.error("The parameters in the folder run_1 and run_"+str(run)+" are different")
            return (False,[])

    invalid_exps = []
    for run in range(1,runs+1):
        for col in cols:
            for row in rows:
                directory = os.path.join(base_dir,"run_"+str(run),"exp_"+col+"_"+row)
                if not check_exp(directory,data,folds):
                    invalid_exps.append(directory)
    
    if len(invalid_exps) == 0:
        return (True,[])
    else:
        logging.error("Following experiments have error\n{}".format(str(pprint.pformat(invalid_exps))))
        logging.error("Length is {}".format(len(invalid_exps)))
        return (False,invalid_exps)

def write_invalid(invalid_exps,fname):
    if fname is not None:
        with open(fname,'w') as f:
            for exp in invalid_exps:
                f.write('/'.join(exp.split('/')[-2:])+'\n')

def calc_exp(directory,data,folds):
    predictions = {'val':[],'test':[]}
    true = {'val':[],'test':[]}
    for i in range(folds):
        for el in DATA_PREFIX[1:]:
            if el == 'val':
                preds_fname = os.path.join(directory,'exp_'+str(i),'valid_preds.txt')
            else:
                preds_fname = os.path.join(directory,'exp_'+str(i),'test_preds.txt')                
            pred_data = np.loadtxt(preds_fname).tolist()
            predictions[el].extend(pred_data)
            true[el].extend(data[el][i])
    mif_val = metrics.f1_score(true['val'],predictions['val'],labels=[1,2,3,4,5],average='micro')
    mif_test = metrics.f1_score(true['test'],predictions['test'],labels=[1,2,3,4,5],average='micro')
    return (mif_val,mif_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', help="Path of the base directory", required=True)
    parser.add_argument('--folds', type=int,default=5,required=True)
    parser.add_argument('--runs', type=int,default=5,required=True)
    parser.add_argument('--ifile', help='File to write the experiment names which failed',type=str,default=None)

    args = parser.parse_args()

    ok,invalid_exps = begin_checks(args.dir,args.folds,args.runs)

    if not ok and len(invalid_exps) > 0:
        write_invalid(invalid_exps,args.ifile)
    

# def parse_folder(dirname):
#     params = os.path.basename(os.path.normpath(dirname)).split('_')
#     return (float(params[1]),float(params[2]))

# def write_table(result,write_path):
#     neg_list = sorted(list(result.keys()))
#     print(neg_list)
#     rho_list = sorted(list(result[neg_list[0]].keys()))
#     columns = ['rho/neg']
#     mydata = {'rho/neg':[]}
#     for x in rho_list:
#         mydata['rho/neg'].append(str(x))
#     for x in neg_list:
#         mydata[str(x)] = []
#         columns.append(str(x))
    
#     for neg in neg_list:
#         for rho in rho_list:
#             mydata[str(neg)].append(result[neg][rho])

#     df = pd.DataFrame(mydata, columns=columns)
#     df.to_csv(write_path,sep='\t',index=False)







# base_folder_path = ''+sys.argv[1]
# result = {}
# for (dirpath, dirnames, filenames) in os.walk(path):
#     #for filename in filenames:
#     for dirname in dirnames:
#         print(dirpath)
#         print(dirname)
#         #if filename.endswith('log.txt'):
#         if len(dirname.split('_')) == 3:
#             neg,rho = parse_folder(dirname)
#             if neg not in result:
#                 result[neg] = {}
#             #result[neg][rho] = get_best(os.path.join(dirpath,filename))
#             result[neg][rho] = cross_get_best(os.path.join(dirpath,dirname))

# write_table(result,os.path.join(path,'results.csv'))


