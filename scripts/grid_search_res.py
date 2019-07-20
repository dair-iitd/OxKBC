# This script is used to compile the results from a grid search into a nice CSV
# NOTE: This does not refer to cross validation grid search. For that see @cross_val_grid_search_res.py

import os
import re
import pandas as pd
import pprint
import sys

p = re.compile('Best score: (0.[0-9]*)')


def get_best(filename):
    ans = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        matches = p.findall(line)
        if len(matches) == 1:
            ans = max(ans, float(matches[0]))
    print("From "+filename+" getting ans: ", ans)
    return ans


def parse_folder(dirname):
    params = os.path.basename(os.path.normpath(dirname)).split('_')
    return (float(params[1]), float(params[2]))


def write_table(result, write_path):
    neg_list = sorted(list(result.keys()))
    print(neg_list)
    rho_list = sorted(list(result[neg_list[0]].keys()))
    columns = ['rho/neg']
    mydata = {'rho/neg': []}
    for x in rho_list:
        mydata['rho/neg'].append(str(x))
    for x in neg_list:
        mydata[str(x)] = []
        columns.append(str(x))

    for neg in neg_list:
        for rho in rho_list:
            mydata[str(neg)].append(result[neg][rho])

    df = pd.DataFrame(mydata, columns=columns)
    df.to_csv(write_path, sep='\t', index=False)


if len(sys.argv) != 2:
    print('Usage is '+sys.argv[0] + ' <folder_path>')
    exit(-1)

path = ''+sys.argv[1]
result = {}
for (dirpath, dirnames, filenames) in os.walk(path):
    for filename in filenames:
        if filename.endswith('log.txt'):
            neg, rho = parse_folder(dirpath)
            if neg not in result:
                result[neg] = {}
            result[neg][rho] = get_best(os.path.join(dirpath, filename))

write_table(result, os.path.join(path, 'results.csv'))
