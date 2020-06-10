import os
import pandas as pd
import functools
import operator
import argparse
import sys

parser = argparse.ArgumentParser()

parser.add_argument('--fdir',required=True, type=str)
parser.add_argument('--t6',required=False, default = 0,  type=int)
parser.add_argument('--outfile',required=False, default = 'faithfulness_plots.csv',  type=str)
args = parser.parse_args(sys.argv[1:])

#fdir = '../logs/faithfulness_data/'
fdir = args.fdir

flist = os.listdir(fdir)
df = None
for f in flist:
    print(f)
    d = pd.read_csv(os.path.join(fdir,f),header='infer')
    if df is None:
        df = d
    else:
        df = df.append(d)


df = df.reset_index(drop=True)

#iter no, sro, prob, rank, sr'o, prob, rank,

df['zmscore'] = df['zmetric']


if args.t6 == 0:
    df['sro'] = df['score_point_0']
    df['sr1o'] = df['score_sample_0']
    df['zrank'] = df['probability_point_0']
    df['zmrr'] = 1.0/df['zrank']

    random_r = ['score_point_{}'.format(x) for x in range(1,10) if 'score_point_{}'.format(x) in df.columns] 
    random_r = functools.reduce(operator.add, [df[x] for x in random_r])/len(random_r)
    
    random_r1 = ['score_sample_{}'.format(x) for x in range(1,10) if 'score_sample_{}'.format(x) in df.columns] 
    random_r1 = functools.reduce(operator.add, [df[x] for x in random_r1])/len(random_r1)

    random_rank = ['probability_point_{}'.format(x) for x in range(1,10) if 'probability_point_{}'.format(x) in df.columns] 
    random_mrr = functools.reduce(operator.add, [1.0/df[x] for x in random_rank])/len(random_rank)
    random_rank = functools.reduce(operator.add, [df[x] for x in random_rank])/len(random_rank)
    
    df['random_r'] = random_r
    df['random_r1'] = random_r1
    df['zrandom_rank'] = random_rank
    df['zrandom_mrr'] = random_mrr
    dfpivot  = df.pivot_table(index=['iteration'],values=['sro','sr1o','random_r','random_r1','zrank','zrandom_rank','zmrr','zrandom_mrr','zmscore'],aggfunc=['mean','count'])
    dfpivot.to_csv(args.outfile)

if args.t6:
    df['sro'] = df['score_point_0']
    df['sr1u'] = df['score_sample_1_0']
    df['ur2o'] = df['score_sample_2_0']
    df['zrank'] = df['probability_point_0']
    df['zmrr'] = 1.0/df['zrank']

    
    random_r = ['score_point_{}'.format(x) for x in range(1,10) if 'score_point_{}'.format(x) in df.columns] 
    random_r = functools.reduce(operator.add, [df[x] for x in random_r])/len(random_r)
    
    random_r1 = ['score_sample_1_{}'.format(x) for x in range(1,10) if 'score_sample_1_{}'.format(x) in df.columns] 
    random_r1 = functools.reduce(operator.add, [df[x] for x in random_r1])/len(random_r1)

    random_r2 = ['score_sample_2_{}'.format(x) for x in range(1,10) if 'score_sample_2_{}'.format(x) in df.columns] 
    random_r2 = functools.reduce(operator.add, [df[x] for x in random_r2])/len(random_r2)
    
    random_rank = ['probability_point_{}'.format(x) for x in range(1,10) if 'probability_point_{}'.format(x) in df.columns] 
    random_mrr  = functools.reduce(operator.add, [1.0/df[x] for x in random_rank])/len(random_rank)
    random_rank = functools.reduce(operator.add, [df[x] for x in random_rank])/len(random_rank)
    
    
    df['random_r'] = random_r
    df['random_r1'] = random_r1
    df['random_r2'] = random_r2
    df['zrandom_rank'] = random_rank
    df['zrandom_mrr'] = random_mrr
    dfpivot  = df.pivot_table(index=['iteration'],values=['sro','sr1u','ur2o','random_r','random_r1','random_r2','zrank','zrandom_rank','zmrr','zrandom_mrr','zmscore'],aggfunc=['mean','count'])
    dfpivot.to_csv(args.outfile)

