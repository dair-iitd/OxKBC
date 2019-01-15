import numpy as np
import sys,os
import pandas as pd

# base_dir = '../sm/cross_val/'
# dataset_name = 'cross_val_yago_90_40/'
dataset_dir=sys.argv[1]
test_or_val = 'test'
# exps = os.listdir(os.path.join(base_dir,dataset_name))
exps = os.listdir(dataset_dir)

flist = []
for exp in exps:
    if os.path.isdir(os.path.join(dataset_dir, exp)):
        fname = os.path.join(dataset_dir,exp,test_or_val)
        if os.path.exists(fname):
            flist.append(fname)
        else:
            print('@@@@@@@@@@@ ', fname, 'does not exists @@@@@@@@@')



def merge_tables(flist,header= None):
    table = None 
    for f in flist:
        this_table = pd.read_csv(f,sep=',',header = header)
        this_table['fname'] = f.split('/')[-2]
        if table is None:
            table = this_table
        else:
            table = table.append(this_table)
    #
    return table

header = 'epoch,mode,loss,count,dataset_size,time,p1,r1,f1,s1,p2,r2,f2,s2,p3,r3,f3,s3,p4,r4,f4,s4,p5,r5,f5,s5,acc,mip,mir,mif'.split(',')
header.append('exp')
table = merge_tables(flist)
table.columns = header 
s =table['exp'].apply(lambda x: x.split('_'))
table['neg_rew'] = s.apply(lambda x: x[1])
table['rho'] = s.apply(lambda x: x[2])

summ = table.groupby(['neg_rew','rho']).agg({'mif': ['mean','std','min','max'],'epoch':'count'})
summ.columns = ['mif_mean','mif_std','mif_min','mif_max','count']
summ = summ.reset_index()

a1 =summ.pivot_table(columns = 'neg_rew',index='rho', values = 'mif_mean', fill_value = 0)
a2 = summ.pivot_table(columns = 'neg_rew',index='rho', values = 'mif_std', fill_value = 0)
a3 = summ.pivot_table(columns = 'neg_rew',index='rho', values = 'mif_min', fill_value = 0)
a4 = summ.pivot_table(columns = 'neg_rew',index='rho', values = 'mif_max', fill_value = 0)
a5 = summ.pivot_table(columns = 'neg_rew',index='rho', values = 'count', fill_value = 0)


fh = open(os.path.join(dataset_dir,test_or_val+'_summary.csv'),'w')


def write_to_file(table,header,f):
    print(header,file =f)
    table.to_csv(f, float_format='%.4f')
    print('',file=f)

write_to_file(a1,'MEAN',fh)
write_to_file(a2,'STD',fh)
write_to_file(a3,'MIN',fh)
write_to_file(a4,'MAX',fh)
write_to_file(a5,'COUNT',fh)

fh.close()



