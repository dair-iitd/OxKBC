import pandas as pd
orig = pd.read_csv('orig.csv')
fname_last = 7
to_add = []
for i in range(1,fname_last+1):
        to_add.extend(pd.read_csv(str(str(i)+'.csv'))['WorkerId'])
to_add = set(to_add)
print(len(to_add))
for w in to_add:
        orig.loc[orig['Worker ID'] == w, 'UPDATE-kbc_time_no_repeat'] = '100'
orig.to_csv('new.csv',sep=',',index=False)
