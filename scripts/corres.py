import pandas as pd
import numpy as np
# inv = {}
# orig = {}
# for i in range(1,6):                                          
#     df = pd.read_csv('logs/exp_words_inv_valid/'+str(i)+'.txt',sep=';',error_bad_lines=False)
#     inv[i] = df                               
#     df1 = pd.read_csv('logs/exp_words_valid/'+str(i)+'.txt',sep=';',error_bad_lines=False)
#     orig[i] = df1

# for tid in range(3,4):
#     mystr = ''
#     orig_data = orig[tid].iloc[:,[0,1,2]].values
#     inv_data = inv[tid].iloc[:,[0,1,2]].values                                                 
#     for i in range(110):                                                                       
#         for j in range(len(inv_data)):            
#             if(np.all(inv_data[j]==orig_data[i])):                                          
#                     mystr += ' ; '.join(map(str,inv[tid].iloc[j,:])) +'\n'
#                     break                        
#     file=open('t.txt','w')                                              
#     file.write(mystr)    
#     file.close()


orig = pd.read_csv('../logs/yago/exp_words/merged.txt',sep='\t')
inv = pd.read_csv('../logs/yago-inv/exp_words/merged.txt',sep='\t')

inv_corr = pd.merge(orig,inv,how='left',on=['e1','r','e2'])

inv_corr_new = inv_corr.drop([inv_corr.columns[i] for i in range(3,31)],axis=1)
inv_corr_new.to_csv('merged_corr.csv',sep='\t',index=False)
