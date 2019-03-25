import pandas as pd
import numpy as npi

NO_EXPLANATION = "No explanation for this fact"

true = pd.read_csv('true/true_hits.csv',sep=',')
pred = pd.read_csv('pred/pred_hits.csv',sep=',')

true_data = true.filter(['fact_0','exp_A_0'])
pred_data = pred.filter(['fact_0','exp_A_0'])

true_data.columns = ['true_fact', 'true_exp']
pred_data.columns = ['pred_fact', 'pred_exp']

data = pd.concat([true_data,pred_data],axis=1)

final_data = data[ (data['true_exp'] != NO_EXPLANATION) | (data['pred_exp'] != NO_EXPLANATION) ]
print('Final Data is {}'.format(final_data.shape))
f = open('true_pred_book.html','w')
css = open('../../../css_style.css','r').read()
f.write(css+'\n\n')
pd.set_option('display.max_colwidth', -1)
final_data.to_html(f, escape=False, justify='center')
f.close()

