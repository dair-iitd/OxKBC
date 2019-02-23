import pandas as pd
import numpy as np
import pprint
import argparse
import collections
import string
import os
import bs4 as bs

ANSWER_OPTIONS = ['our','other','both','none']

def get_key_answer(key,id):
    return string.Template('Answer.${key}_${id}.on').substitute(key=key,id=id)

def get_key_input(key,id):
    return string.Template('Input.${key}_${id}').substitute(key=key,id=id)

def valid_row(row):
    total_sum = 0
    for i in range(5):
        if(row[get_key_input('exp_A',i)] == row[get_key_input('exp_B',i)]):
            quality_ctrl_id = i
        for opt in ANSWER_OPTIONS:
            total_sum += row[get_key_answer(opt,i)]
    if(total_sum != 5):
        return 'You did not mark any option in some questions'
    if(not (row[get_key_answer('both',quality_ctrl_id)] or row[get_key_answer('none',quality_ctrl_id)]) ):
        return 'You did not chose the option both explanations are good/bad, even when both A and B were same'
    return ''
    
def get_invalid_hits(df,outfilename):
    df_new = df.copy()
    df = df.fillna(False)
    invalid_hits = []
    for index,row in df.iterrows():
        message = valid_row(row) 
        if(message!=''):
            df_new['Reject'][index] = message
            invalid_hits.append(row['AssignmentId'])
    if(len(invalid_hits)!=0):
        df_new.to_csv(outfilename,index=False,sep=',')
    return invalid_hits

def get_book(book_filename):
    # TODO: Change this to have a clean pipeline
    with open(book_filename,'r') as f:
        soup = bs.BeautifulSoup(f, 'lxml') 
        table = soup.find('table')
        table_body = table.find('tbody')
        rows = table_body.find_all('tr')
        data = []
        for row in rows:                         
            cols = row.find_all('td')
            cols = [ele.text for ele in cols]
            data.append([ele for ele in cols if ele])
    return pd.DataFrame(data,columns=['fact','our','other'])

def get_winner(answers):
    our_ct = 0
    other_ct = 0
    for el in answers:
        if(el=='our'):
            our_ct += 1
        elif(el=='other'):
            other_ct += 1
        elif(el=='both'):
            our_ct += 1
            other_ct += 1

    if(our_ct > other_ct):
        return ['our']
    elif (other_ct > our_ct):
        return ['other']
    else:
        return []
        # return ['our','other']

def get_results(df,book):
    df = df.fillna(False)
    results = {}
    for index, row in df.iterrows():
        for i in range(5):
            fact = row[get_key_input('fact',i)]
            exp_A = row[get_key_input('exp_A',i)]
            exp_B = row[get_key_input('exp_B',i)]
            if(exp_A == exp_B):
                continue
            
            fact_text = bs.BeautifulSoup(fact,'lxml').text
            exp_B_text = bs.BeautifulSoup(exp_B,'lxml').text
            if(book[book.fact == fact_text]['our'].iloc[0] == exp_B_text):
                exp_A, exp_B = exp_B , exp_A

            if(fact not in results):
                results[fact] = {'our_exp': exp_A, 'other_exp':exp_B, 'answers' : [],'row_idx':[], 'fact_no':[]}
            
            for opt in ANSWER_OPTIONS:
                if(row[get_key_answer(opt,i)]):
                    results[fact]['answers'].append(opt)
                    results[fact]['row_idx'].append(index)
                    results[fact]['fact_no'].append(i)
    for k in results:
        winner = get_winner(results[k]['answers'])
        results[k]['winner'] = winner
    
    return results

def write_results(results,output_file,analysis_str):
    results_df = pd.DataFrame.from_dict(results,orient='index')
    results_df = results_df.reset_index()
    results_df = results_df.drop(['row_idx','fact_no'],axis=1)
    with open('css_style.css','r') as css_file:
        CSS = css_file.read()
    with open(output_file,'w') as f:
        f.write(CSS+'\n\n')
        analysis_str = analysis_str.replace('\n','<br><br>')
        f.write(analysis_str+'\n\n')
        pd.set_option('display.max_colwidth', -1)
        results_df.to_html(f, escape=False, justify='center')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-rf', '--result_file', help="Name of the result csv downloaded from mturk", required=True)
    parser.add_argument('-op', '--output_path', help="Output path for rejected people and results", required=True)
    parser.add_argument('-bf', '--book_file', help="Original HTML (Book) written by get_turk_data", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.result_file)
    df = df[df['AssignmentStatus'] != 'Rejected']
    res_file_last_part = os.path.basename(os.path.normpath(args.result_file)).split('.')[0]
    invalid_hits = get_invalid_hits(df,os.path.join(args.output_path,res_file_last_part+'_rejected.csv'))
    if(len(invalid_hits)!=0):
        print('There are {} invalid assignments which have id \n{}'.format(len(invalid_hits),invalid_hits))
        exit(-1)

    book = get_book(args.book_file)
    results = get_results(df,book)

    answers_list = []
    winner_list = []
    for k in results:
        answers_list.extend(results[k]['answers'])
        winner_list.extend(results[k]['winner'])

    ctr_answers = collections.Counter(answers_list)
    analysis_str = ''
    analysis_str += 'Total number of annotations = {}\n'.format(len(answers_list))
    for el in ctr_answers:
        ctr_answers[el] /= len(answers_list)*0.01
    analysis_str += '{}\n\n'.format(ctr_answers)

    ctr_winner = collections.Counter(winner_list)
    analysis_str += ('Total number of facts = {}\n'.format(len(results)))
    analysis_str += ('Total number of winning facts = {}\n'.format(len(winner_list)))
    for el in ctr_winner:
        ctr_winner[el] /= len(winner_list)*0.01
    analysis_str += '{}\n\n'.format(ctr_winner)
    print(analysis_str)
    write_results(results,os.path.join(args.output_path,res_file_last_part+'_analysis.html'),analysis_str)
