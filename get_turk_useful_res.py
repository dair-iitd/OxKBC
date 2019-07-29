# This code is used to generate an analysis html for the results of the mturk batch of project -
# TexKBC useful? (id=1419750).
# It requires the results.csv downloaded from mturk.
# Quality control is done, by giving all true facts (data from test.txt, which is known to be true)
# If turker choses false, then that hit is rejected.
# It then generates an analysis html file if all the HITs are valid, if Not it generates a CSV with a reason for rejecting the HIT.
# Upload that CSV to Mturk to reject the HITs, not pay the turkers and republish the hits for other workers to do.
import pandas as pd
import numpy as np
import pprint
import argparse
import collections
import string
import os
import bs4 as bs

ANSWER_OPTIONS = ['true','false','na']

def get_key_answer(key,id):
    return string.Template('Answer.${key}_${id}.on').substitute(key=key,id=id)

def get_key_input(key,id):
    return string.Template('Input.${key}_${id}').substitute(key=key,id=id)

def valid_row(row,book):
    total_sum = 0
    for i in range(5):
        for opt in ANSWER_OPTIONS:
            total_sum += row[get_key_answer(opt,i)]
    if(total_sum != 5):
        return 'You did not mark any option in some questions'

    for i in range(5):
        fact = row[get_key_input('fact',i)]
        fact_text = bs.BeautifulSoup(fact,'lxml').text
        if(str(book[book.fact == fact_text]['true?'].iloc[0]) == 'na'):
            continue
        elif(str(int(book[book.fact == fact_text]['true?'].iloc[0])) == '1' and row[get_key_answer('false',i)] == 1 ):
           return 'You did not chose that the fact is true, though the fact was true.'
        elif(str(int(book[book.fact == fact_text]['true?'].iloc[0])) == '0' and row[get_key_answer('true',i)] == 1 ):
           return 'You did not chose that the fact is false, though the fact was false.'
    return ''

def get_invalid_hits(df,outfilename,book):
    df_new = df.copy()
    df = df.fillna(False)
    invalid_hits = []
    for index,row in df.iterrows():
        message = valid_row(row,book)
        if(message!=''):
            print('Invalid HIT at {} with message ==> {} '.format(index, message))
            df_new['Reject'][index] = message
            invalid_hits.append(row['AssignmentId'])
    if(len(invalid_hits)!=0):
        df_new.to_csv(outfilename,index=False,sep=',')
    return invalid_hits

def get_winner(answers):
    true_ct = 0
    false_ct = 0
    for el in answers:
        if(el=='true'):
            true_ct += 1
        elif(el=='false'):
            false_ct += 1
    if(true_ct > false_ct):
        return ['true']
    elif (false_ct > true_ct):
        return ['false']
    else:
        return ['na']

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
    return pd.DataFrame(data,columns=['fact','exp','true?'])

def get_results(df):
    df = df.fillna(False)
    results = {}
    for index, row in df.iterrows():
        for i in range(5):
            fact = row[get_key_input('fact',i)]
            exp = row[get_key_input('exp',i)]

            if(fact not in results):
                results[fact] = {'exp': exp, 'answers' : [],'time_taken': [] , 'row_idx':[], 'fact_no':[]}

#            if(row[get_key_answer('true',i)]):
            results[fact]['time_taken'].append(float(row['WorkTimeInSeconds'])/5.0)

            for opt in ANSWER_OPTIONS:
                if(row[get_key_answer(opt,i)]):
                    results[fact]['answers'].append(opt)
                    results[fact]['row_idx'].append(index)
                    results[fact]['fact_no'].append(i)
    for k in results:
        winner = get_winner(results[k]['answers'])
        results[k]['winner'] = winner
        results[k]['avg_time'] = np.mean(results[k]['time_taken'])

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
    parser.add_argument('-bf', '--book_file', help="Original HTML (Book) written by get_turk_useful_data", required=True)
    args = parser.parse_args()

    book = get_book(args.book_file)

    df = pd.read_csv(args.result_file)
    df = df[df['AssignmentStatus'] != 'Rejected']

    res_file_last_part = os.path.basename(os.path.normpath(args.result_file)).split('.')[0]
    invalid_hits = get_invalid_hits(df,os.path.join(args.output_path,res_file_last_part+'_rejected.csv'),book)
    if(len(invalid_hits)!=0):
        print('There are {} invalid assignments which have id \n{}'.format(len(invalid_hits),invalid_hits))
        exit(-1)

    results = get_results(df)

    answers_list = []
    winner_list = []
    avg_time_list = []
    for k in results:
        answers_list.extend(results[k]['answers'])
        winner_list.extend(results[k]['winner'])
        avg_time_list.extend(results[k]['time_taken'])

    ctr_answers = collections.Counter(answers_list)
    analysis_str = ''
    analysis_str += 'Total number of annotations = {}\n'.format(len(answers_list))
    for el in ctr_answers:
        ctr_answers[el] /= len(answers_list)*0.01
    analysis_str += '{}\n\n'.format(ctr_answers)

    ctr_winner = collections.Counter(winner_list)
    analysis_str += ('Total number of facts = {}\n'.format(len(results)))
    analysis_str += ('Total number of truth determined facts = {}\n'.format(len(winner_list)-winner_list.count('na')))
    for el in ctr_winner:
        ctr_winner[el] /= len(winner_list)*0.01
    analysis_str += '{}\n\n'.format(ctr_winner)
    analysis_str += '\nAverage time taken in seconds: {}\n\n'.format(np.mean(avg_time_list))
    print(analysis_str)
    write_results(results,os.path.join(args.output_path,res_file_last_part+'_analysis.html'),analysis_str)
