from __future__ import print_function
description_str = "Script to create multinode hpc jobs. \n \
exp_{i}.sh scripts are created in args.jobs_dir. Each exp_{i}.sh script is a process to be run on one node. It fires args.num_task_per_process tasks in parallel.\n \
These processes can be run either individually - via job_{i}.sh or through one of multi_job_{k}.sh. \n \
Each multi node job multi_job_{k}.sh will run args.num_process_per_job number of processes by doing an ssh on each of the node in $PBS_NODESFILE. Ensure that passwordless ssh is enabled and number of nodes selected in args.multi_template are in sync with args.num_process_per_job. By default, each multinode jobs runs 6 processes on total of 3 nodes with 2 gpus per node. \n \
args.single_job_file submits all single jobs  job_{i}.sh via qsub. \n \
args.multi_job_file submits all multi node jobs multi_job_{k}.sh via qsub. \n\n \
Each command in exp_{i} runs args.task_script with a combination of input arguments as hard-coded in this script. Different values of an input argument should be provided as a list and a separate list for each input arg should be provided. e.g. params1, params2 and params3 in the code below.  #Tasks = Cross product of params1, params2 and params3.\n \
Jobs are sorted in the decreasing order of time it takes to run them. \n \
Time of each job is decided by one of the arguments to the task script. 'timing_key' in the code below should be set to the argument name that decides the time. 'timing' list contains the time for each job. \n \
NOTE: you may have to modify the last multi node job manually if total number of tasks to be run is not a multiple of args.num_process_per_job*args.num_task_per_process.  \n\
"

import itertools
import argparse
import sys
import os
from time import sleep
import random
import stat
import copy 

parser = argparse.ArgumentParser(description = description_str)
parser.add_argument('-num_task_per_process',default=1,type=int, help='num tasks to run in parallel in each process', required=True)
parser.add_argument('-num_process_per_job',default=6,type=int, help='num processes to be run in each multinode job', required= True)
parser.add_argument('-task_script',required=True,type=str, help='path to the task script')

parser.add_argument('-template', default='single_run.sh', required=False, type=str)
parser.add_argument('-multi_header', default='multinode_header.sh', required=False, type=str)
parser.add_argument('-multi_template', default='multinode_run.sh', required=False, type=str)
parser.add_argument('-single_job_file',default='all_single_jobs.sh',type=str, required=False)
parser.add_argument('-multi_job_file',default='all_multi_jobs.sh',type=str, required = False)
parser.add_argument('-jobs_dir',default='multinodejobs',type=str,help='directory to be created where all generated files/scripts will reside')
parser.add_argument('-job_name',default='mnj',type=str)
parser.add_argument('-selectos',default=' ',type=str)
parser.add_argument('-num_seq',default = 1,type=int, help='how many blocks in sequence one after the other?')
args = parser.parse_args(sys.argv[1:])

working_dir = os.path.dirname(os.path.join(os.getcwd(),args.task_script))
ack_dir = os.path.join(os.getenv('PWD'),args.jobs_dir)


######################
#To be changed as per the input arguments of the task_script ####
# In the demo example, dummy_task_script.py takes three input arguments named input1, input2 and input3. Timing for each job has to be decided by timing_key parameter

#module_load_str = 'module load apps/pythonpackages/3.6.0/pytorch/0.4.1/gpu'
module_load_str = 'module load apps/anaconda3/4.6.9' 

def get_functional_setting_string(log_str, this_settings, base_logs, run_id):
    run_dir = '{}/run_{}/exp_{}'.format(base_logs, run_id, log_str)
    return '--dir {}'.format(run_dir)


def get_log_file_path(log_str, this_settings, base_logs,run_id):
    run_dir = '{}/run_{}/exp_{}'.format(base_logs, run_id, log_str)
    try:
        os.makedirs(os.path.join(working_dir,run_dir))
    except:
        pass
    #
    return os.path.join(run_dir,'_LOGS')

num_runs = 3
folds=5
dataset='fb15k'
supervision="semi"
unlabelled_training_data_path='../logs/{}/sm_with_id.data.pkl'.format(dataset)
kldiv_dist_file='../data/{}/labelled_train/label_distribution_y6.yml'.format(dataset)
base_model_file='../dumps/{}_distmult_dump_norm.pkl'.format(dataset)
exclude_default = 1
base_logs = 'cross_val/{}/{}'.format(dataset,supervision)

common_setting_string = '--folds {} --labelled_training_data_path {} --unlabelled_training_data_path {} --num_epochs 20 --batch_size 2048 --num_templates 6 --each_input_size 7 --supervision {} --label_distribution_file {} --exclude_default {}'.format(folds, base_logs, unlabelled_training_data_path, supervision, kldiv_dist_file, exclude_default)  


#neg_reward = [-0.5, -1, -2]
#rho = [0.01, 0.05, 0.1, 0.125, 0.25, 0.5]
#config = ['configs/fb15k_config_90_40.yml', 'configs/fb15k_config_7_4.yml'] 
#kldiv_lambda = [0, 1]

neg_reward = [-1, -2]
rho = [0.1, 0.125]
config = ['configs/fb15k_config.yml'] 
kldiv_lambda = [0, 1]
exclude_t_ids = ['2 5']
hidden_unit_list = ['90 40','7 5 5 3']
default_value = [0, -0.05, -0.1]
#

names = ['neg_reward','rho','kldiv_lambda','config','exclude_t_ids','hidden_unit_list','default_value']
all_params = [neg_reward,rho, kldiv_lambda, config,exclude_t_ids,hidden_unit_list,default_value]
short_names = ['n','r','k','c','ex','hul','df']

assert len(names) == len(all_params)
assert len(all_params) == len(short_names)

timing_key = 'hidden_unit_list'
timing = [10]*len(hidden_unit_list)
#assert(len(globals()[timing_key]) == len(timing))
assert len(all_params[names.index(timing_key)]) == len(timing),'len of timing should be same as len of timing_key param'
timing_dict = dict(zip(all_params[names.index(timing_key)],timing))
all_jobs = list(itertools.product(*all_params))

additional_names = ['train_ml','eval_ml']
additional_job_list = [
                [0,0],
                [1,1]
                ]

names = names + additional_names
additional_short_names = ['tml','eml']
short_names = short_names + additional_short_names


assert len(names) == len(short_names)


name2short = dict(zip(names,short_names))
all_jobs = list(itertools.product(all_jobs,additional_job_list))
sorted_names = copy.deepcopy(names)
sorted_names.sort()

jobs_list = {}
sorted_names = copy.deepcopy(names)
all_settings = {}
sorted_names.sort()
job_name_to_time = {}

################################


time_header ='#PBS -l walltime={}:00:00'
#PBS -q workshop

if not os.path.exists(ack_dir):
    os.makedirs(ack_dir)

slurm_cmd = open(args.template).read()+'\n'

pid_closing = 'for pid in ${pids[*]}; do \n \
        wait $pid \n\
done\n'

#hack_str = ". /etc/profile.d/modules.sh"
hack_str = " "
multi_header = open(args.multi_header).read()
multi_header = multi_header.replace('${selectos}',args.selectos)
multi_run_script = open(args.multi_template).read()
multi_run_script = multi_run_script.replace('${exp_dir}',ack_dir)

hpcpy  = '$HOME/anaconda3/bin/python'
base_cmd = '{} {} {}'.format(hpcpy, os.path.join(os.getcwd(),args.task_script), common_setting_string) 

for i, setting in enumerate(all_jobs):
    setting = list(itertools.chain(*setting))
    name_setting = {n: s for n, s in zip(names, setting)}
    setting_list = ['--%s %s' % (name, str(value)) for name, value in name_setting.items() if value is not None]
    setting_str = ' '.join(setting_list)
    log_str = '_'.join(['%s-%s' % (name2short[n], str(name_setting[n]).replace('/','.').replace(' ','.')) for n in sorted_names])
    jobs_list[log_str] = setting_str
    job_name_to_time[log_str] = timing_dict[name_setting[timing_key]]
    all_settings[log_str] = name_setting 

sorted_job_names = list(job_name_to_time.keys())
sorted_job_names.sort(key=lambda x: job_name_to_time[x], reverse=True)

print('Running %d jobs' % (len(jobs_list)))

hpcfile = os.path.join(args.jobs_dir, args.single_job_file)
fh = open(hpcfile,'w')
#fhdair = open(os.path.join(args.jobs_dir, args.single_job_file+'_dair.sh'),'w')
mode = stat.S_IROTH | stat.S_IRWXU | stat.S_IXOTH | stat.S_IRGRP | stat.S_IXGRP
log_str_single_job_file  = os.path.join(args.jobs_dir, args.single_job_file+'_logstr.txt')
log_str_file  = open(log_str_single_job_file,'w')
count = 0
jcount = 0
mjcount = 0
fhj = None
#for log_str, setting_str in jobs_list.items():
for run_id in range(1,(num_runs+1)):
    for log_str in sorted_job_names:
        setting_str = jobs_list[log_str]
        functional_setting_string = get_functional_setting_string(log_str, all_settings[log_str], base_logs, run_id)
        log_file_path = get_log_file_path(log_str, all_settings[log_str], base_logs, run_id)
        bash_cmd = '{} {} {}'.format(base_cmd, setting_str, functional_setting_string)
        if count % args.num_task_per_process == 0:
            if fhj is not None:
                print(pid_closing, file = fhexp)
                print('touch {}/JACK_{}'.format(ack_dir,jcount), file = fhexp)
                fhexp.close()
                print('bash {}'.format(os.path.basename(tfname)),file=fhj)
                fhj.close()
                os.chmod(tfname,mode)
                os.chmod(tfname_job,mode)
                print('qsub {}'.format(os.path.basename(tfname_job)), file = fh)
                jcount += 1
            
            if jcount % args.num_process_per_job == 0:
                print("Creating new multi job. count: {},  jcount: {}, mjcount: {}".format(count, jcount, mjcount))
                fhmjname = os.path.join(args.jobs_dir, 'multi_job_'+str(mjcount)+'.sh')
                fhmj = open(fhmjname, 'w')
                header = '#PBS -N {}_mn_{}_{}'.format(args.job_name,mjcount,log_str[:10])
                print(header, file = fhmj)
                print(time_header.format(job_name_to_time[log_str]),file=fhmj)
                print(multi_header,file = fhmj)
                print('count={}'.format(jcount),file = fhmj)
                print(multi_run_script, file = fhmj)
                fhmj.close()
                os.chmod(fhmjname, mode)
    
                mjcount += 1
    
            
            tfname = os.path.join(args.jobs_dir,'exp_'+str(jcount)+'.sh')
            tfname_job = os.path.join(args.jobs_dir,'job_'+str(jcount)+'.sh')
            fhj = open(tfname_job,'w')
            fhexp = open(tfname,'w') 
            this_time_header = time_header.format(job_name_to_time[log_str])
            header = '#PBS -N job_{}_{}\n{}\n{}\n'.format(jcount,log_str[:10],this_time_header,slurm_cmd)
            print(header, file = fhj)
            
            print(hack_str, file = fhexp)
            print(module_load_str, file = fhexp)
            print('cd {}'.format(working_dir), file= fhexp)
            print('rm {}/JACK_{}'.format(ack_dir,jcount), file = fhexp)
    
        #
    
        print("count: {},  jcount: {}, mjcount: {}".format(count, jcount, mjcount))
        print('{} > {} 2>&1 &'.format(bash_cmd,log_file_path), file =fhexp)
        print("pids[{}]=$!".format(count%args.num_task_per_process),file = fhexp)
        print('{} {}'.format(count, log_str), file = log_str_file)
        count += 1
    

if fhj is not None:
    print("Closing last job")
    print(pid_closing, file = fhexp)
    print('touch {}/JACK_{}'.format(ack_dir,jcount), file = fhexp)
    fhexp.close()
    print('bash {}'.format(os.path.basename(tfname)),file=fhj)
    fhj.close()
    os.chmod(tfname,mode)
    os.chmod(tfname_job,mode)
    print('qsub {}'.format(os.path.basename(tfname_job)), file = fh)
    #jcount += 1
    if jcount % args.num_process_per_job == 0:
        print("Writing last multi job")
        fhmjname = os.path.join(args.jobs_dir, 'multi_job_'+str(mjcount)+'.sh')
        fhmj = open(fhmjname, 'w')
        header = '#PBS -N {}_mn_{}_{}\n{}\n'.format(args.job_name,mjcount,log_str[:10],slurm_cmd)
        print(header, file = fhmj)
        print(multi_header,file = fhmj)
        print('count={}'.format(jcount),file = fhmj)
        print(multi_run_script, file = fhmj)
        fhmj.close()
        os.chmod(fhmjname, mode)
        mjcount += 1
    

fh.close()
os.chmod(hpcfile,mode)
log_str_file.close()

all_multi_file_name = os.path.join(os.getcwd(),args.jobs_dir, args.multi_job_file)
fh = open(all_multi_file_name,'w')
for i in range(mjcount):
    print('qsub multi_job_{}.sh'.format(i),file=fh)

fh.close()
os.chmod(all_multi_file_name, mode)

print("Finished")
