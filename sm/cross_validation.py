## Use this file to run cross validation.
## Use with @grid_search_cross_val.sh
import argparse
import os
import pickle
import numpy as np
import utils
from IPython.core.debugger import Pdb
def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def generate_data(args):
    create_dir(args.dir)
    labelled_data = pickle.load(open(args.labelled_training_data_path, 'rb'))
    labels = open(args.train_labels_path).readlines()
    assert len(labelled_data) == len(labels)
    data_with_labels = list(zip(labelled_data, labels))
    np.random.seed(args.seed)
    np.random.shuffle(data_with_labels)
    #np.random.shuffle(labelled_data)
    length = labelled_data.shape[0]
    test_sample_size = int(1.0/args.folds*length)
    for i in range(args.folds):
        test_range_beg = i*test_sample_size
        test_range_end = (i+1)*test_sample_size
        #test_sample = labelled_data[test_range_beg:test_range_end]
        test_sample = np.array([data_with_labels[j][0] for j in range(test_range_beg,test_range_end)])
        test_sample_y = [data_with_labels[j][1] for j in range(test_range_beg, test_range_end)]

        test_data_path = os.path.join(args.dir, "test_"+str(i)+".pkl")
        pickle.dump(test_sample, open(test_data_path, "wb"))
        test_label_path = os.path.join(args.dir, "test_label_"+str(i)+".csv")
        with open(test_label_path,'w') as tlfh:
            tlfh.write(''.join(test_sample_y))

        train_val_sample = np.concatenate(
            (labelled_data[0:test_range_beg], labelled_data[test_range_end:]), axis=0)
        
        
        train_val_data= data_with_labels[:test_range_beg] + data_with_labels[test_range_end:]

        np.random.seed(args.seed+args.seed)
        np.random.shuffle(train_val_data)

        #train_val_sample = np.concatenate(
        #    (
        #        np.array([data_with_labels[j][0] for j in range(0,test_range_beg)]), 
        #        np.array([data_with_labels[j][0] for j in range(test_range_end,len(data_with_labels))])
        #    ), axis=0)
        
        #np.random.shuffle(train_val_sample)

        if(args.supervision == 'un'):
            train_sample_size = 0
        else:
            train_sample_size = int(0.6*length)
    
        train_sample, train_sample_labels = list(zip(*train_val_data[0:train_sample_size]))
        train_sample = np.array(list(train_sample))
        
        val_sample, val_sample_labels = list(zip(*train_val_data[train_sample_size:]))
        val_sample = np.array(list(val_sample))


        #train_sample = train_val_sample[0:train_sample_size]
        #val_sample = train_val_sample[train_sample_size:]
        train_data_path = os.path.join(args.dir, "train_"+str(i)+".pkl")
        val_data_path = os.path.join(args.dir, "val_"+str(i)+".pkl")
        pickle.dump(train_sample, open(train_data_path, "wb"))
        pickle.dump(val_sample, open(val_data_path, "wb"))

        train_data_label_path = os.path.join(args.dir, "train_label_"+str(i)+".csv")
        val_data_label_path = os.path.join(args.dir, "val_label_"+str(i)+".csv")
        
        with open(train_data_label_path, 'w') as tlfh:
            tlfh.write(''.join(list(train_sample_labels)))

        with open(val_data_label_path, 'w') as tlfh:
            tlfh.write(''.join(list(val_sample_labels)))


"""
def check(x,y):
    a = [int(x[-1]) for x in pickle.load(open(x,'rb'))]
    b = [int(x.strip().split(',')[0]) for x in open(y,'r').readlines()]
    assert len(a) == len(b)
    for i in range(len(a)):
        assert a[i] == b[i]

for ttype in ['train_','val_','test_']:
    for i in range(5):
        print(ttype,i)
        check(ttype+str(i)+'.pkl', ttype+'label_'+str(i)+'.csv')

"""


def main(args):
    if(args.gen_data):
        generate_data(args)
        exit(0)

    val_log = os.path.join(args.dir, "val")
    test_log = os.path.join(args.dir, "test")
    #os.system("rm -rf "+val_log)
    #os.system("rm -rf "+test_log)

    data_root = os.path.abspath(args.labelled_training_data_path)
    
    
    for key in['exclude_t_ids','hidden_unit_list']:
        args[key] = ' '.join(map(str,args[key]))
    
    common_arg_list =  ['val_data_path', 'labelled_training_data_path','exp_name','num_epochs','batch_size','each_input_size','num_templates','config','supervision','kldiv_lambda','label_distribution_file','neg_reward','rho','exclude_t_ids','train_labels_path','val_labels_path','train_ml','eval_ml','hidden_unit_list','exclude_default','default_value']

    for i in range(args.folds):
        print("FOLD: {}".format(i))
        exp_name = "exp_"+str(i)
        directory = os.path.join(args.dir, exp_name)
        create_dir(directory)

        train_data_path = os.path.join(data_root, "train_"+str(i)+".pkl")
        val_data_path = os.path.join(data_root, "val_"+str(i)+".pkl")
        test_data_path = os.path.join(data_root, "test_"+str(i)+".pkl")

        train_labels_path  = os.path.join(data_root, "train_label_{}.csv".format(i))
        val_labels_path  = os.path.join(data_root, "val_label_{}.csv".format(i))
        test_labels_path  = os.path.join(data_root, "test_label_{}.csv".format(i))
        
        args['train_labels_path'] = train_labels_path
        args['val_labels_path'] = val_labels_path

        args['val_data_path'] = val_data_path
        args['labelled_training_data_path'] = train_data_path
        args['exp_name'] = exp_name

        if not (os.path.isfile(train_data_path) and os.path.isfile(val_data_path) and os.path.isfile(test_data_path)):
            print("Valid data not present at {}, {}, {}".format(
                train_data_path, val_data_path, test_data_path))
            exit(0)

        command = args.python_path + " main.py --training_data_path " + args.unlabelled_training_data_path + " "
        command += "--output_path " + str(args.dir) + " --mil --cuda "
        for key in common_arg_list:
            command += ' --{} {}'.format(key,args[key])
            
        print('TRAIN: {}'.format(command))
        # continue
        os.system(command)

        files = os.listdir(directory)
        filename = None
        for x in files:
            if ".pth0" in x:
                filename = x
        if filename is None:
            continue

        checkpoint_path = os.path.join(directory, str(filename))
       
        for eval_ml in [0,1]:
            this_val_log = '{}_ml_{}'.format(val_log ,eval_ml)
            this_pred_file = 'val_pred_ml_{}.txt'.format(eval_ml)
            command = args.python_path + " main.py --training_data_path " + args.unlabelled_training_data_path + " "
            command += "--output_path " + str(args.dir) + " --mil --cuda "
            command += "--only_eval --log_eval "+this_val_log+" --checkpoint "+checkpoint_path
            command += " --pred_file "+this_pred_file
            command += " --train_ml {} --eval_ml {}".format(eval_ml, eval_ml)

            for key in common_arg_list:
                command += ' --{} {}'.format(key,args[key])
        
            print('VAL ML: {} {} '.format(eval_ml, command))
            os.system(command)

        args['val_data_path'] = test_data_path
        args['val_labels_path'] = test_labels_path 
        for eval_ml in [0,1]:
            this_val_log = '{}_ml_{}'.format(test_log ,eval_ml)
            this_pred_file = 'test_pred_ml_{}.txt'.format(eval_ml)
            command = args.python_path + " main.py --training_data_path " + args.unlabelled_training_data_path + " "
            command += "--output_path " + str(args.dir) + " --mil --cuda "
            command += "--only_eval --log_eval "+this_val_log+" --checkpoint "+checkpoint_path
            command += " --pred_file "+this_pred_file
            command += " --train_ml {} --eval_ml {}".format(eval_ml, eval_ml)

            for key in common_arg_list:
                command += ' --{} {}'.format(key,args[key])
        
            print('TEST ML: {} {} '.format(eval_ml, command))
            os.system(command)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--folds', help='No of folds in cross-validation', type=int, default=2)
    parser.add_argument(
        '--dir', help='directory to store models, results and data', type=str)
    parser.add_argument('--unlabelled_training_data_path',
                        help="Unlabelled Training data path (pkl file)", type=str, default=None)
    parser.add_argument('--labelled_training_data_path',
                        help="Labelled Training data path (pkl file) or base_folder path where data previously generated", type=str)

    
    #parser.add_argument('--python_path', help='path to python bin', type=str,default='/home/cse/btech/cs1150210/anaconda3/bin/python3')
    #parser.add_argument('--python_path', help='path to python bin', type=str,default='/home/yatin/anaconda3/bin/python3')
    #parser.add_argument('--python_path', help='path to python bin', type=str,default='/home/cse/phd/csz178057/anaconda3/bin/python3')
    parser.add_argument('--python_path', help='path to python bin', type=str,default='$HOME/anaconda3/bin/python3')


    # Training parameters
    parser.add_argument('--num_epochs', help='epochs', type=int, default=100)
    parser.add_argument('--batch_size', help='batch size',
                        type=int, default=256)
    # Model parameters
    parser.add_argument('--each_input_size',
                        help='Input size of each template', type=int, default=7)
    parser.add_argument(
        '--num_templates', help='number of templates excluding other', type=int, default=5)

    parser.add_argument('--config', help='yaml config file',
                        type=str, default='default_config.yml')
    parser.add_argument('--cuda', help='if cuda available, use it or not?',
                        action='store_true', default=False)
    parser.add_argument('--supervision', help='possible values - un, semi, sup',
                        type=str, default='un')

    parser.add_argument('--kldiv_lambda', help='KL Divergence lambda', type=float, default=0.0)
    parser.add_argument('--label_distribution_file', help='KL Divergence distribution file', type=str, default='')
    parser.add_argument('--neg_reward', help='negative reward', default=-1, type=float)
    parser.add_argument('--rho', help='rho ', default=0.0125, type=float)


    parser.add_argument('--gen_data', help='Just Generate Data and exit',
                        action='store_true', default=False)
    
    parser.add_argument(
        '--train_labels_path', help="Input Training data Labels path for multi-label training", type=str, default=None)

    parser.add_argument('--seed',help='seed to be used before shuffling', type=int, default = 42)
    #parser.add_argument('--exclude_t_ids', type= str, default = '', help='List of templates to be excluded while making predictions')
    parser.add_argument('--exclude_t_ids', nargs='*', type=int, required=False,default=[], help='List of templates to be excluded while making predictions')
    
    parser.add_argument('--default_value',help='default value of template score when it is undefined?', default = 0, type=float)
    parser.add_argument('--exclude_default',help='should default value be excluded while computing stats?', default = 0, type=int)

    parser.add_argument('--train_ml',help='should use multi label loss?', default = 1, type=int)
    parser.add_argument('--eval_ml',help='should eval multi label ?', default = 1, type=int)
 
    parser.add_argument('--hidden_unit_list', nargs='*', type=int, required=False,default=[], help='number of hidden neurons in each layer')
    
    
    
    args = parser.parse_args()
    config = {}
    config.update(vars(args))
    args = utils.Map(config)
    print(args)
    main(args)
