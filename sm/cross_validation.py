## Use this file to run cross validation.
## Use with @grid_search_cross_val.sh
import argparse
import os
import pickle
import numpy as np


def create_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def generate_data(args):
    create_dir(args.dir)
    labelled_data = pickle.load(open(args.labelled_training_data_path, 'rb'))
    np.random.shuffle(labelled_data)
    length = labelled_data.shape[0]
    test_sample_size = int(1.0/args.folds*length)
    for i in range(args.folds):
        test_range_beg = i*test_sample_size
        test_range_end = (i+1)*test_sample_size
        test_sample = labelled_data[test_range_beg:test_range_end]
        test_data_path = os.path.join(args.dir, "test_"+str(i)+".pkl")
        pickle.dump(test_sample, open(test_data_path, "wb"))

        train_val_sample = np.concatenate(
            (labelled_data[0:test_range_beg], labelled_data[test_range_end:]), axis=0)
        np.random.shuffle(train_val_sample)

        if(args.supervision == 'un'):
            train_sample_size = 0
        else:
            train_sample_size = int(0.6*length)

        train_sample = train_val_sample[0:train_sample_size]
        val_sample = train_val_sample[train_sample_size:]
        train_data_path = os.path.join(args.dir, "train_"+str(i)+".pkl")
        val_data_path = os.path.join(args.dir, "val_"+str(i)+".pkl")
        pickle.dump(train_sample, open(train_data_path, "wb"))
        pickle.dump(val_sample, open(val_data_path, "wb"))


def main(args):
    if(args.gen_data):
        generate_data(args)
        exit(0)

    val_log = os.path.join(args.dir, "val")
    test_log = os.path.join(args.dir, "test")
    os.system("rm -rf "+val_log)
    os.system("rm -rf "+test_log)

    for i in range(args.folds):
        exp_name = "exp_"+str(i)
        directory = os.path.join(args.dir, exp_name)
        create_dir(directory)

        train_data_path = os.path.join(os.path.abspath(
            args.labelled_training_data_path), "train_"+str(i)+".pkl")
        val_data_path = os.path.join(os.path.abspath(
            args.labelled_training_data_path), "val_"+str(i)+".pkl")
        test_data_path = os.path.join(os.path.abspath(
            args.labelled_training_data_path), "test_"+str(i)+".pkl")

        if not (os.path.isfile(train_data_path) and os.path.isfile(val_data_path) and os.path.isfile(test_data_path)):
            print("Valid data not present at {}, {}, {}".format(
                train_data_path, val_data_path, test_data_path))
            exit(0)

        command = "/home/cse/btech/cs1150210/anaconda3/bin/python3 main.py --training_data_path " + \
            args.unlabelled_training_data_path+" --val_data_path " + \
            val_data_path+" --exp_name "+exp_name+" "
        command += "--labelled_training_data_path " + \
            train_data_path+" --output_path "+args.dir+" "
        command += "--num_epochs "+str(args.num_epochs)+" --batch_size "+str(
            args.batch_size)+" --each_input_size "+str(args.each_input_size)+" "
        command += "--num_templates " + \
            str(args.num_templates)+" --mil --config "+args.config+" --cuda "
        command += "--supervision "+args.supervision+" "
        if(args.kldiv_lambda != 0 and args.label_distribution_file != ''):
            command += "--kldiv_lambda "+str(args.kldiv_lambda) +" --label_distribution_file "+args.label_distribution_file + " "

        print(command)
        # continue
        os.system(command)

        files = os.listdir(directory)
        filename = None
        for x in files:
            if ".pth0" in x:
                filename = x
        if filename is None:
            continue

        checkpoint_path = os.path.join(directory, filename)
        command = "/home/cse/btech/cs1150210/anaconda3/bin/python3 main.py --training_data_path " + \
            args.unlabelled_training_data_path+" --val_data_path " + \
            val_data_path+" --exp_name "+exp_name+" "
        command += "--labelled_training_data_path " + \
            train_data_path+" --output_path "+args.dir+" "
        command += "--num_epochs "+str(args.num_epochs)+" --batch_size "+str(
            args.batch_size)+" --each_input_size "+str(args.each_input_size)+" "
        command += "--num_templates " + \
            str(args.num_templates)+" --mil --config "+args.config+" --cuda "
        command += "--supervision "+args.supervision + " "
        if(args.kldiv_lambda != 0 and args.label_distribution_file != ''):
            command += "--kldiv_lambda "+str(args.kldiv_lambda) +" --label_distribution_file "+args.label_distribution_file + " "
        command += "--only_eval --log_eval "+val_log+" --checkpoint "+checkpoint_path
        command += " --pred_file valid_preds.txt"
        print(command)

        os.system(command)

        command = "/home/cse/btech/cs1150210/anaconda3/bin/python3 main.py --training_data_path " + \
            args.unlabelled_training_data_path+" --val_data_path " + \
            test_data_path+" --exp_name "+exp_name+" "
        command += "--labelled_training_data_path " + \
            train_data_path+" --output_path "+args.dir+" "
        command += "--num_epochs "+str(args.num_epochs)+" --batch_size "+str(
            args.batch_size)+" --each_input_size "+str(args.each_input_size)+" "
        command += "--num_templates " + \
            str(args.num_templates)+" --mil --config "+args.config+" --cuda "
        command += "--supervision "+args.supervision+" "
        if(args.kldiv_lambda != 0 and args.label_distribution_file != ''):
            command += "--kldiv_lambda "+str(args.kldiv_lambda) +" --label_distribution_file "+args.label_distribution_file + " "
        command += "--only_eval --log_eval "+test_log+" --checkpoint "+checkpoint_path
        command += " --pred_file test_preds.txt"
        print(command)

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

    parser.add_argument('--gen_data', help='Just Generate Data and exit',
                        action='store_true', default=False)

    args = parser.parse_args()

    main(args)
