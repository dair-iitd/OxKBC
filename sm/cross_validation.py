import argparse
import os
import pickle
import numpy as np

def create_dir(dir_path):
    if(os._exists(dir_path)):
        return
    os.system("mkdir -p "+dir_path)
    return

def main(args):
    labelled_data=pickle.load(open(args.labelled_training_data_path,'rb'))
    length=labelled_data.shape[0]
    create_dir(args.dir)
    np.random.shuffle(labelled_data)

    val_log=os.path.join(args.dir,"val")
    test_log=os.path.join(args.dir,"test")
    os.system("rm -rf "+val_log)
    os.system("rm -rf "+test_log)

    test_sample_size=int(0.2*length)
    for i in range(args.folds):
        test_range_beg=i*test_sample_size
        test_range_end=(i+1)*test_sample_size
        test_sample=labelled_data[test_range_beg:test_range_end]
        test_data_path = os.path.join(args.dir,"test_"+str(i)+".pkl")
        pickle.dump(test_sample,open(test_data_path,"wb"))

        train_val_sample=np.concatenate((labelled_data[0:test_range_beg],labelled_data[test_range_end:]),axis=0)
        np.random.shuffle(train_val_sample)

        if(args.supervision=='un'):
            train_sample_size=0
        else:
            train_sample_size=int(0.6*length)

        train_sample=train_val_sample[0:train_sample_size]
        val_sample=train_val_sample[train_sample_size:]
        train_data_path = os.path.join(args.dir,"train_"+str(i)+".pkl")
        val_data_path = os.path.join(args.dir,"val_"+str(i)+".pkl")
        pickle.dump(train_sample,open(train_data_path,"wb"))
        pickle.dump(val_sample,open(val_data_path,"wb"))

        exp_name="exp_"+str(i)
        directory=os.path.join(args.dir,exp_name)
        create_dir(directory)

        command="python main.py --training_data_path "+args.unlabelled_training_data_path+" --val_data_path "+val_data_path+" --exp_name "+exp_name+" "
        command+="--labelled_training_data_path "+train_data_path+" --output_path "+args.dir+" "
        command+="--num_epochs "+str(args.num_epochs)+" --batch_size "+str(args.batch_size)+" --each_input_size "+str(args.each_input_size)+" "
        command+="--num_templates "+str(args.num_templates)+" --mil --config "+args.config+" --cuda "
        command+="--supervision "+args.supervision+" "
        # print(command)
        # continue
        os.system(command)
        files=os.listdir(directory)
        filename=None
        # print(files)
        for x in files:
            if ".pth0" in x:
                filename=x
        # print(filename)
        if(filename is None):
            continue
        
        checkpoint_path=os.path.join(directory,filename)
        command="python main.py --training_data_path "+args.unlabelled_training_data_path+" --val_data_path "+val_data_path+" --exp_name "+exp_name+" "
        command+="--labelled_training_data_path "+train_data_path+" --output_path "+args.dir+" "
        command+="--num_epochs "+str(args.num_epochs)+" --batch_size "+str(args.batch_size)+" --each_input_size "+str(args.each_input_size)+" "
        command+="--num_templates "+str(args.num_templates)+" --mil --config "+args.config+" --cuda "
        command+="--supervision "+args.supervision+" "
        command+="--only_eval --log_eval "+val_log+" --checkpoint "+checkpoint_path
        os.system(command)
        # print(command)

        command="python main.py --training_data_path "+args.unlabelled_training_data_path+" --val_data_path "+test_data_path+" --exp_name "+exp_name+" "
        command+="--labelled_training_data_path "+train_data_path+" --output_path "+args.dir+" "
        command+="--num_epochs "+str(args.num_epochs)+" --batch_size "+str(args.batch_size)+" --each_input_size "+str(args.each_input_size)+" "
        command+="--num_templates "+str(args.num_templates)+" --mil --config "+args.config+" --cuda "
        command+="--supervision "+args.supervision+" "
        command+="--only_eval --log_eval "+test_log+" --checkpoint "+checkpoint_path
        os.system(command)
        # print(command)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', help='No of folds in cross-validation',type=int,default=2)
    parser.add_argument('--dir', help='directory to store models and results',type=str)
    parser.add_argument('--unlabelled_training_data_path',
                        help="Unlabelled Training data path (pkl file)", type=str)
    parser.add_argument('--labelled_training_data_path',
                        help="Labelled Training data path (pkl file)", type=str)
    # parser.add_argument('--base_model_file',
    #                     help="Base model dump for loading embeddings", type=str)
    # Training parameters
    parser.add_argument('--num_epochs', help='epochs', type=int, default=100)
    parser.add_argument('--batch_size', help='batch size',
                        type=int, default=256)
    # Model parameters
    parser.add_argument('--each_input_size',
                        help='Input size of each template', type=int, default=7)
    parser.add_argument(
        '--num_templates', help='number of templates excluding other', type=int, default=5)
    # parser.add_argument('--mil', help='Use MIL model',
    #                     action='store_true', default=False)
    # Optimizer parameters
    parser.add_argument('--config', help='yaml config file',
                        type=str, default='default_config.yml')
    parser.add_argument('--cuda', help='if cuda available, use it or not?',
                        action='store_true', default=False)
    parser.add_argument('--supervision', help='possible values - un, semi, sup',
                        type=str, default='un')
    # parser.add_argument('--labelled_training_data_path',
    #                     help="Labelled Training data path (pkl file)", type=str)
    args = parser.parse_args()

    main(args)
