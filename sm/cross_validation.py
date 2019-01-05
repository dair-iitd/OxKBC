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
    val_data=pickle.load(open(args.val_data_path,'rb'))
    length=val_data.shape[0]
    # print(length)
    # print(val_data.shape)
    create_dir(args.dir)

    np.random.shuffle(val_data)
    test_sample_size=length//args.folds

    val_log=os.path.join(args.dir,"val")
    test_log=os.path.join(args.dir,"test")
    os.system("rm -rf "+val_log)
    os.system("rm -rf "+test_log)
    for i in range(args.folds):
        test_range_beg=i*test_sample_size
        test_range_end=(i+1)*test_sample_size
        test_sample=val_data[test_range_beg:test_range_end]
        val_sample=np.concatenate((val_data[0:test_range_beg],val_data[test_range_end:]),axis=0)
        val_data_path = os.path.join(args.dir,"val_"+str(i)+".pkl")
        test_data_path = os.path.join(args.dir,"test_"+str(i)+".pkl")
        pickle.dump(val_sample,open(val_data_path,"wb"))
        pickle.dump(test_sample,open(test_data_path,"wb"))

        exp_name="exp_"+str(i)
        directory=os.path.join(args.dir,exp_name)
        create_dir(directory)

        command="python main.py --training_data_path "+args.training_data_path+" --val_data_path "+val_data_path+" --exp_name "+exp_name+" --output_path "+args.dir+" "
        command+="--num_epochs "+str(args.num_epochs)+" --batch_size "+str(args.batch_size)+" --each_input_size "+str(args.each_input_size)+" "
        command+="--num_templates "+str(args.num_templates)+" --mil --config "+args.config+" --cuda "
        print(command)
        os.system(command)
        files=os.listdir(directory)
        filename=None
        # print(files)
        for x in files:
            if ".pth0" in x:
                filename=x
        # print(filename)
        
        checkpoint_path=os.path.join(directory,filename)
        command="python main.py --training_data_path "+args.training_data_path+" --val_data_path "+val_data_path+" --exp_name "+exp_name+" --output_path "+directory+" "
        command+="--num_epochs "+str(args.num_epochs)+" --batch_size "+str(args.batch_size)+" --each_input_size "+str(args.each_input_size)+" "
        command+="--num_templates "+str(args.num_templates)+" --mil --config "+args.config+" --cuda "
        command+="--only_eval --log_eval "+val_log+" --checkpoint "+checkpoint_path
        os.system(command)
        # print(command)

        command="python main.py --training_data_path "+args.training_data_path+" --val_data_path "+test_data_path+" --exp_name "+exp_name+" --output_path "+directory+" "
        command+="--num_epochs "+str(args.num_epochs)+" --batch_size "+str(args.batch_size)+" --each_input_size "+str(args.each_input_size)+" "
        command+="--num_templates "+str(args.num_templates)+" --mil --config "+args.config+" --cuda "
        command+="--only_eval --log_eval "+test_log+" --checkpoint "+checkpoint_path
        os.system(command)
        # print(command)

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', help='No of folds in cross-validation',type=int,default=2)
    parser.add_argument('--dir', help='directory to store models and results',type=str)
    parser.add_argument('--training_data_path',
                        help="Training data path (pkl file)", type=str)
    parser.add_argument(
        '--val_data_path', help="Validation data path in the same format as training data", type=str, default='')

    parser.add_argument('--base_model_file',
                        help="Base model dump for loading embeddings", type=str)
    parser.add_argument('--exp_name', help='Experiment name',
                        type=str, default='default_exp')
    parser.add_argument(
        '--output_path', help='Output path to store models, and logs', type=str)

    # Training parameters
    parser.add_argument('--num_epochs', help='epochs', type=int, default=100)
    parser.add_argument('--batch_size', help='batch size',
                        type=int, default=256)
    # Model parameters
    parser.add_argument('--each_input_size',
                        help='Input size of each template', type=int, default=7)
    parser.add_argument(
        '--num_templates', help='number of templates excluding other', type=int, default=5)
    parser.add_argument('--mil', help='Use MIL model',
                        action='store_true', default=False)
    # Optimizer parameters
    parser.add_argument('--config', help='yaml config file',
                        type=str, default='default_config.yml')
    parser.add_argument('--cuda', help='if cuda available, use it or not?',
                        action='store_true', default=False)
    args = parser.parse_args()

    main(args)
