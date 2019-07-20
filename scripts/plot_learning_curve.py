import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
import sys
import matplotlib
matplotlib.use('agg')


def make_graph(data, output_file):
    x = [i for i in range(1, data['nepochs']+2)]

    plt.xlabel('Number of epochs')
    plt.ylabel('Loss and Micro F-score')
    plt.title('Learning Curve')
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    max_y = -1e9
    min_y = 1e9
    for key in data.keys():
        if(key != 'nepochs'):
            max_y = max(max(data[key]), max_y)
            min_y = min(min(data[key]), min_y)
    plt.yticks(np.arange(min_y, max_y+1, (max_y+1-min_y)/20.0))
    for key in data.keys():
        if(key != 'nepochs'):
            plt.plot(x, data[key], '-.', label=key)
    plt.legend()
    plt.savefig(output_file, format="pdf", bbox_inches='tight')


def parse_file(filename):
    nepochs = 0
    eval_mif = []
    train_sup_loss = []
    train_sup_mif = []
    train_un_loss = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        els = line.split(',')
        if(els[0] == 'epoch'):
            continue
        epoch_no = int(els[0])
        if(epoch_no == -1):
            continue
        nepochs = max(nepochs, epoch_no)
        if(els[1] == 'eval'):
            eval_mif.append(float(els[-1]))
        elif(els[1] == 'train_sup'):
            train_sup_mif.append(float(els[-1]))
            train_sup_loss.append(float(els[2]))
        elif(els[1] == 'train_un'):
            train_un_loss.append(float(els[2]))
        else:
            raise "Invalid type of mode in file at line " + line

    data = {'eval_mif': eval_mif,
            'train_sup_loss': train_sup_loss,
            'train_sup_mif': train_sup_mif,
            'train_un_loss': train_un_loss,
            'nepochs': nepochs}

    return data


if len(sys.argv) != 2:
    print('Usage is '+sys.argv[0] +
          ' <folder_name which contains learning_curve.txt>')
    print('To plot learning curve just give the path of the directory inside which learning_curve.txt is present.')
    print('It will plot a learning_plot.pdf in the same directory')
    exit(-1)

folder_path = ''+sys.argv[1]
data = parse_file(os.path.join(folder_path, 'learning_curve.txt'))
make_graph(data, os.path.join(folder_path, 'learning_plot.pdf'))
