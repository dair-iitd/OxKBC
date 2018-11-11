import pickle
import os
import numpy as np
import argparse

def read_data(path):
    data = []
    with open(path,"r",errors='ignore') as infile:
        lines = infile.readlines()
        data = [ line.strip().split() for line in lines]
    return data

def augment_data(data):
    new_data = []
    for el in data:
        l1 = el
        l2 = [el[2],el[1]+'__INV',el[0]]
        new_data.append(l1)
        new_data.append(l2)
    return new_data

def write_data(data,filename):
    with open(filename,"w") as f:
        for el in data:
            f.write(' '.join(el)+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help="Name of the dataset as in data folder", required=True)
    parser.add_argument('--data_repo_root',required=False, default='data')
    args = parser.parse_args()

    dataset_root = os.path.join(args.data_repo_root, args.dataset)
    inv_dataset_root = os.path.join(args.data_repo_root,args.dataset+"-inv")
    os.makedirs(inv_dataset_root,exist_ok=True)

    train = read_data(os.path.join(dataset_root,"train.txt"))
    test = read_data(os.path.join(dataset_root,"valid.txt"))
    valid = read_data(os.path.join(dataset_root,"test.txt"))

    write_data(augment_data(train),os.path.join(inv_dataset_root,"train.txt"))
    write_data(augment_data(test),os.path.join(inv_dataset_root,"valid.txt"))
    write_data(augment_data(valid),os.path.join(inv_dataset_root,"test.txt"))