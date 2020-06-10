
import argparse
import os
import pickle
import numpy as np


def shuffle_and_split(args):
    labelled_data = pickle.load(open(args.labelled_total_data_path, 'rb'))
    labels = open(args.total_labels_path).readlines()
    assert len(labelled_data) == len(labels)
    data_with_labels = list(zip(labelled_data, labels))
    np.random.seed(args.seed)
    np.random.shuffle(data_with_labels)
    length = len(data_with_labels)
    train_length = int(length*args.train_split)
    
    #----subset supervision-----
    train_length1 = int(0.33*length*args.train_split)
    train_length2 = int(0.5*length*args.train_split) 
    train_length3 = int(0.67*length*args.train_split)

    train_samples1 = data_with_labels[:train_length1]
    train_samples2 = data_with_labels[:train_length2]
    train_samples3 = data_with_labels[:train_length3]

    train_x1 = np.array([x[0] for x in train_samples1])
    train_y1 = [x[1] for x in train_samples1]
    train_x2 = np.array([x[0] for x in train_samples2])
    train_y2 = [x[1] for x in train_samples2]
    train_x3 = np.array([x[0] for x in train_samples3])
    train_y3 = [x[1] for x in train_samples3]

    pickle.dump(train_x1, open("../logs/yago/subset_supervision/sm_sup_train_with_id_subset1.pkl",'wb'))
    with open("../logs/yago/subset_supervision/sm_sup_train_multilabels_subset1.txt",'w') as fh:
        fh.write(''.join(train_y1))
    pickle.dump(train_x2, open("../logs/yago/subset_supervision/sm_sup_train_with_id_subset2.pkl",'wb'))
    with open("../logs/yago/subset_supervision/sm_sup_train_multilabels_subset2.txt",'w') as fh:
        fh.write(''.join(train_y2))
    pickle.dump(train_x3, open("../logs/yago/subset_supervision/sm_sup_train_with_id_subset3.pkl",'wb'))
    with open("../logs/yago/subset_supervision/sm_sup_train_multilabels_subset3.txt",'w') as fh:
        fh.write(''.join(train_y3))
    #---------------------------


    train_samples = data_with_labels[:train_length]
    val_samples = data_with_labels[train_length:]

    train_x = np.array([x[0] for x in train_samples])
    train_y = [x[1] for x in train_samples]

    val_x = np.array([x[0] for x in val_samples])
    val_y = [x[1] for x in val_samples]

    pickle.dump(train_x, open(args.labelled_training_data_path,'wb'))
    pickle.dump(val_x, open(args.val_data_path,'wb'))

    with open(args.train_labels_path,'w') as fh:
        fh.write(''.join(train_y))


    with open(args.val_labels_path,'w') as fh:
        fh.write(''.join(val_y))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--labelled_total_data_path',
                        help="Input Labelled Training data path (pkl file)", type=str)

    parser.add_argument(
        '--total_labels_path', help="Input Training data Labels path for multi-label training", type=str, default=None)

    parser.add_argument('--labelled_training_data_path',
                        help="Output Labelled Training data path (pkl file)", type=str)

    parser.add_argument(
        '--train_labels_path', help="Output Training data Labels path for multi-label training", type=str, default=None)

    parser.add_argument(
        '--val_data_path', help="Output Validation data path in the same format as training data", type=str, default='')
    
    parser.add_argument(
        '--val_labels_path', help="Output Validation data Labels path for multi-label evaluation", type=str, default=None)

    parser.add_argument('--train_split',help='pct of data for train', type=float, default = 0.8)
    parser.add_argument('--seed',help='seed to be used before shuffling', type=int, default = 42)
    parser.add_argument('--num_templates',help='num_templates', type=int, default = 0)

    args = parser.parse_args()
    shuffle_and_split(args) 
