import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import argparse

def normalize(data,each_input_size):
    means = np.mean(data[:,:-1], axis=0).reshape(1, -1)
    stds = np.std(data[:, :-1], axis=0).reshape(1, -1)
    for i in range((means.shape[1])//each_input_size):
        idx = i*each_input_size
        temp = np.concatenate((data[:, idx], data[:, idx+1])).reshape(-1)
        new_mean = np.mean(temp)
        new_std = np.std(temp)
        means[0, idx] = new_mean
        means[0, idx+1] = new_mean
        stds[0, idx] = new_std
        stds[0, idx+1] = new_std
        means[0, idx+3] = 0
        means[0, idx+4] = 0
        stds[0, idx+3] = 1
        stds[0, idx+4] = 1
    data[:, 0:-1] = (data[:, 0:-1]-means)/stds
    return data

def template_plots(datapath,save_path,each_input_size,num_templates):
    data = pickle.load(open(datapath,'rb'))
    data = normalize(data,each_input_size)
    data_pos = data[data[:,-1]>0][:,:-1]
    x = np.arange(0,data_pos.shape[0])*100.0/data_pos.shape[0]
    for i in range(num_templates):
        plt.plot(x,sorted(data_pos[:,i*each_input_size],reverse=True),'b.')
        plt.xlabel('%% of Samples')
        plt.ylabel('Template Score')
        plt.title('Score vs Sample count for template '+str(i+1))
        print("Saving Figure for ",i)
        plt.savefig(os.path.join(save_path,"norm_plot_"+str(i+1)+".png"),format="png",bbox_inches='tight')
        print("Saved Figure for ",i)
        plt.clf()


def thres_plot(datapath,save_path,each_input_size,num_templates):
    data = pickle.load(open(datapath,'rb'))
    data = normalize(data,each_input_size)
    data_pos = data[data[:,-1]>0][:,:-1]
    y = np.concatenate([data_pos[:,i*each_input_size].reshape(-1,1) for i in range(num_templates)],axis=1)
    y = np.max(y,axis=1)
    x = np.arange(0,data_pos.shape[0])*100.0/data_pos.shape[0]
    plt.plot(x,sorted(y,reverse=True),'b.')
    plt.xlabel('%% of Samples Explainable')
    plt.ylabel('Normalized Template Score')
    plt.title('Normalized Score vs Sample count')
    plt.savefig(os.path.join(save_path,"plot_thres_exp.png"),format="png",bbox_inches='tight')
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help="Path of the pkl data", required=True)
    parser.add_argument('-s', '--save_path', help="Path where to save graphs", required=True)
    parser.add_argument('--num_templates',required=False, type=int,default=5)
    parser.add_argument('--each_input_size',required=False, type=int,default=7)
    args = parser.parse_args()

    os.makedirs(args.save_path,exist_ok=True)

    template_plots(args.data,args.save_path,args.each_input_size,args.num_templates)
    thres_plot(args.data,args.save_path,args.each_input_size,args.num_templates)
