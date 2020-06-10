#!/bin/bash
import os
import time

#fb15k
train_ml="0"
nt="6"
dataset='fb15k'
tp="../logs/{}/sm_with_id.data.pkl".format(dataset) 
ltp="../logs/{}/sm_sup_train_with_id.pkl".format(dataset)
vp="../logs/{}/sm_sup_valid_with_id.pkl".format(dataset)
ldp="../logs/{}/label_distribution.yml".format(dataset)
testp="../logs/{}/test_hits1_single_label_sm.data.pkl.pkl".format(dataset)
config="configs/{}_config.yml".format(dataset)
exclude_t_ids="2 5"
hidden_unit_list="90 40"
testlp="../data/{}/test/test_hits_1_ordered_y.txt".format(dataset)
vlp1="../logs/{}/sm_sup_valid_multilabels.txt".format(dataset)
tlp="../logs/{}/sm_sup_train_multilabels.txt".format(dataset)
df="-0.05"
exdf="1"
supervision="sup"

#yago
# train_ml="0"
# nt="6"
# dataset='yago'
# tp="../logs/{}/sm_with_id.data.pkl".format(dataset) 
# ltp="../logs/{}/sm_sup_train_with_id.pkl".format(dataset)
# vp="../logs/{}/sm_sup_valid_with_id.pkl".format(dataset)
# ldp="../logs/{}/label_distribution.yml".format(dataset)
# testp="../logs/{}/test_hits1_sm.data.pkl.pkl".format(dataset)
# config="configs/{}_config.yml".format(dataset)
# exclude_t_ids="2 5"
# hidden_unit_list="90 40"
# testlp="../data/{}/test/test_hits1_y.txt".format(dataset)
# vlp1="../logs/{}/sm_sup_valid_multilabels.txt".format(dataset)
# tlp="../logs/{}/sm_sup_train_multilabels.txt".format(dataset)
# df="-0.05"
# exdf="1"
# supervision="sup"

num_runs = 5
kldiv_lambdas = [0]# IDEA is to run with 0 on shakuntla and with 1 yatin's system
neg_rewards = [-2]
rhos = [0.125]

# kldiv_lambdas = [1]# IDEA is to run with 0 on shakuntla and with 1 yatin's system
# neg_rewards = [-2]
# rhos = [0.125]

for kl in kldiv_lambdas:
	for nr in neg_rewards:
		for rho in rhos:
			for i in range(1,num_runs+1):
				exp_name="grid_search/{}/{}/kl_{}_rho_{}_nr_{}".format(dataset,supervision,kl,rho,nr)
				print("---------------------------------------Running TRAIN for (kl,nr,rho,run):",kl,nr,rho,i)
				time.sleep(2)
				os.system('CUDA_VISIBLE_DEVICES=0 python3 main.py --training_data_path {} --labelled_training_data_path {} --val_data_path {} --exp_name train --num_epochs 20 --config {} --hidden_unit_list {} --kldiv_lambda {} --neg_reward {} --rho {} --lr 0.001 --cuda --batch_size 2048 --mil --num_templates {} --each_input_size 7 --supervision {} --output_path {}/run_{}/ --label_distribution_file {} --exclude_t_ids {} --default_value {} --exclude_default {} --train_labels_path {} --val_labels_path {} --eval_ml {} --train_ml {}'.format(tp,ltp,vp,config,hidden_unit_list,kl,nr,rho,nt,supervision,exp_name,i,ldp,exclude_t_ids,df,exdf,tlp,vlp1,train_ml,train_ml))
				print("---------------------------------------Running TEST for (kl,nr,rho,run):",kl,nr,rho,i)
				time.sleep(2)
				os.system("python3 main.py --training_data_path {} --labelled_training_data_path {} --val_data_path {} --exp_name test --num_epochs 20 --config {} --hidden_unit_list {} --kldiv_lambda {} --neg_reward {} --rho {} --lr 0.001 --cuda --batch_size 2048 --mil --num_templates {} --each_input_size 7 --supervision {} --output_path {}/run_{}/ --checkpoint {}/run_{}/train/train_r{}_p1_n{}_i4_k{}_best_checkpoint.pth0 --only_eval --pred_file preds.txt --log_eval {}/train_x_eval_sl.csv  --label_distribution_file {} --exclude_t_ids {} --default_value {} --exclude_default {} --val_labels_path {} --eval_ml 0;".format(tp,ltp,testp,config,hidden_unit_list,kl,nr,rho,nt,supervision,exp_name,i,exp_name,i,rho,float(nr),float(kl),exp_name,ldp,exclude_t_ids,df,exdf,testlp))
				# os.system("python3 main.py --training_data_path $tp --labelled_training_data_path $ltp --val_data_path $testp --exp_name test_ml --num_epochs 20 --config $config --hidden_unit_list ${hidden_unit_list[@]} --kldiv_lambda $kl --neg_reward $nr --rho $rho --lr 0.001 --cuda --batch_size 2048 --mil --num_templates $nt --each_input_size 7 --supervision sup --output_path ${expname}/run_$i/ --checkpoint ${expname}/run_$i/train/train_r${rho}_p1_n${nr}_i4_k${kl}_best_checkpoint.pth0 --exclude_t_ids ${exclude_t_ids[@]} --only_eval --pred_file preds.txt --log_eval ${expname}/train_x_eval_ml.csv --val_labels_path $testlp --label_distribution_file $ldp --default_value $df --exclude_default $exdf --eval_ml 1;")
				os.system("python3 main.py --training_data_path {} --labelled_training_data_path {} --val_data_path {} --exp_name test_ml --num_epochs 20 --config {} --hidden_unit_list {} --kldiv_lambda {} --neg_reward {} --rho {} --lr 0.001 --cuda --batch_size 2048 --mil --num_templates {} --each_input_size 7 --supervision {} --output_path {}/run_{}/ --checkpoint {}/run_{}/train/train_r{}_p1_n{}_i4_k{}_best_checkpoint.pth0 --only_eval --pred_file preds.txt --log_eval {}/train_x_eval_ml.csv  --label_distribution_file {} --exclude_t_ids {} --default_value {} --exclude_default {} --val_labels_path {} --eval_ml 1;".format(tp,ltp,testp,config,hidden_unit_list,kl,nr,rho,nt,supervision,exp_name,i,exp_name,i,rho,float(nr),float(kl),exp_name,ldp,exclude_t_ids,df,exdf,testlp))
				print("---------------------------------------DONE for (kl,ne,rho,run):",kl,nr,rho,i)
				time.sleep(2)


