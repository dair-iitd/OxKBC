#!/bin/bash

# folder_path=6_6_Experiments

NEG_REWARD=(1 0.5 0.25 0.125 0.01 0 -0.05 -0.125 -0.5 -1 -2 -4 -8)
RHO=(-2 -1 -0.5 -0.25 -0.125 -0.05 -0.01 0 0.01 0.05 0.125 0.25 0.5)
# NEG_REWARD=(-0.5)
# RHO=(-5)

dataset='fb15k'
hidden_unit_list="hidden_unit_list: [90,40]"
logs="cross_val/cross_val_fb15k"
mkdir -p $logs
unlabelled_training_data_path='../logs/'${dataset}'/sm_with_id.data.pkl'
labelled_training_data_path='../logs/'${dataset}'/exp_words/sm_valid_with_id.pkl'
# val_data_path='../logs/'${dataset}'/exp_words/sm_sup_valid_with_id.pkl'
base_model_file='../dumps/'${dataset}'_distmult_dump_norm.pkl'
folds=5

# cd $folder_path
for i in "${NEG_REWARD[@]}"
do
	# cd $i
	sh="exp_"$i".sh"
	echo $sh
	cp "single_run_cross_val.sh" "$sh"
	counter=0
	for j in "${RHO[@]}"
	do
		# echo $i $j
		temp="exp_""$i""_""$j"
		dir=${logs}"/"${temp}
		yml=$logs/$temp'.yml'
		echo $yml
		cp "config_template.yml" "$yml"
		echo "neg_reward: $i" >> $yml
		echo "rho: $j" >> $yml
		echo "${hidden_unit_list}" >> $yml
		echo "/home/cse/btech/cs1150210/anaconda3/bin/python3 cross_validation.py --folds ${folds} --dir ${dir} --labelled_training_data_path $labelled_training_data_path --unlabelled_training_data_path ${unlabelled_training_data_path} --num_epochs 20 --config $yml --batch_size 2048 --num_templates 5 --each_input_size 7 --supervision semi &" >> $sh
		echo "pids[${counter}]=""$""!" >> $sh
		counter=$((counter+1))
		# qsub -P cse $temp
	done
	echo "for pid in \${pids[*]}; do" >> $sh
	echo "	wait \$pid" >> $sh
	echo "done" >> $sh
	qsub -P cse $sh
	rm $sh		
done
