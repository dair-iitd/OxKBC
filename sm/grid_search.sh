#!/bin/bash

# folder_path=6_6_Experiments

NEG_REWARD=(1 0.5 0.25 0.125 0.01 0 -0.05 -0.125 -0.5 -1 -2 -4 -8)
RHO=(-5 -2 -1 -0.25 -0.125 -0.05 -0.01 0 0.01 0.05 0.125 0.25 0.5)
# NEG_REWARD=(-0.5)
# RHO=(-5)

dataset='fb15k'
logs='grid_search_2/grid_search_'${dataset}'_90_40_robust'
mkdir -p $logs
training_data_path='../logs/'${dataset}'/sm_with_id.data.pkl'
val_data_path='../logs/'${dataset}'/exp_words/sm_valid_with_id.pkl'
base_model_file='../dumps/'${dataset}'_distmult_dump_norm.pkl'


# cd $folder_path
for i in "${NEG_REWARD[@]}"
do
	# cd $i
	for j in "${RHO[@]}"
	do
		# echo $i $j
		temp="exp_""$i""_""$j"
		yml=$logs/$temp'.yml'
		sh=$temp'.sh'
		echo $yml
		echo $sh
		cp "config_template.yml" "$yml"
		cp "single_run.sh" "$sh"
		echo "neg_reward: $i" >> $yml
		echo "rho: $j" >> $yml
		echo "/home/cse/btech/cs1150210/anaconda3/bin/python3 main.py --training_data_path $training_data_path --exp_name $temp --output_path $logs --num_epochs 20 --config $yml --lr 0.001 --log_after 500000 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --val_data_path $val_data_path --base_model_file $base_model_file" >> $sh
		qsub -P cse $sh
		rm $sh		
		# qsub -P cse $temp
	done
done
