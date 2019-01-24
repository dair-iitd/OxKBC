#!/bin/bash

## Script to run cross validation for the dataset


rerun=false
invalid_file=""
if [ $# -eq 2 ] && [ "$1" == "-re" ]
then
	echo "Running the script in rerun mode"
	rerun=true
	invalid_file=$2
	echo $rerun
	echo $invalid_file
fi

## Parameters to search for
NEG_REWARD=(0 -0.05 -0.125 -0.25 -0.5 -1 -1.5 -2 -4 -8)
RHO=(0 0.01 0.05 0.1 0.125 0.2 0.25 0.35 0.4 0.5 0.75)

## Repeat the cross-validation NTIMES times
NTIMES=5

## Global Variables
folds=5
dataset='yago'
hidden_unit_list="hidden_unit_list: [90,40]"
supervision="semi"
unlabelled_training_data_path='../logs/'${dataset}'/sm_with_id.data.pkl'
labelled_training_data_path='../logs/'${dataset}'/exp_words/sm_valid_with_id.pkl'
base_model_file='../dumps/'${dataset}'_distmult_dump_norm.pkl'
base_logs="cross_val/"${dataset}"_90_40_"${supervision}
mkdir -p $base_logs
if [ "$rerun" = false ]
then
	echo "Genearating Data"
	# /home/cse/btech/cs1150210/anaconda3/bin/python3 cross_validation.py --folds ${folds} --dir ${base_logs} --labelled_training_data_path $labelled_training_data_path --supervision ${supervision} --gen_data
fi
for run in $(seq 1 $NTIMES); do
	logs=${base_logs}"/run_"${run}
	mkdir -p $logs
	echo "Created folder " "$logs"
	for i in "${NEG_REWARD[@]}"; do
		sh="exp_"$i".sh"
		echo "Running for negative reward " "$i"
		cp "single_run_cross_val.sh" "$sh"
		counter=0
		for j in "${RHO[@]}"; do
			temp="exp_""$i""_""$j"
			if [ "$rerun" = true ]
			then
				if grep -Fxq "run_"$run"/"$temp $invalid_file
				then
					echo $temp " Found in invalid file"
				else
					continue
				fi
			fi
			dir=${logs}"/"${temp}
			yml=$logs/$temp'.yml'
			echo $yml
			cp "config_template.yml" "$yml"
			echo "neg_reward: $i" >>$yml
			echo "rho: $j" >>$yml
			echo "${hidden_unit_list}" >>$yml
			echo "/home/cse/btech/cs1150210/anaconda3/bin/python3 cross_validation.py --folds ${folds} --dir ${dir} --labelled_training_data_path $base_logs --unlabelled_training_data_path ${unlabelled_training_data_path} --num_epochs 20 --config $yml --batch_size 2048 --num_templates 5 --each_input_size 7 --supervision ${supervision} &" >>$sh
			echo "pids[${counter}]=""$""!" >>$sh
			counter=$((counter + 1))
		done
		echo "for pid in \${pids[*]}; do" >>$sh
		echo "	wait \$pid" >>$sh
		echo "done" >>$sh
		if [ $counter -ne 0 ]
		then
			echo "Submitting Job"
			qsub -P cse $sh
		fi
		rm $sh
	done
done
