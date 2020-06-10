 
module load apps/anaconda3/4.6.9
cd /home/cse/phd/csz178057/hpcscratch/Aman_BTP/Interpretable-KBC-tlp/sm
rm /home/yatin/hpcscratch_copy/texkbc/Interpretable-KBC-tlp/sm/hpc_scripts/gs_fb15k_3runs_rep/JACK_3
$HOME/anaconda3/bin/python /home/cse/phd/csz178057/hpcscratch/Aman_BTP/Interpretable-KBC-tlp/sm/cross_validation.py --folds 5 --labelled_training_data_path cross_val/fb15k/semi --unlabelled_training_data_path ../logs/fb15k/sm_with_id.data.pkl --num_epochs 20 --batch_size 2048 --num_templates 6 --each_input_size 7 --supervision semi --label_distribution_file ../data/fb15k/labelled_train/label_distribution_y6.yml --exclude_default 1 --neg_reward -1 --rho 0.125 --kldiv_lambda 1 --config configs/fb15k_config.yml --exclude_t_ids 2 5 --hidden_unit_list 90 40 --default_value 0 --train_ml 0 --eval_ml 0 --dir cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df-0_eml-0_ex-2.5_hul-90.40_k-1_n--1_r-0.125_tml-0 > cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df-0_eml-0_ex-2.5_hul-90.40_k-1_n--1_r-0.125_tml-0/_LOGS 2>&1 &
pids[0]=$!
$HOME/anaconda3/bin/python /home/cse/phd/csz178057/hpcscratch/Aman_BTP/Interpretable-KBC-tlp/sm/cross_validation.py --folds 5 --labelled_training_data_path cross_val/fb15k/semi --unlabelled_training_data_path ../logs/fb15k/sm_with_id.data.pkl --num_epochs 20 --batch_size 2048 --num_templates 6 --each_input_size 7 --supervision semi --label_distribution_file ../data/fb15k/labelled_train/label_distribution_y6.yml --exclude_default 1 --neg_reward -1 --rho 0.125 --kldiv_lambda 1 --config configs/fb15k_config.yml --exclude_t_ids 2 5 --hidden_unit_list 90 40 --default_value 0 --train_ml 1 --eval_ml 1 --dir cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df-0_eml-1_ex-2.5_hul-90.40_k-1_n--1_r-0.125_tml-1 > cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df-0_eml-1_ex-2.5_hul-90.40_k-1_n--1_r-0.125_tml-1/_LOGS 2>&1 &
pids[1]=$!
$HOME/anaconda3/bin/python /home/cse/phd/csz178057/hpcscratch/Aman_BTP/Interpretable-KBC-tlp/sm/cross_validation.py --folds 5 --labelled_training_data_path cross_val/fb15k/semi --unlabelled_training_data_path ../logs/fb15k/sm_with_id.data.pkl --num_epochs 20 --batch_size 2048 --num_templates 6 --each_input_size 7 --supervision semi --label_distribution_file ../data/fb15k/labelled_train/label_distribution_y6.yml --exclude_default 1 --neg_reward -1 --rho 0.125 --kldiv_lambda 1 --config configs/fb15k_config.yml --exclude_t_ids 2 5 --hidden_unit_list 90 40 --default_value -0.05 --train_ml 0 --eval_ml 0 --dir cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df--0.05_eml-0_ex-2.5_hul-90.40_k-1_n--1_r-0.125_tml-0 > cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df--0.05_eml-0_ex-2.5_hul-90.40_k-1_n--1_r-0.125_tml-0/_LOGS 2>&1 &
pids[2]=$!
$HOME/anaconda3/bin/python /home/cse/phd/csz178057/hpcscratch/Aman_BTP/Interpretable-KBC-tlp/sm/cross_validation.py --folds 5 --labelled_training_data_path cross_val/fb15k/semi --unlabelled_training_data_path ../logs/fb15k/sm_with_id.data.pkl --num_epochs 20 --batch_size 2048 --num_templates 6 --each_input_size 7 --supervision semi --label_distribution_file ../data/fb15k/labelled_train/label_distribution_y6.yml --exclude_default 1 --neg_reward -1 --rho 0.125 --kldiv_lambda 1 --config configs/fb15k_config.yml --exclude_t_ids 2 5 --hidden_unit_list 90 40 --default_value -0.05 --train_ml 1 --eval_ml 1 --dir cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df--0.05_eml-1_ex-2.5_hul-90.40_k-1_n--1_r-0.125_tml-1 > cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df--0.05_eml-1_ex-2.5_hul-90.40_k-1_n--1_r-0.125_tml-1/_LOGS 2>&1 &
pids[3]=$!
$HOME/anaconda3/bin/python /home/cse/phd/csz178057/hpcscratch/Aman_BTP/Interpretable-KBC-tlp/sm/cross_validation.py --folds 5 --labelled_training_data_path cross_val/fb15k/semi --unlabelled_training_data_path ../logs/fb15k/sm_with_id.data.pkl --num_epochs 20 --batch_size 2048 --num_templates 6 --each_input_size 7 --supervision semi --label_distribution_file ../data/fb15k/labelled_train/label_distribution_y6.yml --exclude_default 1 --neg_reward -1 --rho 0.125 --kldiv_lambda 1 --config configs/fb15k_config.yml --exclude_t_ids 2 5 --hidden_unit_list 90 40 --default_value -0.1 --train_ml 0 --eval_ml 0 --dir cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df--0.1_eml-0_ex-2.5_hul-90.40_k-1_n--1_r-0.125_tml-0 > cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df--0.1_eml-0_ex-2.5_hul-90.40_k-1_n--1_r-0.125_tml-0/_LOGS 2>&1 &
pids[4]=$!
$HOME/anaconda3/bin/python /home/cse/phd/csz178057/hpcscratch/Aman_BTP/Interpretable-KBC-tlp/sm/cross_validation.py --folds 5 --labelled_training_data_path cross_val/fb15k/semi --unlabelled_training_data_path ../logs/fb15k/sm_with_id.data.pkl --num_epochs 20 --batch_size 2048 --num_templates 6 --each_input_size 7 --supervision semi --label_distribution_file ../data/fb15k/labelled_train/label_distribution_y6.yml --exclude_default 1 --neg_reward -1 --rho 0.125 --kldiv_lambda 1 --config configs/fb15k_config.yml --exclude_t_ids 2 5 --hidden_unit_list 90 40 --default_value -0.1 --train_ml 1 --eval_ml 1 --dir cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df--0.1_eml-1_ex-2.5_hul-90.40_k-1_n--1_r-0.125_tml-1 > cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df--0.1_eml-1_ex-2.5_hul-90.40_k-1_n--1_r-0.125_tml-1/_LOGS 2>&1 &
pids[5]=$!
$HOME/anaconda3/bin/python /home/cse/phd/csz178057/hpcscratch/Aman_BTP/Interpretable-KBC-tlp/sm/cross_validation.py --folds 5 --labelled_training_data_path cross_val/fb15k/semi --unlabelled_training_data_path ../logs/fb15k/sm_with_id.data.pkl --num_epochs 20 --batch_size 2048 --num_templates 6 --each_input_size 7 --supervision semi --label_distribution_file ../data/fb15k/labelled_train/label_distribution_y6.yml --exclude_default 1 --neg_reward -1 --rho 0.125 --kldiv_lambda 1 --config configs/fb15k_config.yml --exclude_t_ids 2 5 --hidden_unit_list 7 5 5 3 --default_value 0 --train_ml 0 --eval_ml 0 --dir cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df-0_eml-0_ex-2.5_hul-7.5.5.3_k-1_n--1_r-0.125_tml-0 > cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df-0_eml-0_ex-2.5_hul-7.5.5.3_k-1_n--1_r-0.125_tml-0/_LOGS 2>&1 &
pids[6]=$!
$HOME/anaconda3/bin/python /home/cse/phd/csz178057/hpcscratch/Aman_BTP/Interpretable-KBC-tlp/sm/cross_validation.py --folds 5 --labelled_training_data_path cross_val/fb15k/semi --unlabelled_training_data_path ../logs/fb15k/sm_with_id.data.pkl --num_epochs 20 --batch_size 2048 --num_templates 6 --each_input_size 7 --supervision semi --label_distribution_file ../data/fb15k/labelled_train/label_distribution_y6.yml --exclude_default 1 --neg_reward -1 --rho 0.125 --kldiv_lambda 1 --config configs/fb15k_config.yml --exclude_t_ids 2 5 --hidden_unit_list 7 5 5 3 --default_value 0 --train_ml 1 --eval_ml 1 --dir cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df-0_eml-1_ex-2.5_hul-7.5.5.3_k-1_n--1_r-0.125_tml-1 > cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df-0_eml-1_ex-2.5_hul-7.5.5.3_k-1_n--1_r-0.125_tml-1/_LOGS 2>&1 &
pids[7]=$!
$HOME/anaconda3/bin/python /home/cse/phd/csz178057/hpcscratch/Aman_BTP/Interpretable-KBC-tlp/sm/cross_validation.py --folds 5 --labelled_training_data_path cross_val/fb15k/semi --unlabelled_training_data_path ../logs/fb15k/sm_with_id.data.pkl --num_epochs 20 --batch_size 2048 --num_templates 6 --each_input_size 7 --supervision semi --label_distribution_file ../data/fb15k/labelled_train/label_distribution_y6.yml --exclude_default 1 --neg_reward -1 --rho 0.125 --kldiv_lambda 1 --config configs/fb15k_config.yml --exclude_t_ids 2 5 --hidden_unit_list 7 5 5 3 --default_value -0.05 --train_ml 0 --eval_ml 0 --dir cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df--0.05_eml-0_ex-2.5_hul-7.5.5.3_k-1_n--1_r-0.125_tml-0 > cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df--0.05_eml-0_ex-2.5_hul-7.5.5.3_k-1_n--1_r-0.125_tml-0/_LOGS 2>&1 &
pids[8]=$!
$HOME/anaconda3/bin/python /home/cse/phd/csz178057/hpcscratch/Aman_BTP/Interpretable-KBC-tlp/sm/cross_validation.py --folds 5 --labelled_training_data_path cross_val/fb15k/semi --unlabelled_training_data_path ../logs/fb15k/sm_with_id.data.pkl --num_epochs 20 --batch_size 2048 --num_templates 6 --each_input_size 7 --supervision semi --label_distribution_file ../data/fb15k/labelled_train/label_distribution_y6.yml --exclude_default 1 --neg_reward -1 --rho 0.125 --kldiv_lambda 1 --config configs/fb15k_config.yml --exclude_t_ids 2 5 --hidden_unit_list 7 5 5 3 --default_value -0.05 --train_ml 1 --eval_ml 1 --dir cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df--0.05_eml-1_ex-2.5_hul-7.5.5.3_k-1_n--1_r-0.125_tml-1 > cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df--0.05_eml-1_ex-2.5_hul-7.5.5.3_k-1_n--1_r-0.125_tml-1/_LOGS 2>&1 &
pids[9]=$!
$HOME/anaconda3/bin/python /home/cse/phd/csz178057/hpcscratch/Aman_BTP/Interpretable-KBC-tlp/sm/cross_validation.py --folds 5 --labelled_training_data_path cross_val/fb15k/semi --unlabelled_training_data_path ../logs/fb15k/sm_with_id.data.pkl --num_epochs 20 --batch_size 2048 --num_templates 6 --each_input_size 7 --supervision semi --label_distribution_file ../data/fb15k/labelled_train/label_distribution_y6.yml --exclude_default 1 --neg_reward -1 --rho 0.125 --kldiv_lambda 1 --config configs/fb15k_config.yml --exclude_t_ids 2 5 --hidden_unit_list 7 5 5 3 --default_value -0.1 --train_ml 0 --eval_ml 0 --dir cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df--0.1_eml-0_ex-2.5_hul-7.5.5.3_k-1_n--1_r-0.125_tml-0 > cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df--0.1_eml-0_ex-2.5_hul-7.5.5.3_k-1_n--1_r-0.125_tml-0/_LOGS 2>&1 &
pids[10]=$!
$HOME/anaconda3/bin/python /home/cse/phd/csz178057/hpcscratch/Aman_BTP/Interpretable-KBC-tlp/sm/cross_validation.py --folds 5 --labelled_training_data_path cross_val/fb15k/semi --unlabelled_training_data_path ../logs/fb15k/sm_with_id.data.pkl --num_epochs 20 --batch_size 2048 --num_templates 6 --each_input_size 7 --supervision semi --label_distribution_file ../data/fb15k/labelled_train/label_distribution_y6.yml --exclude_default 1 --neg_reward -1 --rho 0.125 --kldiv_lambda 1 --config configs/fb15k_config.yml --exclude_t_ids 2 5 --hidden_unit_list 7 5 5 3 --default_value -0.1 --train_ml 1 --eval_ml 1 --dir cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df--0.1_eml-1_ex-2.5_hul-7.5.5.3_k-1_n--1_r-0.125_tml-1 > cross_val/fb15k/semi/run_1/exp_c-configs.fb15k_config.yml_df--0.1_eml-1_ex-2.5_hul-7.5.5.3_k-1_n--1_r-0.125_tml-1/_LOGS 2>&1 &
pids[11]=$!
for pid in ${pids[*]}; do 
         wait $pid 
done

touch /home/yatin/hpcscratch_copy/texkbc/Interpretable-KBC-tlp/sm/hpc_scripts/gs_fb15k_3runs_rep/JACK_3
