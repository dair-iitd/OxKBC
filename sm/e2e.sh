#convert vector to ids and names
python3 map_inverse.py -d fb15k -w ../dumps/fb15k_distmult_dump_norm.pkl -cf ../turk_experiments/turk_which_better_hits10_nothits10_sampled/pred_not_hits_1_test_fb15k_170.txt -of ../turk_experiments/new/pred_not_hits_1_test_fb15k.txt --data_repo_root ../../data

#preprocessing unlabeled train data
python3 preprocessing.py -d fb15k -m distmult -f ../data/fb15k/train.txt -s logs/fb15k/sm_with_id.data -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k -v 1 --t_ids 1 2 3 4 5 6 --data_repo_root ../data --negative_count 2

#preprocessing labelled train data
python3 preprocessing.py -d fb15k -m distmult -f data/fb15k/labelled_train/labelled_train_x.txt -s logs/fb15k/sm_valid_with_id.data -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k -v 1 --t_ids 1 2 3 4 5 6 --data_repo_root ../data --negative_count 0 --y_labels data/fb15k/labelled_train/labelled_train_y6.txt

#create train val split
python create_train_val_split.py --labelled_total_data_path ../logs/fb15k/sm_valid_with_id.data.pkl --total_labels_path ../data/fb15k/labelled_train/labelled_train_y6.txt --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --train_labels_path ../logs/fb15k/sm_sup_train_multilabels.txt --val_data_path ../logs/fb15k/sm_sup_valid_with_id.pkl --val_labels_path ../logs/fb15k/sm_sup_valid_multilabels.txt --train_split 0.8 --seed 242 --num_templates 6

#create splits for cross val
python3 cross_validation.py --gen_data --folds 5 --dir cross_val/fb15k/semi --labelled_training_data_path ../logs/fb15k/sm_valid_with_id.data.pkl --supervision semi --train_labels_path ../data/fb15k/labelled_train/labelled_train_y6.txt --seed 142 


#preprocessing of test data
#hits@1
python3 preprocessing.py -d fb15k -m distmult -f data/fb15k/test/test_hits_1_ordered_x.txt -s logs/fb15k/test_hits1_single_label_sm.data.pkl -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k -v 1 --t_ids 1 2 3 4 5 6 --data_repo_root ../data --negative_count 0 --y_labels data/fb15k/test/test_hits_1_ordered_y.txt

#hits@10
python3 preprocessing.py -d fb15k -m distmult -f data/fb15k/test/test_hits_10_ordered_x.txt -s logs/fb15k/test_hits10_single_label_sm.data.pkl -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k -v 1 --t_ids 1 2 3 4 5 6 --data_repo_root ../data --negative_count 0 --y_labels data/fb15k/test/test_hits_10_ordered_y.txt

#preprocessing of turk data
#hits@1
python3 preprocessing.py -d fb15k -m distmult -f data/fb15k/turk_test/test_hits_1_id_small.txt -s logs/fb15k/turk_test/test_hits1_single_label_sm.data.pkl -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k -v 1 --t_ids 1 2 3 4 5 6 --data_repo_root ../data --negative_count 0

#not hits@1
python3 preprocessing.py -d fb15k -m distmult -f data/fb15k/turk_test/pred_not_hits_1_test_fb15k_small.txt -s logs/fb15k/turk_test/test_not_hits1_single_label_sm.data.pkl -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k -v 1 --t_ids 1 2 3 4 5 6 --data_repo_root ../data --negative_count 0

#hits@10
python3 preprocessing.py -d fb15k -m distmult -f data/fb15k/turk_test/hits10_not_hits1.txt -s logs/fb15k/turk_test/test_hits10_not_hits1_single_label_sm.data.pkl -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k -v 1 --t_ids 1 2 3 4 5 6 --data_repo_root ../data --negative_count 0

# turk second experiment is useful?
python3 preprocessing.py -d fb15k -m distmult -f data/fb15k/turk_test/hits1_not_hits1_mixed.txt -s logs/fb15k/turk_test_useful/hits1_not_hits1_single_label_sm.data.pkl -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k -v 1 --t_ids 1 2 3 4 5 6 --data_repo_root ../data --negative_count 0

###### Semi Supervised exclude 2.5 ##########
train_ml=0
kl=0.0
nr=-2.0
rho=0.125
nt=6
tp=../logs/fb15k/sm_with_id.data.pkl 
ltp=../logs/fb15k/sm_sup_train_with_id.pkl 
vp=../logs/fb15k/sm_sup_valid_with_id.pkl
ldp=../logs/fb15k/label_distribution.yml  
testp=../logs/fb15k/test_hits1_single_label_sm.data.pkl.pkl 
config=configs/fb15k_config.yml 
exclude_t_ids=(2 5)
hidden_unit_list=(90 40)
testlp=../data/fb15k/test/test_hits_1_ordered_y.txt 
vlp1=../logs/fb15k/sm_sup_valid_multilabels.txt 
tlp=../logs/fb15k/sm_sup_train_multilabels.txt
df=-0.05
exdf=1
expname=temp/best_fb15k_kl_${kl}_ml_${train_ml}_nr_${nr}

for i in {1..5}; do
    CUDA_VISIBLE_DEVICES=0 python3 main.py --training_data_path $tp --labelled_training_data_path $ltp --val_data_path $vp --exp_name train --num_epochs 20 --config $config --hidden_unit_list ${hidden_unit_list[@]} --kldiv_lambda $kl --neg_reward $nr --rho $rho --lr 0.001 --cuda --batch_size 2048 --mil --num_templates $nt --each_input_size 7 --supervision semi --output_path ${expname}/run_$i/ --label_distribution_file ${ldp} --exclude_t_ids ${exclude_t_ids[@]} --default_value $df --exclude_default $exdf --train_labels_path $tlp --val_labels_path $vlp1 --eval_ml $train_ml --train_ml $train_ml &;
done

# single label test
for i in {1..5}; do
    python3 main.py --training_data_path $tp --labelled_training_data_path $ltp --val_data_path $testp --exp_name test --num_epochs 20 --config $config --hidden_unit_list ${hidden_unit_list[@]} --kldiv_lambda $kl --neg_reward $nr --rho $rho --lr 0.001 --cuda --batch_size 2048 --mil --num_templates $nt --each_input_size 7 --supervision semi --output_path ${expname}/run_$i/ --checkpoint ${expname}/run_$i/train/train_r${rho}_p1_n${nr}_i4_k${kl}_best_checkpoint.pth0 --only_eval --pred_file preds.txt --log_eval ${expname}/train_x_eval_sl.csv  --label_distribution_file ${ldp} --exclude_t_ids ${exclude_t_ids[@]} --default_value $df --exclude_default $exdf --val_labels_path $testlp --eval_ml 0;
done
for i in {1..5}; do
    python3 main.py --training_data_path $tp --labelled_training_data_path $ltp --val_data_path $testp --exp_name test_ml --num_epochs 20 --config $config --hidden_unit_list ${hidden_unit_list[@]} --kldiv_lambda $kl --neg_reward $nr --rho $rho --lr 0.001 --cuda --batch_size 2048 --mil --num_templates $nt --each_input_size 7 --supervision semi --output_path ${expname}/run_$i/ --checkpoint ${expname}/run_$i/train/train_r${rho}_p1_n${nr}_i4_k${kl}_best_checkpoint.pth0 --exclude_t_ids ${exclude_t_ids[@]} --only_eval --pred_file preds.txt --log_eval ${expname}/train_x_eval_ml.csv --val_labels_path $testlp --label_distribution_file $ldp --default_value $df --exclude_default $exdf --eval_ml 1;
done



######## TURK EVALUATION. GENERATE EXPLANATIONS
#HYPER PARAMETERS OF THE BEST MODEL
train_ml=0
kl=1.0
nr=-1.0
rho=0.125
nt=6
tp=../logs/fb15k/sm_with_id.data.pkl 
ltp=../logs/fb15k/sm_sup_train_with_id.pkl 
vp=../logs/fb15k/sm_sup_valid_with_id.pkl
ldp=../logs/fb15k/label_distribution.yml  
testp=../logs/fb15k/turk_test/test_hits1_single_label_sm.data.pkl.pkl  
pred_file='pred_turk_test_hits1.txt'
config=configs/fb15k_config.yml 
exclude_t_ids=(2 5)
hidden_unit_list=(90 40)
df=-0.05
exdf=1
expname=temp/best_fb15k_kl_${kl}_ml_${train_ml}_nr_${nr}
i=5

python3 main.py --training_data_path $tp --labelled_training_data_path $ltp --val_data_path $testp --exp_name turk_test --num_epochs 20 --config $config --hidden_unit_list ${hidden_unit_list[@]} --kldiv_lambda $kl --neg_reward $nr --rho $rho --lr 0.001 --cuda --batch_size 2048 --mil --num_templates $nt --each_input_size 7 --supervision semi --output_path ${expname}/run_$i/ --checkpoint ${expname}/run_$i/train/train_r${rho}_p1_n${nr}_i4_k${kl}_best_checkpoint.pth0 --exclude_t_ids ${exclude_t_ids[@]} --only_eval --pred_file ${pred_file} --log_eval ${expname}/turk_eval.csv --label_distribution_file $ldp --default_value $df --exclude_default $exdf --eval_ml 0;

testp=../logs/fb15k/turk_test/test_hits10_not_hits1_single_label_sm.data.pkl.pkl 
pred_file='pred_turk_test_hits10_not_hits1.txt'

python3 main.py --training_data_path $tp --labelled_training_data_path $ltp --val_data_path $testp --exp_name turk_test --num_epochs 20 --config $config --hidden_unit_list ${hidden_unit_list[@]} --kldiv_lambda $kl --neg_reward $nr --rho $rho --lr 0.001 --cuda --batch_size 2048 --mil --num_templates $nt --each_input_size 7 --supervision semi --output_path ${expname}/run_$i/ --checkpoint ${expname}/run_$i/train/train_r${rho}_p1_n${nr}_i4_k${kl}_best_checkpoint.pth0 --exclude_t_ids ${exclude_t_ids[@]} --only_eval --pred_file ${pred_file} --log_eval ${expname}/turk_eval.csv --label_distribution_file $ldp --default_value $df --exclude_default $exdf --eval_ml 0;



#### Getting Explanations from rule mining
python3 Answer_triplets.py -o explanation_rule_mining_turk_hits10_not_hits1.pkl -i ../data/fb15k/turk_test/hits10_not_hits1.txt   
python3 Answer_triplets.py -o explanation_rule_mining_turk_hits1.pkl -i ../data/fb15k/turk_test/test_hits_1_id_small.txt 

# Running get_turk_data.py
# python get_turk_data.py -d fb15k -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k -o logs/fb15k/turk_test -tf data/fb15k/turk_test/test_hits_1_id_small.txt -tp logs/fb15k/turk_test/pred_turk_test_hits1.txt -rp Rule-Mining-Distmult/fb15k_rule_mining_tmp/explanation_rule_mining_turk_hits1.pkl --data_repo_root ../data --num 5 --t_ids 1 2 3 4 5 6
# hits@1
python get_turk_data.py -d fb15k -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k -o logs/fb15k/turk_test -tf data/fb15k/turk_test/test_hits_1_id_small.txt -tp logs/fb15k/turk_test/pred_turk_test_hits1.txt -rp logs/fb15k/turk_test/explanation_rule_mining_turk_hits1.pkl --data_repo_root ../data --num 5 --t_ids 1 2 3 4 5 6

# not hits@1
python get_turk_data.py -d fb15k -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k -o logs/fb15k/turk_test/not_hits1/with_qid_after_tut_mixed_rules -tf data/fb15k/turk_test/pred_not_hits_1_test_fb15k_small.txt -tp logs/fb15k/turk_test/pred_turk_test_not_hits1.txt -rp logs/fb15k/turk_test/explanation_rule_mining_turk_not_hits1.pkl --data_repo_root ../data --num 5 --t_ids 1 2 3 4 5 6

# result
python get_turk_res.py -rf logs/fb15k/turk_test/Batch_3760734_batch_results2.csv -op logs/fb15k/turk_test -bf logs/fb15k/turk_test/turk_test_book.html -st majority -thr 0

# Second experiment. Are explanations useful?
python get_turk_useful_data.py -d fb15k -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k -o logs/fb15k/turk_test_useful/first_try -tf data/fb15k/turk_test/hits1_not_hits1_mixed.txt -tp logs/fb15k/turk_test_useful/pred_turk_test_hits1_not_hits1.txt  --data_repo_root ../data --num 5 --t_ids 1 2 3 4 5 6 --y_label data/fb15k/turk_test/hits1_not_hits1_mixed_y.txt


###### Semi Supervised KL = 0  Single Label Train EXCLUDE 2.5##########






# train
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/sm_sup_valid_with_id.pkl --exp_name train_sl --num_epochs 20 --config configs/fb15k_config_90_40.yml --kldiv_lambda 0 --neg_reward -2 --rho 0.125 --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 6 --each_input_size 7 --supervision semi --output_path temp/best_fb15k_semi_sl_ex25/run_$i/ --exclude_t_ids 2 5 &
done

# single label test
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/test_hits1_single_label_sm.data.pkl.pkl --exp_name test --num_epochs 20 --config configs/fb15k_config_90_40.yml --kldiv_lambda 0 --neg_reward -2 --rho 0.125 --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 6 --each_input_size 7 --supervision semi --output_path temp/best_fb15k_semi_sl_ex25/run_$i/ --checkpoint temp/best_fb15k_semi_sl_ex25/run_$i/train_sl/train_sl_r0.125_p1_n-2.0_i4_k0.0_best_checkpoint.pth0 --exclude_t_ids 2 5 --only_eval --pred_file preds.txt --log_eval temp/best_fb15k_semi_sl_ex25/train_sl_eval_sl.csv  ;
done

# mutli label test_hits1_single_label_sm
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/test_hits1_single_label_sm.data.pkl.pkl --exp_name test --num_epochs 20 --config configs/fb15k_config_90_40.yml --kldiv_lambda 0 --neg_reward -2 --rho 0.125 --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 6 --each_input_size 7 --supervision semi --output_path temp/best_fb15k_semi_sl_ex25/run_$i/ --checkpoint temp/best_fb15k_semi_sl_ex25/run_$i/train_sl/train_sl_r0.125_p1_n-2.0_i4_k0.0_best_checkpoint.pth0 --exclude_t_ids 2 5 --only_eval --pred_file preds.txt --log_eval temp/best_fb15k_semi_sl_ex25/train_sl_eval_ml.csv --val_labels_path ../data/fb15k/test/test_hits_1_ordered_y.txt ;
done

###### Semi Supervised KL = 0  Multilabel Train DO NOT exclude 2.5 ##########
kl=0.0
nr=-2.0
rho=0.125
nt=6
expname=temp/best_fb15k_semi_ml
tp=../logs/fb15k/sm_with_id.data.pkl 
ltp=../logs/fb15k/sm_sup_train_with_id.pkl 
vp=../logs/fb15k/sm_sup_valid_with_id.pkl
ldp=../logs/fb15k/label_distribution.yml  
testp=../logs/fb15k/test_hits1_single_label_sm.data.pkl.pkl 
exclude_t_ids=()
testlp=../data/fb15k/test/test_hits_1_ordered_y.txt 
vlp1=../logs/fb15k/sm_sup_valid_multilabels.txt 
tlp=../logs/fb15k/sm_sup_train_multilabels.txt 
df=-0.05
exdf=1

for i in {1..5}; do
    python3 main.py --training_data_path $tp --labelled_training_data_path $ltp --val_data_path $vp --exp_name train --num_epochs 20 --config configs/fb15k_config_90_40.yml --kldiv_lambda $kl --neg_reward $nr --rho $rho --lr 0.001 --cuda --batch_size 2048 --mil --num_templates $nt --each_input_size 7 --supervision semi --output_path ${expname}/run_$i/ --label_distribution_file ${ldp} --exclude_t_ids ${exclude_t_ids[@]} --default_value $df --exclude_default $exdf --train_labels_path $tlp --val_labels_path $vlp1 &;
done

# single label test
for i in {1..5}; do
    python3 main.py --training_data_path $tp --labelled_training_data_path $ltp --val_data_path $testp --exp_name test --num_epochs 20 --config configs/fb15k_config_90_40.yml --kldiv_lambda $kl --neg_reward $nr --rho $rho --lr 0.001 --cuda --batch_size 2048 --mil --num_templates $nt --each_input_size 7 --supervision semi --output_path ${expname}/run_$i/ --checkpoint ${expname}/run_$i/train/train_r${rho}_p1_n${nr}_i4_k${kl}_best_checkpoint.pth0 --only_eval --pred_file preds.txt --log_eval ${expname}/train_x_eval_sl.csv  --label_distribution_file ${ldp} --exclude_t_ids ${exclude_t_ids[@]} --default_value $df --exclude_default $exdf;
done

for i in {1..5}; do
    python3 main.py --training_data_path $tp --labelled_training_data_path $ltp --val_data_path $testp --exp_name test_ml --num_epochs 20 --config configs/fb15k_config_90_40.yml --kldiv_lambda $kl --neg_reward $nr --rho $rho --lr 0.001 --cuda --batch_size 2048 --mil --num_templates $nt --each_input_size 7 --supervision semi --output_path ${expname}/run_$i/ --checkpoint ${expname}/run_$i/train/train_r${rho}_p1_n${nr}_i4_k${kl}_best_checkpoint.pth0 --exclude_t_ids ${exclude_t_ids[@]} --only_eval --pred_file preds.txt --log_eval ${expname}/train_x_eval_ml.csv --val_labels_path $testlp --label_distribution_file $ldp --default_value $df --exclude_default $exdf ;
done



###### Semi Supervised KL = 0  Single Label Train DO NOT EXCLUDE 2.5##########
# train
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/sm_sup_valid_with_id.pkl --exp_name train_sl --num_epochs 20 --config configs/fb15k_config_90_40.yml --kldiv_lambda 0 --neg_reward -2 --rho 0.125 --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 6 --each_input_size 7 --supervision semi --output_path temp/best_fb15k_semi_sl/run_$i/ &
done

# single label test
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/test_hits1_single_label_sm.data.pkl.pkl --exp_name test --num_epochs 20 --config configs/fb15k_config_90_40.yml --kldiv_lambda 0 --neg_reward -2 --rho 0.125 --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 6 --each_input_size 7 --supervision semi --output_path temp/best_fb15k_semi_sl/run_$i/ --checkpoint temp/best_fb15k_semi_sl/run_$i/train_sl/train_sl_r0.125_p1_n-2.0_i4_k0.0_best_checkpoint.pth0  --only_eval --pred_file preds.txt --log_eval temp/best_fb15k_semi_sl/train_sl_eval_sl.csv  ;
done

# mutli label test_hits1_single_label_sm
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/test_hits1_single_label_sm.data.pkl.pkl --exp_name test --num_epochs 20 --config configs/fb15k_config_90_40.yml --kldiv_lambda 0 --neg_reward -2 --rho 0.125 --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 6 --each_input_size 7 --supervision semi --output_path temp/best_fb15k_semi_sl/run_$i/ --checkpoint temp/best_fb15k_semi_sl/run_$i/train_sl/train_sl_r0.125_p1_n-2.0_i4_k0.0_best_checkpoint.pth0 --only_eval --pred_file preds.txt --log_eval temp/best_fb15k_semi_sl/train_sl_eval_ml.csv --val_labels_path ../data/fb15k/test/test_hits_1_ordered_y.txt ;
done


###### Semi Supervised KL = 1  Single Label Train EXCLUDE 6 ##########
# train
kl=1.0
nr=-1.0
rho=0.1
nt=6
expname=temp/best_fb15k_kl_sl_ex6_def0_mean_incldef 
tp=../logs/fb15k/sm_with_id.data.pkl 
ltp=../logs/fb15k/sm_sup_train_with_id.pkl 
vp=../logs/fb15k/sm_sup_valid_with_id.pkl
ldp=../logs/fb15k/label_distribution.yml  
testp=../logs/fb15k/test_hits1_single_label_sm.data.pkl.pkl 
exclude_t_ids=6
vlp=../data/fb15k/test/test_hits_1_ordered_y.txt 
df=0.0
exdf=0

for i in {1..5}; do
    python3 main.py --training_data_path $tp --labelled_training_data_path $ltp --val_data_path $vp --exp_name train_sl --num_epochs 20 --config configs/fb15k_config_90_40.yml --kldiv_lambda $kl --neg_reward $nr --rho $rho --lr 0.001 --cuda --batch_size 2048 --mil --num_templates $nt --each_input_size 7 --supervision semi --output_path ${expname}/run_$i/ --label_distribution_file ${ldp} --exclude_t_ids ${exclude_t_ids} --default_value $df --exclude_default $exdf &;
done

# single label test
for i in {1..5}; do
    python3 main.py --training_data_path $tp --labelled_training_data_path $ltp --val_data_path $testp --exp_name test --num_epochs 20 --config configs/fb15k_config_90_40.yml --kldiv_lambda $kl --neg_reward $nr --rho $rho --lr 0.001 --cuda --batch_size 2048 --mil --num_templates $nt --each_input_size 7 --supervision semi --output_path ${expname}/run_$i/ --checkpoint ${expname}/run_$i/train_sl/train_sl_r${rho}_p1_n${nr}_i4_k${kl}_best_checkpoint.pth0 --only_eval --pred_file preds.txt --log_eval ${expname}/train_x_eval_sl.csv  --label_distribution_file ${ldp} --exclude_t_ids ${exclude_t_ids} --default_value $df --exclude_default $exdf;
done

for i in {1..5}; do
    python3 main.py --training_data_path $tp --labelled_training_data_path $ltp --val_data_path $vp --exp_name test_ml --num_epochs 20 --config configs/fb15k_config_90_40.yml --kldiv_lambda $kl --neg_reward $nr --rho $rho --lr 0.001 --cuda --batch_size 2048 --mil --num_templates $nt --each_input_size 7 --supervision semi --output_path ${expname}/run_$i/ --checkpoint ${expname}/run_$i/train_sl/train_sl_r${rho}_p1_n${nr}_i4_k${kl}_best_checkpoint.pth0 --exclude_t_ids ${exclude_t_ids} --only_eval --pred_file preds.txt --log_eval ${expname}/train_x_eval_ml.csv --val_labels_path $vlp --label_distribution_file $ldp --default_value $df --exclude_default $exdf ;
done


###### USING OLD DATA Semi Supervised KL = 1  Single Label Train EXCLUDE 6 ##########
# train
kl=1.0
nr=-1.0
rho=0.1
nt=5
expname=temp/best_fb15k_kl_old_default0_meansubset
tp=../../Interpretable-KBC/logs/fb15k/sm_with_id.data.pkl
ltp=../../Interpretable-KBC/logs/fb15k/exp_words/sm_sup_train_with_id.pkl
vp=../../Interpretable-KBC/logs/fb15k/exp_words/sm_sup_valid_with_id.pkl
ldp=../../Interpretable-KBC/logs/fb15k/exp_words/label_distribution.yml 
testp=../../Interpretable-KBC/logs/fb15k/test_exp/test_hits1_single_label_sm.data.pkl

for i in {1..5}; do
    python3 main.py --training_data_path $tp --labelled_training_data_path $ltp --val_data_path $vp --exp_name train_sl --num_epochs 20 --config configs/fb15k_config_90_40.yml --kldiv_lambda $kl --neg_reward $nr --rho $rho --lr 0.001 --cuda --batch_size 2048 --mil --num_templates $nt --each_input_size 7 --supervision semi --output_path ${expname}/run_$i/ --label_distribution_file ${ldp} &;
done

# single label test
for i in {1..5}; do
    python3 main.py --training_data_path $tp --labelled_training_data_path $ltp --val_data_path $testp --exp_name test --num_epochs 20 --config configs/fb15k_config_90_40.yml --kldiv_lambda $kl --neg_reward $nr --rho $rho --lr 0.001 --cuda --batch_size 2048 --mil --num_templates $nt --each_input_size 7 --supervision semi --output_path ${expname}/run_$i/ --checkpoint ${expname}/run_$i/train_sl/train_sl_r${rho}_p1_n${nr}_i4_k${kl}_best_checkpoint.pth0 --only_eval --pred_file preds.txt --log_eval ${expname}/train_sl_eval_sl.csv  --label_distribution_file ${ldp} ;
done

# mutli label test_hits1_single_label_sm
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/test_hits1_single_label_sm.data.pkl.pkl --exp_name test --num_epochs 20 --config configs/fb15k_config_90_40.yml --kldiv_lambda $kl --neg_reward $nr --rho $rho --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 6 --each_input_size 7 --supervision semi --output_path temp/best_fb15k_semi_sl_kl_ex6/run_$i/ --checkpoint temp/best_fb15k_semi_sl_kl_ex6/run_$i/train_sl/train_sl_r${rho}_p1_n${nr}_i4_k${kl}_best_checkpoint.pth0 --exclude_t_ids 6 --only_eval --pred_file preds.txt --log_eval temp/best_fb15k_semi_sl_kl_ex6/train_sl_eval_ml.csv --val_labels_path ../data/fb15k/test/test_hits_1_ordered_y.txt --label_distribution_file ../logs/fb15k/label_distribution.yml ;
done






############ Unsupervised ###########

# train
# export valid_path="temp-sm_sup_valid_with_id.pkl"
export valid_path="sm_sup_valid_with_id.pkl"
for i in {1..5}; do
    mkdir -p temp/best_fb15k_un/run_$i; 
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --val_data_path ../logs/fb15k/$valid_path --exp_name train --num_epochs 20 --config cross_val/fb15k_90_40_un/run_1/exp_-2_0.yml --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --supervision un --output_path temp/best_fb15k_un/run_$i/ &
done

# single label test
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --val_data_path ../logs/fb15k/test_exp/test_hits1_single_label_sm.data.pkl --exp_name test --num_epochs 20 --config cross_val/fb15k_90_40_un/run_1/exp_-2_0.yml --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --supervision un --output_path temp/best_fb15k_un/run_$i/ --only_eval --pred_file preds.txt --checkpoint temp/best_fb15k_un/run_$i/train/train_r0_p1_n-2_i4_best_checkpoint.pth0
done


######### Semi supervised KL != 0 ###########

# train
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/sm_sup_valid_with_id.pkl --exp_name train --num_epochs 20 --config cross_val/fb15k_90_40_semi_kl_1/run_1/exp_-1_0.1.yml --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --supervision semi --output_path temp/best_fb15k_kl/run_$i/ --kldiv_lambda 1 --label_distribution_file ../logs/fb15k/label_distribution.yml &
done

# single label test
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/test_exp/test_hits1_single_label_sm.data.pkl --exp_name test --num_epochs 20 --config cross_val/fb15k_90_40_semi_kl_1/run_1/exp_-1_0.1.yml --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --supervision semi --output_path temp/best_fb15k_kl/run_$i/ --kldiv_lambda 1 --label_distribution_file ../logs/fb15k/label_distribution.yml --checkpoint temp/best_fb15k_kl/run_$i/train/train_r0.1_p1_n-1_i4_best_checkpoint.pth0 --only_eval --pred_file preds.txt    
done



######## Supervised KL = 0 ##########

# train
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/sm_sup_valid_with_id.pkl --exp_name train --num_epochs 2000 --config cross_val/fb15k_90_40_semi/run_1/exp_-2_0.125.yml --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --supervision sup --output_path temp/fb15k_sup/run_$i/ &
done

# single label test
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/test_exp/test_hits1_single_label_sm.data.pkl --exp_name test --num_epochs 2000 --config cross_val/fb15k_90_40_semi/run_1/exp_-2_0.125.yml --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --supervision sup --output_path temp/fb15k_sup/run_$i/ --checkpoint temp/fb15k_sup/run_$i/train/train_r0.125_p1_n-2_i4_best_checkpoint.pth0 --only_eval --pred_file preds.txt;
done

### Ablation KL=1 1/3 of train data #####

## Train ##
for i in {1..5}; do
    mkdir -p temp/ablation_test/1_3/run_$i
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/ablation_test/sm_sup_train_1_3_with_id.pkl --val_data_path ../logs/fb15k/sm_sup_valid_with_id.pkl --exp_name train --num_epochs 20 --config cross_val/fb15k_90_40_semi_kl_1/run_1/exp_-1_0.1.yml --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --supervision semi --output_path temp/ablation_test/1_3/run_$i/ --kldiv_lambda 1 --label_distribution_file ../logs/fb15k/label_distribution.yml &
done

### TEST hits 1###
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/ablation_test/sm_sup_train_1_3_with_id.pkl --val_data_path ../logs/fb15k/test_exp/test_hits1_single_label_sm.data.pkl --exp_name test_hits1 --num_epochs 20 --config cross_val/fb15k_90_40_semi_kl_1/run_1/exp_-1_0.1.yml --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --supervision semi --output_path temp/ablation_test/1_3/run_$i/ --kldiv_lambda 1 --label_distribution_file ../logs/fb15k/label_distribution.yml --checkpoint temp/ablation_test/1_3/run_$i/train/train_r0.1_p1_n-1_i4_best_checkpoint.pth0 --only_eval --pred_file preds.txt
done


### Test hits10 ##
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/ablation_test/sm_sup_train_1_3_with_id.pkl --val_data_path ../logs/fb15k/test_exp/hits10/test_hits10_sm.data.pkl --exp_name test_hits10 --num_epochs 20 --config cross_val/fb15k_90_40_semi_kl_1/run_1/exp_-1_0.1.yml --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --supervision semi --output_path temp/ablation_test/1_3/run_$i/ --kldiv_lambda 1 --label_distribution_file ../logs/fb15k/label_distribution.yml --checkpoint temp/ablation_test/1_3/run_$i/train/train_r0.1_p1_n-1_i4_best_checkpoint.pth0 --only_eval --pred_file preds.txt
done




### Ablation KL=1 2/3 of train data #####

## Train ##
for i in {1..5}; do
    mkdir -p temp/ablation_test/2_3/run_$i
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/ablation_test/sm_sup_train_2_3_with_id.pkl --val_data_path ../logs/fb15k/sm_sup_valid_with_id.pkl --exp_name train --num_epochs 20 --config cross_val/fb15k_90_40_semi_kl_1/run_1/exp_-1_0.1.yml --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --supervision semi --output_path temp/ablation_test/2_3/run_$i/ --kldiv_lambda 1 --label_distribution_file ../logs/fb15k/label_distribution.yml &
done

### TEST hits 1###
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/ablation_test/sm_sup_train_2_3_with_id.pkl --val_data_path ../logs/fb15k/test_exp/test_hits1_single_label_sm.data.pkl --exp_name test_hits1 --num_epochs 20 --config cross_val/fb15k_90_40_semi_kl_1/run_1/exp_-1_0.1.yml --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --supervision semi --output_path temp/ablation_test/2_3/run_$i/ --kldiv_lambda 1 --label_distribution_file ../logs/fb15k/label_distribution.yml --checkpoint temp/ablation_test/2_3/run_$i/train/train_r0.1_p1_n-1_i4_best_checkpoint.pth0 --only_eval --pred_file preds.txt
done


### Test hits10 ##
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/ablation_test/sm_sup_train_2_3_with_id.pkl --val_data_path ../logs/fb15k/test_exp/hits10/test_hits10_sm.data.pkl --exp_name test_hits10 --num_epochs 20 --config cross_val/fb15k_90_40_semi_kl_1/run_1/exp_-1_0.1.yml --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --supervision semi --output_path temp/ablation_test/2_3/run_$i/ --kldiv_lambda 1 --label_distribution_file ../logs/fb15k/label_distribution.yml --checkpoint temp/ablation_test/2_3/run_$i/train/train_r0.1_p1_n-1_i4_best_checkpoint.pth0 --only_eval --pred_file preds.txt
done


