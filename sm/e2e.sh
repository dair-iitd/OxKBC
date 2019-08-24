#preprocessing unlabeled train data
python3 preprocessing.py -d fb15k -m distmult -f ../data/fb15k/train.txt -s logs/fb15k/sm_with_id.data -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k -v 1 --t_ids 1 2 3 4 5 6 --data_repo_root ../data --negative_count 2

#preprocessing labelled train data
python3 preprocessing.py -d fb15k -m distmult -f data/fb15k/labelled_train/labelled_train_x.txt -s logs/fb15k/sm_vaild_with_id.data -w dumps/fb15k_distmult_dump_norm.pkl -l logs/fb15k -v 1 --t_ids 1 2 3 4 5 6 --data_repo_root ../data --negative_count 0 --y_labels data/fb15k/labelled_train/labelled_train_y6.txt

#create train val split
python create_train_val_split.py --labelled_total_data_path ../logs/fb15k/sm_valid_with_id.data.pkl --total_labels_path ../data/fb15k/labelled_train/labelled_train_y6.txt --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --train_labels_path ../logs/fb15k/sm_sup_train_multilabels.txt --val_data_path ../logs/fb15k/sm_sup_valid_with_id.pkl --val_labels_path ../logs/fb15k/sm_sup_valid_multilabels.txt --train_split 0.8 --seed 42 --num_templates 6


###### Semi Supervised KL = 0  Multilabel Train  ##########

# train
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/sm_sup_valid_with_id.pkl --exp_name train_ml --num_epochs 20 --config configs/fb15k_config_90_40.yml --kldiv_lambda 0 --neg_reward -2 --rho 0.125 --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 6 --each_input_size 7 --supervision semi --output_path temp/best_fb15k/run_$i/ --val_labels_path ../logs/fb15k/sm_sup_valid_multilabels.txt --train_labels_path ../logs/fb15k/sm_sup_train_multilabels.txt &
done

# single label test
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/sm_sup_valid_with_id.pkl --exp_name test --num_epochs 20 --config configs/fb15k_config_90_40.yml --kldiv_lambda 0 --neg_reward -2 --rho 0.125 --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --supervision semi --output_path temp/best_fb15k/run_$i/ --checkpoint temp/best_fb15k/run_$i/train_ml/train_ml_r0.125_p1_n-2.0_i4_k0.0_best_checkpoint.pth0 --only_eval --pred_file preds.txt ;
done

# mutli label test
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/sm_sup_valid_with_id.pkl --val_labels_path ../logs/fb15k/sm_sup_valid_multilabels.txt  --exp_name test_ml --num_epochs 20 --config configs/fb15k_config_90_40.yml --kldiv_lambda 0 --neg_reward -2 --rho 0.125 --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 6 --each_input_size 7 --supervision semi --output_path temp/best_fb15k/run_$i/ --checkpoint temp/best_fb15k/run_$i/train_ml_r0.125_p1_n-2.0_i4_k0.0/train_ml_r0.125_p1_n-2.0_i4_k0.0_best_checkpoint.pth0 --only_eval --pred_file preds.txt;
done



###### Semi Supervised KL = 0 ##########

# train
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/sm_sup_valid_with_id.pkl --exp_name train --num_epochs 20 --config cross_val/fb15k_90_40_semi/run_1/exp_-2_0.125.yml --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --supervision semi --output_path temp/best_fb15k/run_$i/ & done 
# single label test
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/test_exp/test_hits1_single_label_sm.data.pkl --exp_name test --num_epochs 20 --config cross_val/fb15k_90_40_semi/run_1/exp_-2_0.125.yml --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --supervision semi --output_path temp/best_fb15k/run_$i/ --checkpoint temp/best_fb15k/run_$i/train/train_r0.125_p1_n-2_i4_best_checkpoint.pth0 --only_eval --pred_file preds.txt;
    
done

# mutli label test
for i in {1..5}; do
    python3 main.py --training_data_path ../logs/fb15k/sm_with_id.data.pkl --labelled_training_data_path ../logs/fb15k/sm_sup_train_with_id.pkl --val_data_path ../logs/fb15k/test_exp/test_hits1_single_label_sm.data.pkl --val_labels_path  ../logs/fb15k/test_exp/test_y_true_hits1.txt --exp_name test_ml --num_epochs 20 --config cross_val/fb15k_90_40_semi/run_1/exp_-2_0.125.yml --lr 0.001 --cuda --batch_size 2048 --mil --num_templates 5 --each_input_size 7 --supervision semi --output_path temp/best_fb15k/run_$i/ --checkpoint temp/best_fb15k/run_$i/train/train_r0.125_p1_n-2_i4_best_checkpoint.pth0 --only_eval --pred_file preds.txt;
    
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


