export CUDA_VISIBLE_DEVICES=$1
models=("iTransformer")
#models=("iTransformer" "PatchTST" "DLinear")
#root_paths=("./data/long_term_forecast/ETT/" "./data/long_term_forecast/ETT/")
root_paths=("./data/long_term_forecast/ETT/" "./data/long_term_forecast/ETT/" "./data/long_term_forecast/ETT/" "./data/long_term_forecast/ETT/")
#data_paths=("ETTm1.csv" "ETTh1.csv") 
#datasets=("ETTm1" "ETTh1")
data_paths=("ETTh2.csv" "ETTm2.csv" "ETTm1.csv" "ETTh1.csv") 
datasets=("ETTh2" "ETTm2" "ETTm1" "ETTh1")
# Optional $2: space-separated dataset indices into the arrays above, so the grid
# can be split across GPUs (e.g. "0 1" on one card, "2 3" on another).
dataset_idx=(${2:-0 1 2 3})
#pred_lengths=(720 336 192 96)
pred_lengths=(336)
num_experts=(1 2 3 4 5)
# 1 = MoE (deterministic experts, pe0/ug0), 2 = MoG (prob_expert, pe1/ug0),
# 4 = CQR (split-conformal quantile), 5 = CQR with periodic retraining.
# Config 3 (MoGU, unc_gating/ug1) is deliberately excluded from this pl336 grid.
configurations=(1 2 4 5)
#seeds=(2351 2352 2353 2354 2355)
seeds=(4021 4022 4023 4024 4025)
model_id="test"
# CQR trains a different head (2*enc_in outputs), so it gets its own model_id to
# avoid colliding with the deterministic checkpoints under model_id="test".
cqr_model_id="cqr"
features="M"
seq_len=96
label_len=48
batch_size=8


for seed in "${seeds[@]}"
do
  for model_name in "${models[@]}"
  do
    for i in "${dataset_idx[@]}"
    do
      for pred_len in "${pred_lengths[@]}"
      do
        root_path=${root_paths[$i]}
        data_path=${data_paths[$i]}
        dataset=${datasets[$i]}
        for ne in ${num_experts[@]}
            do
            for config in "${configurations[@]}"
              do
                    
                    if [ $config -eq 1 ]; then
                        echo "python -u run.py --task_name long_term_forecast --root_path $root_path --data_path $data_path --model $model_name --data $dataset --pred_len $pred_len --num_experts $ne --do_cp_calibration --seed $seed"
                        python -u run.py \
                        --task_name long_term_forecast \
                        --is_training 1 \
                        --root_path $root_path \
                        --data_path $data_path \
                        --model_id $model_id \
                        --model $model_name \
                        --data $dataset \
                        --features $features \
                        --seq_len $seq_len \
                        --label_len $label_len \
                        --batch_size $batch_size \
                        --pred_len $pred_len \
                        --seed $seed \
                        --num_experts $ne \
                        --overwrite \
                        --do_cp_calibration
                    fi
                    if [ $config -eq 2 ]; then
                        echo "python -u run.py --task_name long_term_forecast --root_path $root_path --data_path $data_path --model $model_name --data $dataset --pred_len $pred_len --num_experts $ne --prob_expert --do_cp_calibration --do_cpvs_calibration --seed $seed --do_aleatoric_mog_calibration --do_aleatoric_mog_calibration_second_option --do_aleatoric_only_calibration"
                        python -u run.py \
                        --task_name long_term_forecast \
                        --is_training 1 \
                        --root_path $root_path \
                        --data_path $data_path \
                        --model_id $model_id \
                        --model $model_name \
                        --data $dataset \
                        --features $features \
                        --seq_len $seq_len \
                        --label_len $label_len \
                        --batch_size $batch_size \
                        --pred_len $pred_len \
                        --seed $seed \
                        --num_experts $ne \
                        --prob_expert \
                        --overwrite \
                        --do_cp_calibration \
                        --do_cpvs_calibration \
                        --do_aleatoric_mog_calibration \
                        --do_aleatoric_mog_calibration_second_option \
                        --do_aleatoric_only_calibration \
                        --do_aleatoric_scale_calibration \
                        --do_adaptive_variance_calibration \
                        --do_adaptive_window_calibration \
                        --do_cp_dvs_calibration \
                        --do_moecp_calibration
                    fi
                    if [ $config -eq 3 ]; then
                        if [ $pred_len -eq 96 ]; then
                            echo "python -u run.py --task_name long_term_forecast --root_path $root_path --data_path $data_path --model $model_name --data $dataset --pred_len $pred_len --num_experts $ne --prob_expert --unc_gating --do_cp_calibration --do_cpvs_calibration --seed $seed --do_aleatoric_mog_calibration --do_aleatoric_mog_calibration_second_option --do_aleatoric_only_calibration"
                            python -u run.py \
                            --task_name long_term_forecast \
                            --is_training 1 \
                            --root_path $root_path \
                            --data_path $data_path \
                            --model_id $model_id \
                            --model $model_name \
                            --data $dataset \
                            --features $features \
                            --seq_len $seq_len \
                            --batch_size $batch_size \
                            --label_len $label_len \
                            --pred_len $pred_len \
                            --seed $seed \
                            --num_experts $ne \
                            --prob_expert \
                            --unc_gating \
                            --do_cp_calibration \
                            --do_cpvs_calibration \
                            --save_outputs \
                            --save_unc \
                            --save_expert_outputs \
                            --do_aleatoric_mog_calibration \
                            --do_aleatoric_mog_calibration_second_option \
                            --do_aleatoric_only_calibration
                        else
                         echo "python -u run.py --task_name long_term_forecast --root_path $root_path --data_path $data_path --model $model_name --data $dataset --pred_len $pred_len --num_experts $ne --prob_expert --unc_gating --do_cp_calibration --do_cpvs_calibration --seed $seed --do_aleatoric_mog_calibration --do_aleatoric_mog_calibration_second_option --do_aleatoric_only_calibration"
                            python -u run.py \
                            --task_name long_term_forecast \
                            --is_training 1 \
                            --root_path $root_path \
                            --data_path $data_path \
                            --model_id $model_id \
                            --model $model_name \
                            --data $dataset \
                            --features $features \
                            --seq_len $seq_len \
                            --batch_size $batch_size \
                            --label_len $label_len \
                            --pred_len $pred_len \
                            --seed $seed \
                            --num_experts $ne \
                            --prob_expert \
                            --unc_gating \
                            --do_cp_calibration \
                            --do_cpvs_calibration \
                            --do_aleatoric_mog_calibration \
                            --do_aleatoric_mog_calibration_second_option \
                            --do_aleatoric_only_calibration
                        fi
                    fi
                    if [ $config -eq 4 ]; then
                        echo "python -u run.py --task_name long_term_forecast --root_path $root_path --data_path $data_path --model $model_name --data $dataset --pred_len $pred_len --num_experts $ne --use_quantile_loss --do_cqr_calibration --seed $seed"
                        python -u run.py \
                        --task_name long_term_forecast \
                        --is_training 1 \
                        --root_path $root_path \
                        --data_path $data_path \
                        --model_id $cqr_model_id \
                        --model $model_name \
                        --data $dataset \
                        --features $features \
                        --seq_len $seq_len \
                        --label_len $label_len \
                        --batch_size $batch_size \
                        --pred_len $pred_len \
                        --seed $seed \
                        --num_experts $ne \
                        --use_quantile_loss \
                        --do_cqr_calibration
                    fi
                    if [ $config -eq 5 ]; then
                        # Configs 4 and 5 build an identical "setting" string (same
                        # model_id="cqr", same arch), so is_training=1 here would hit the
                        # results/{setting} skip-check that config 4 just satisfied and do
                        # nothing. Evaluate-only reuses config 4's checkpoint and always
                        # runs calibrate_cqr_retrain, which has no skip-check.
                        echo "python -u run.py --task_name long_term_forecast --root_path $root_path --data_path $data_path --model $model_name --data $dataset --pred_len $pred_len --num_experts $ne --use_quantile_loss --do_cqr_retrain_calibration --seed $seed"
                        python -u run.py \
                        --task_name long_term_forecast \
                        --is_training 0 \
                        --root_path $root_path \
                        --data_path $data_path \
                        --model_id $cqr_model_id \
                        --model $model_name \
                        --data $dataset \
                        --features $features \
                        --seq_len $seq_len \
                        --label_len $label_len \
                        --batch_size $batch_size \
                        --pred_len $pred_len \
                        --seed $seed \
                        --num_experts $ne \
                        --use_quantile_loss \
                        --do_cqr_retrain_calibration
                    fi
                done
            done    
        done
    done
done
done

