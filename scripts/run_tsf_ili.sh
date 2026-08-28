export CUDA_VISIBLE_DEVICES=$1
#models=("iTransformer" "PatchTST")
models=("DLinear" "iTransformer" "PatchTST")
root_paths=("./data/long_term_forecast/illness/")
data_paths=("national_illness.csv") 
datasets=("custom")
pred_lengths=(24 36 48 60)
num_experts=(1 3)
configurations=(1 2 3 4)
seeds=(4021 4022 4023 4024 4025)
model_id="test"
# CQR trains a different head (2*enc_in outputs), so it gets its own model_id to
# keep its checkpoints/results separate from the deterministic/prob runs above.
cqr_model_id="cqr"
enc_in=7   # national_illness.csv has 7 feature columns (date excluded)
features="M"
seq_len=96
label_len=48
batch_size=8


length=${#root_paths[@]}
for seed in "${seeds[@]}"
do
  for model_name in "${models[@]}"
  do
    for ((i=0; i<$length; i++))
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
                        echo "python -u run.py --task_name long_term_forecast --root_path $root_path --data_path $data_path --model $model_name --data $dataset --pred_len $pred_len --num_experts $ne"
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
                        --num_experts $ne 
                    fi
                    if [ $config -eq 2 ]; then
                        echo "python -u run.py --task_name long_term_forecast --root_path $root_path --data_path $data_path --model $model_name --data $dataset --pred_len $pred_len --num_experts $ne --prob_expert --do_aleatoric_mog_calibration --do_aleatoric_mog_calibration_second_option --do_aleatoric_only_calibration"
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
                        --max_grad_norm 1 \
                        --do_aleatoric_mog_calibration \
                        --do_aleatoric_mog_calibration_second_option \
                        --do_aleatoric_only_calibration
                    fi
                    if [ $config -eq 3 ]; then
                        echo "python -u run.py --task_name long_term_forecast --root_path $root_path --data_path $data_path --model $model_name --data $dataset --pred_len $pred_len --num_experts $ne --prob_expert --unc_gating --do_aleatoric_mog_calibration --do_aleatoric_mog_calibration_second_option --do_aleatoric_only_calibration"
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
                        --max_grad_norm 1 \
                        --do_aleatoric_mog_calibration \
                        --do_aleatoric_mog_calibration_second_option \
                        --do_aleatoric_only_calibration
                    fi
                    if [ $config -eq 4 ]; then
                        # CQR needs the plain (non-MoE, non-probabilistic) path: the quantile
                        # head is only wired into the single-expert branch. Run it once per
                        # (model, pred_len, seed) instead of once per num_experts value.
                        if [ $ne -eq ${num_experts[0]} ]; then
                            echo "python -u run.py --task_name long_term_forecast --root_path $root_path --data_path $data_path --model $model_name --data $dataset --pred_len $pred_len --num_experts 1 --use_quantile_loss --do_cqr_calibration"
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
                            --num_experts 1 \
                            --enc_in $enc_in \
                            --dec_in $enc_in \
                            --c_out $enc_in \
                            --use_quantile_loss \
                            --do_cqr_calibration
                        fi
                    fi

                done
            done    
        done
    done
done
done

