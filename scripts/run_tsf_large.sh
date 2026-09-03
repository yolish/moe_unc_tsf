export CUDA_VISIBLE_DEVICES=$1
models=("iTransformer")
root_paths=("./data/long_term_forecast/electricity/" "./data/long_term_forecast/weather/" )
data_paths=("electricity.csv" "weather.csv") 
datasets=("custom" "custom")
pred_lengths=(96)
num_experts=(1 3)
# config 2 = MoGE trunk (CP-MoG, CP-MoG-a, CP-fixed); config 4 = pinball trunk (both CQR variants).
configurations=(2 4)
seeds=(4021 4022 4023 4024 4025)
model_id="test"
# CQR trains a separate pinball-loss network; a distinct model_id keeps its
# checkpoint from colliding with the ne=1 non-probabilistic one (same ne/pe/ug).
cqr_model_id="cqr"
features="M"
seq_len=96
label_len=48
batch_size=4


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
                        --learning_rate 0.001 \
                        --num_experts $ne 
                    fi
                    if [ $config -eq 2 ]; then
                        echo "python -u run.py --task_name long_term_forecast --root_path $root_path --data_path $data_path --model $model_name --data $dataset --pred_len $pred_len --num_experts $ne --prob_expert"
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
                        --learning_rate 0.001 \
                        --max_grad_norm 1 \
                        --aci_alpha 0.1 \
                        --aci_gamma 0.001 \
                        --do_aci_cpvs_calibration \
                        --do_aci_aleatoric_scale_g001_calibration \
                        --do_aci_cp_calibration
                    fi
                    if [ $config -eq 3 ]; then
                        echo "python -u run.py --task_name long_term_forecast --root_path $root_path --data_path $data_path --model $model_name --data $dataset --pred_len $pred_len --num_experts $ne --prob_expert --unc_gating"
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
                        --learning_rate 0.001 \
                        --max_grad_norm 1
                     fi
                    # CQR is a single-expert trunk in the paper: the reported CQR columns
                    # are read from the ne=1 table for every dataset, so an ne=3 CQR run
                    # would train for hours and feed no cell.
                    if [ $config -eq 4 ] && [ $ne -eq 1 ]; then
                        echo "python -u run.py ... --data $dataset --data_path $data_path --pred_len $pred_len --use_quantile_loss --do_aci_cqr_calibration --do_aci_cqr_retrain_calibration --seed $seed"
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
                        --aci_alpha 0.1 \
                        --aci_gamma 0.001 \
                        --do_aci_cqr_calibration \
                        --do_aci_cqr_retrain_calibration
                    fi
                    
                done
            done    
        done
    done
done
done

