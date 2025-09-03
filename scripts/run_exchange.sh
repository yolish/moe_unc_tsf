export CUDA_VISIBLE_DEVICES=$1
models=("iTransformer" "PatchTST")
#models=("iTransformer" "PatchTST" "DLinear")
root_paths=("./data/long_term_forecast/exchange_rate/")
data_paths=("exchange_rate.csv") 
datasets=("custom")
pred_lengths=(720 336 192 96)
num_experts=(3)
configurations=(1 3)
seeds=(2021)
model_id="test"
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
                        --prob_expert
                    fi
                    if [ $config -eq 3 ]; then
                        if [$pl -eq 96]; then
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
                            --save_outputs \
                            --save_unc 
                        else
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
                            --unc_gating 
                        fi
                    fi
                    
                done
            done    
        done
    done
done
done

