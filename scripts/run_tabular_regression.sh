#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

data_sets=("Synthetic" "Bike" "Temperature")
calibrators=("moecp" "cp_vs")
seeds=(4021)
model_id="tabular_exp"
batch_size=128

for seed in "${seeds[@]}"
do
    for data in "${data_sets[@]}"
    do
        if [ "$data" == "Synthetic" ]; then
            enc_in=1
        elif [ "$data" == "Bike" ]; then
            enc_in=12
        elif [ "$data" == "Temperature" ]; then
            enc_in=21
        fi

        for cal in "${calibrators[@]}"
        do
            echo "Running: Data=$data, Calibrator=$cal, enc_in=$enc_in, Seed=$seed"
            
            python -u run.py \
              --task_name tabular_regression \
              --is_training 1 \
              --root_path ./dataset/ \
              --data_path "${data}.csv" \
              --data "$data" \
              --model MoE \
              --model_id "${model_id}_${data}_${cal}" \
              --enc_in $enc_in \
              --c_out 1 \
              --num_experts 3 \
              --batch_size $batch_size \
              --train_epochs 20 \
              --learning_rate 0.001 \
              --calibrator "$cal" \
              --tau 100 \
              --patience 5 \
              --seed $seed
        done
    done
done