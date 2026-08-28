export CUDA_VISIBLE_DEVICES=$1
# Calibration-method comparison on national_illness (ILI).
#
# Mirrors config 2 of run_tsf_ili.sh (prob_expert, ug0, ftM, sl96/ll48, batch 8,
# model_id="test") and reuses those checkpoints via is_training=0. Every method below is
# run from the same checkpoint in the same process, so the comparison is paired: any
# difference is the calibrator, not the model.
#
# ILI is the interesting case for the adaptive-window method: its validation split is a
# few dozen points, so the hardcoded window of 1000 never fills.
models=("iTransformer")
root_path="./data/long_term_forecast/illness/"
data_path="national_illness.csv"
dataset="custom"
pred_lengths=(24 36 48 60)
num_experts=(1 3)
seeds=(4021 4022 4023 4024 4025)
model_id="test"
enc_in=7
features="M"
seq_len=96
label_len=48
batch_size=8

PY=${PYTHON_BIN:-python}

for seed in "${seeds[@]}"
do
  for model_name in "${models[@]}"
  do
    for pred_len in "${pred_lengths[@]}"
    do
      for ne in ${num_experts[@]}
      do
        echo "=== ILI $model_name pl$pred_len ne$ne seed$seed ==="
        $PY -u run.py \
        --task_name long_term_forecast \
        --is_training 0 \
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
        --enc_in $enc_in \
        --dec_in $enc_in \
        --c_out $enc_in \
        --prob_expert \
        --do_cp_calibration \
        --do_cpvs_calibration \
        --do_aleatoric_only_calibration \
        --do_aleatoric_scale_calibration \
        --do_adaptive_variance_calibration \
        --do_adaptive_window_calibration
      done
    done
  done
done
