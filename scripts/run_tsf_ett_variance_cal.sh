export CUDA_VISIBLE_DEVICES=$1
# Aleatoric Scale + Adaptive Variance calibration on the ETT grid.
#
# Mirrors config 2 of run_tsf_ett.sh exactly (iTransformer, prob_expert, ug0, ftM,
# sl96/ll48/pl96, batch 8, model_id="test"), so it reuses those checkpoints rather than
# retraining: is_training=0 loads checkpoint.pth for the same setting string. Both
# calibrators need the MoG aleatoric/epistemic split, hence --prob_expert.
models=("iTransformer")
root_paths=("./data/long_term_forecast/ETT/" "./data/long_term_forecast/ETT/" "./data/long_term_forecast/ETT/" "./data/long_term_forecast/ETT/")
data_paths=("ETTh2.csv" "ETTm2.csv" "ETTm1.csv" "ETTh1.csv")
datasets=("ETTh2" "ETTm2" "ETTm1" "ETTh1")
# Optional $2: space-separated dataset indices into the arrays above, so the grid
# can be split across GPUs (e.g. "0 1" on one card, "2 3" on another).
dataset_idx=(${2:-0 1 2 3})
pred_lengths=(96)
num_experts=(1 2 3 4 5)
seeds=(4021 4022 4023 4024 4025)
model_id="test"
features="M"
seq_len=96
label_len=48
batch_size=8

PY=${PYTHON_BIN:-python}

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
          echo "python -u run.py --task_name long_term_forecast --root_path $root_path --data_path $data_path --model $model_name --data $dataset --pred_len $pred_len --num_experts $ne --prob_expert --seed $seed --do_aleatoric_scale_calibration --do_adaptive_variance_calibration"
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
          --prob_expert \
          --do_aleatoric_scale_calibration \
          --do_adaptive_variance_calibration
        done
      done
    done
  done
done
