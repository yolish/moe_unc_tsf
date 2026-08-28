#!/bin/bash
# CQR-retrain only, deliberately run last: it's the slow method (dozens of model refits
# per dataset), so it must not sit ahead of faster methods in the queue and block them.
# Covers every retrain config still outstanding: ne3 electricity+traffic, and all 9
# datasets at ne1.
#
#   bash scripts/run_aci_cqr_retrain_all.sh [gpu]
set -u
cd "$(dirname "$0")/.."
GPU=${1:-0}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/aci_cqr_retrain_all
mkdir -p "$LOGDIR"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
    local tag=$1; shift
    local log="${LOGDIR}/${tag}_${STAMP}.log"
    echo "[$(date +%T)] START $tag -> $log"
    local t0=$SECONDS
    $PY -u run.py "$@" > "$log" 2>&1
    local rc=$?
    echo "[$(date +%T)] END   $tag rc=$rc ($((SECONDS - t0))s)"
    grep -E 'Coverage:|Avg Width|Retrains:|Final alpha_t' "$log" | tail -4
}

RETRAIN_NE3="--task_name long_term_forecast --model_id cqr --model iTransformer \
     --features M --seq_len 96 --label_len 48 --pred_len 96 --seed 4021 --num_experts 3 \
     --is_training 0 --use_quantile_loss --aci_gamma 0.001 --aci_alpha 0.1 \
     --do_aci_cqr_retrain_calibration"

RETRAIN_NE1="--task_name long_term_forecast --model_id cqr --model iTransformer \
     --features M --seq_len 96 --label_len 48 --seed 4021 --num_experts 1 \
     --is_training 0 --use_quantile_loss --aci_gamma 0.001 --aci_alpha 0.1 \
     --do_aci_cqr_retrain_calibration"

ETT="./data/long_term_forecast/ETT/"

# --- ne1 retrain, small/medium datasets ---
for ds in ETTh1 ETTh2 ETTm1 ETTm2; do
    run "ne1_${ds,,}" $RETRAIN_NE1 --pred_len 96 --batch_size 8 \
        --root_path $ETT --data_path ${ds}.csv --data $ds \
        --enc_in 7 --dec_in 7 --c_out 7
done
run ne1_weather $RETRAIN_NE1 --pred_len 96 --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/weather/ --data_path weather.csv \
    --enc_in 21 --dec_in 21 --c_out 21
run ne1_exchange $RETRAIN_NE1 --pred_len 96 --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/exchange_rate/ --data_path exchange_rate.csv \
    --enc_in 8 --dec_in 8 --c_out 8
run ne1_illness $RETRAIN_NE1 --pred_len 24 --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/illness/ --data_path national_illness.csv \
    --enc_in 7 --dec_in 7 --c_out 7

# --- ne3 + ne1 retrain, electricity/traffic (the slowest of all: large channel count AND retrain) ---
run ne3_electricity $RETRAIN_NE3 --batch_size 4 --data custom \
    --root_path ./data/long_term_forecast/electricity/ --data_path electricity.csv \
    --enc_in 321 --dec_in 321 --c_out 321
run ne1_electricity $RETRAIN_NE1 --pred_len 96 --batch_size 4 --data custom \
    --root_path ./data/long_term_forecast/electricity/ --data_path electricity.csv \
    --enc_in 321 --dec_in 321 --c_out 321

run ne3_traffic $RETRAIN_NE3 --batch_size 4 --data custom \
    --root_path ./data/long_term_forecast/traffic/ --data_path traffic.csv \
    --enc_in 862 --dec_in 862 --c_out 862
run ne1_traffic $RETRAIN_NE1 --pred_len 96 --batch_size 4 --data custom \
    --root_path ./data/long_term_forecast/traffic/ --data_path traffic.csv \
    --enc_in 862 --dec_in 862 --c_out 862

echo "[$(date +%T)] all remaining CQR-retrain runs done"
echo "Now regenerate the report:  $PY scripts/collect_calibration_results.py"
