#!/bin/bash
# Second ACI Aleatoric Scale variant: same clipped-alpha_t calibrator, gamma fixed at 0.001
# via --do_aci_aleatoric_scale_g001_calibration, run for direct comparison against the
# gamma=0.01 grid in run_aci_gc_seed4021_ne3.sh (same 8-dataset config: iTransformer, ne=3,
# MOG, ug=0, seed 4021, pred_len default 96 except national-illness=24).
#
# Results append to result_calibration_aci_aleatoric_scale_g001_tsf.txt -- a separate file
# from the gamma=0.01 variant, so both can be collected and shown side by side.
#
#   bash scripts/run_aci_g001_seed4021_ne3.sh [gpu]
set -u
cd "$(dirname "$0")/.."
GPU=${1:-1}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/aci_g001_seed4021
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
    grep -E 'Coverage:|Final alpha_t' "$log" | tail -3
}

BASE="--task_name long_term_forecast --model_id test --model iTransformer \
      --features M --seq_len 96 --label_len 48 --seed 4021 --num_experts 3 \
      --prob_expert --is_training 0 --do_aci_aleatoric_scale_g001_calibration"

ETT="./data/long_term_forecast/ETT/"

for ds in ETTh1 ETTh2 ETTm1 ETTm2; do
    run "${ds,,}" $BASE --batch_size 8 \
        --root_path $ETT --data_path ${ds}.csv --data $ds \
        --enc_in 7 --dec_in 7 --c_out 7
done

run exchange $BASE --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/exchange_rate/ --data_path exchange_rate.csv \
    --enc_in 8 --dec_in 8 --c_out 8

run illness $BASE --batch_size 8 --data custom --pred_len 24 \
    --root_path ./data/long_term_forecast/illness/ --data_path national_illness.csv \
    --enc_in 7 --dec_in 7 --c_out 7

run electricity $BASE --batch_size 4 --data custom \
    --root_path ./data/long_term_forecast/electricity/ --data_path electricity.csv \
    --enc_in 321 --dec_in 321 --c_out 321

run traffic $BASE --batch_size 4 --data custom \
    --root_path ./data/long_term_forecast/traffic/ --data_path traffic.csv \
    --enc_in 862 --dec_in 862 --c_out 862

echo "[$(date +%T)] all 8 ACI (g=0.001) cells attempted"
