#!/bin/bash
# MoECP on ClassicMoE (num_experts=3, deterministic softmax-gated experts, MSE loss,
# NO --prob_expert) across the TSF grid, seed 4021, tau=1.
#
# Why: every prior MoECP run in this repo used the MOG variant (--prob_expert). MoECP
# only needs the gate distribution pi(x) and the raw residual |y - yhat| -- it never
# touches per-expert variance -- so it is equally well-defined for ClassicMoE. Until this
# session run.py additionally required --prob_expert to reach calibrate_moecp(); that
# guard has been loosened (run.py, do_moecp_calibration branches) to only require a
# softmax gate (num_experts>1, and prob_expert only if --unc_gating is set, since
# unc_gating's inverse-variance weights need per-expert variance).
#
# Calibration only (--is_training 0): checkpoints already exist for these 6 cells
# (long_term_forecast_test_iTransformer_<ds>_ne3_pe0_ug0_..._seed4021). weather,
# electricity and traffic have no ne3/pe0/ug0 checkpoint yet and are NOT in this script.
#
#   bash scripts/run_moecp_classicmoe_ne3_seed4021.sh [gpu]
set -u
cd "$(dirname "$0")/.."
GPU=${1:-0}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/moecp_classicmoe_ne3_seed4021
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
    grep -E 'Coverage:|Avg Width|Unbounded' "$log" | tail -3
}

BASE="--task_name long_term_forecast --model_id test --model iTransformer \
      --features M --seq_len 96 --label_len 48 --seed 4021 --num_experts 3 \
      --is_training 0 --do_moecp_calibration --moecp_temperature 1"

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

echo "[$(date +%T)] all 6 ClassicMoE MoECP cells attempted"
