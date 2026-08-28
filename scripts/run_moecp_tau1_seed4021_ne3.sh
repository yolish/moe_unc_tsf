#!/bin/bash
# Re-run MoECP at tau=1 (new default, was 100) across the 9-dataset grid this session has
# been using: iTransformer, ne=3, MOG, ug=0, seed 4021, pred_len 96 except illness=24.
#
# Why: at tau=100 (the paper's tabular default) the test point's own localization weight
# exceeds alpha often enough to make a meaningful fraction of intervals unbounded --
# up to 25% on national-illness -- which makes MoECP's width column not directly
# comparable to the other calibrators here (none of which ever go unbounded). Theorem 1's
# coverage guarantee holds at any tau in N+, so lowering it is a comparability choice, not
# a correctness fix. Measured on illness/ETTh2/exchange-rate: tau=1 brings unbounded rates
# to ~0 while width stays well under standard_cp's on the same data, i.e. MoECP still
# localizes -- it has not collapsed to uniform weights (that only happens at tau=0).
#
# Results append to result_calibration_moecp_tsf.txt; the collector picks the last
# observation per (setting, method) as canonical, so the old tau=100 rows are naturally
# superseded once these land -- no manual cleanup needed.
#
#   bash scripts/run_moecp_tau1_seed4021_ne3.sh [gpu]
set -u
cd "$(dirname "$0")/.."
GPU=${1:-1}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/moecp_tau1_seed4021
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
      --prob_expert --is_training 0 --do_moecp_calibration --moecp_temperature 1"

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

run weather $BASE --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/weather/ --data_path weather.csv \
    --enc_in 21 --dec_in 21 --c_out 21

run electricity $BASE --batch_size 4 --data custom \
    --root_path ./data/long_term_forecast/electricity/ --data_path electricity.csv \
    --enc_in 321 --dec_in 321 --c_out 321

run traffic $BASE --batch_size 4 --data custom \
    --root_path ./data/long_term_forecast/traffic/ --data_path traffic.csv \
    --enc_in 862 --dec_in 862 --c_out 862

echo "[$(date +%T)] all 9 MoECP (tau=1) cells attempted"
