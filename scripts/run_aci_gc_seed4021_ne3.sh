#!/bin/bash
# Re-run ACI aleatoric scale on the 8 headline datasets after the Gibbs & Candes fix
# (unbounded alpha_t, C_t = R / {} saturation instead of clipping).
#
# Grid: iTransformer, num_experts=3, MOG (--prob_expert), ug=0, features=M, seed 4021,
# default pred_len = 96 -- except national-illness, whose project default is 24 (the only
# ne=3/seed-4021 ILI checkpoints are pl 24/36/48/60; ILI at 96 has never been trained here).
#
# Calibration only (--is_training 0): every checkpoint below already exists.
#
# Strictly serial on one card. Both A100s are ~90% occupied by another tenant, so extra
# lanes only produce OOMs; the wide-channel datasets get batch_size 4 for the same reason.
#
#   bash scripts/run_aci_gc_seed4021_ne3.sh [gpu]
#
# Results append to result_calibration_aci_aleatoric_scale_tsf.txt. NOTE that file also
# holds rows from the pre-fix (clipped) implementation -- they are not comparable.
set -u
cd "$(dirname "$0")/.."
GPU=${1:-1}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/aci_gc_seed4021
mkdir -p "$LOGDIR"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {   # tag, then full arg list for run.py
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
      --prob_expert --is_training 0 --do_aci_aleatoric_scale_calibration"

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

echo "[$(date +%T)] all 8 ACI (G&C) cells attempted"
