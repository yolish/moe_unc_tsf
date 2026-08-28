#!/bin/bash
# Fill the missing cells of the seed-4021 / iTransformer / ne=1 / MOE headline table
# (docs/table_seed4021_ne1_moe_itransformer.csv), pred_len 96 except national-illness=24.
#
# The MOE variant has no predicted MoG variance, so the six variance-scaled columns are
# n/a by construction (build_headline_table.py:40). What is genuinely missing is:
#   (A) CQR retrain on weather / electricity / traffic / exchange-rate / national-illness.
#       The model_id=cqr ne=1 checkpoints already exist (trained by
#       run_calibration_gap_seed4021_ne1.sh), so these are calibration-only:
#       --is_training 0 --use_quantile_loss --do_cqr_retrain_calibration, defaults
#       finetune / window 1000 / interval=pred_len / 3 epochs, matching the ETT rows.
#   (B) CP on electricity and traffic. Unlike the other seven datasets these have no
#       model_id=test pe0 (MOE) checkpoint at ne=1, so the backbone is trained here and
#       calibrated in the same process. This also fills their mse/mae cells, which
#       build_headline_table.py takes from the standard_cp row of the requested variant.
#
# Hyperparameters follow scripts/run_calibration_gap_seed4021_ne1.sh: run.py defaults
# (lr 1e-4, 10 epochs, patience 3) with batch_size 4 on the wide datasets (electricity
# 321ch, traffic 862ch) and 8 elsewhere.
#
# Cheapest first, strictly serial: both A100s are already busy with other tenants, so
# extra lanes only produce OOMs. Resumable in the sense that finished results are appended
# to the result_calibration_*.txt files -- re-running repeats work but does not corrupt it.
#
#   bash scripts/run_calibration_gap_seed4021_ne1_moe.sh [gpu]
#
# Afterwards: ./unc_moe/bin/python scripts/collect_calibration_results.py
#             ./unc_moe/bin/python scripts/build_headline_table.py --num-experts 1 --variant MOE
set -u
cd "$(dirname "$0")/.."
GPU=${1:-0}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/gap_seed4021_ne1_moe
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {   # tag, then full arg list for run.py
    local tag=$1; shift
    local log="logs/gap_seed4021_ne1_moe/${tag}_${STAMP}.log"
    echo "[$(date +%T)] START $tag -> $log"
    local t0=$SECONDS
    $PY -u run.py "$@" > "$log" 2>&1
    echo "[$(date +%T)] END   $tag rc=$? ($((SECONDS - t0))s)"
}

COMMON="--task_name long_term_forecast --model iTransformer --features M \
        --seq_len 96 --label_len 48 --seed 4021 --num_experts 1"

# --------------------------------------------------- (A) CQR retrain on existing cqr models
CQR_RE="--model_id cqr --is_training 0 --use_quantile_loss \
        --do_cqr_retrain_calibration $COMMON"

run exchange_cqr_retrain $CQR_RE --data custom --batch_size 8 --pred_len 96 \
    --root_path ./data/long_term_forecast/exchange_rate/ --data_path exchange_rate.csv \
    --enc_in 8 --dec_in 8 --c_out 8

run illness_cqr_retrain $CQR_RE --data custom --batch_size 8 --pred_len 24 \
    --root_path ./data/long_term_forecast/illness/ --data_path national_illness.csv \
    --enc_in 7 --dec_in 7 --c_out 7

run weather_cqr_retrain $CQR_RE --data custom --batch_size 8 --pred_len 96 \
    --root_path ./data/long_term_forecast/weather/ --data_path weather.csv \
    --enc_in 21 --dec_in 21 --c_out 21

run electricity_cqr_retrain $CQR_RE --data custom --batch_size 4 --pred_len 96 \
    --root_path ./data/long_term_forecast/electricity/ --data_path electricity.csv \
    --enc_in 321 --dec_in 321 --c_out 321

run traffic_cqr_retrain $CQR_RE --data custom --batch_size 4 --pred_len 96 \
    --root_path ./data/long_term_forecast/traffic/ --data_path traffic.csv \
    --enc_in 862 --dec_in 862 --c_out 862

# ------------------------------------- (B) train the missing MOE backbones, then calibrate CP
MOE_CAL="--model_id test --is_training 1 --do_cp_calibration $COMMON"

run electricity_moe $MOE_CAL --data custom --batch_size 4 --pred_len 96 \
    --root_path ./data/long_term_forecast/electricity/ --data_path electricity.csv \
    --enc_in 321 --dec_in 321 --c_out 321

run traffic_moe $MOE_CAL --data custom --batch_size 4 --pred_len 96 \
    --root_path ./data/long_term_forecast/traffic/ --data_path traffic.csv \
    --enc_in 862 --dec_in 862 --c_out 862

echo "[$(date +%T)] all ne=1 MOE gap cells attempted"
