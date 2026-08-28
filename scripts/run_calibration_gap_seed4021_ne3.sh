#!/bin/bash
# Fill the missing cells of the seed-4021 / iTransformer / ne=3 / features=M calibration
# table, one pred_len per dataset (96 everywhere except national-illness = 24).
#
# What is already covered elsewhere and therefore NOT here:
#   * ETTh1/ETTh2/ETTm1/ETTm2 @96 -- complete for every method in the table.
#   * weather @96 -- CP / CPVS / aleatoric-only live in logs/run_mog_cp_all_datasets2.log,
#     CP-MOG / ACI / MoECP in the result_calibration_*.txt files. Only CQR is missing.
#   * traffic @96 CP-MOG + MoECP -- already running (calib_gap_traffic_mog_seed4021_*).
#   * CQR retrain -- deliberately excluded.
#
# Two kinds of job:
#   (A) calibration-only on existing MOG checkpoints (--prob_expert, ug=0, model_id=test).
#   (B) CQR quantile: a separate pinball-loss model (model_id=cqr, no --prob_expert),
#       so it has to be trained before it can be calibrated.
#
# Strictly serial. Both A100s are ~97% occupied by another tenant and the earlier parallel
# attempt at electricity died with "CUDA out of memory ... 2.12 MiB is free", so extra lanes
# only produce OOMs. Big-channel datasets get batch_size 4 for the same reason.
#
#   bash scripts/run_calibration_gap_seed4021_ne3.sh [gpu]
#
# Afterwards run scripts/collect_calibration_results.py to refresh
# docs/calibration_results_tsf.{md,csv}.
set -u
cd "$(dirname "$0")/.."
GPU=${1:-1}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/gap_seed4021
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {   # tag, then full arg list for run.py
    local tag=$1; shift
    local log="logs/gap_seed4021/${tag}_${STAMP}.log"
    echo "[$(date +%T)] START $tag -> $log"
    local t0=$SECONDS
    $PY -u run.py "$@" > "$log" 2>&1
    local rc=$?
    echo "[$(date +%T)] END   $tag rc=$rc ($((SECONDS - t0))s)"
}

MOG="--task_name long_term_forecast --model_id test --model iTransformer --data custom \
     --features M --seq_len 96 --label_len 48 --seed 4021 --num_experts 3 \
     --prob_expert --is_training 0"

# ---------------------------------------------------------------- (A) MOG calibration
# exchange-rate: has CP + CPVS only.
run exchange_mog $MOG --batch_size 8 \
    --root_path ./data/long_term_forecast/exchange_rate/ --data_path exchange_rate.csv \
    --enc_in 8 --dec_in 8 --c_out 8 \
    --do_aleatoric_only_calibration --do_aleatoric_scale_calibration \
    --do_aci_aleatoric_scale_calibration --do_moecp_calibration

# national-illness: has CP + CPVS + aleatoric-only + CP-MOG.
run illness_mog $MOG --batch_size 8 --pred_len 24 \
    --root_path ./data/long_term_forecast/illness/ --data_path national_illness.csv \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --do_aci_aleatoric_scale_calibration --do_moecp_calibration

# electricity: has CP + CPVS + aleatoric-only.
run electricity_mog $MOG --batch_size 4 \
    --root_path ./data/long_term_forecast/electricity/ --data_path electricity.csv \
    --enc_in 321 --dec_in 321 --c_out 321 \
    --do_aleatoric_scale_calibration --do_aci_aleatoric_scale_calibration \
    --do_moecp_calibration

# traffic: CP-MOG + MoECP already in flight, so only ACI here.
run traffic_mog $MOG --batch_size 4 \
    --root_path ./data/long_term_forecast/traffic/ --data_path traffic.csv \
    --enc_in 862 --dec_in 862 --c_out 862 \
    --do_aci_aleatoric_scale_calibration

# ----------------------------------------------------------------- (B) CQR quantile
CQR="--task_name long_term_forecast --model_id cqr --model iTransformer --data custom \
     --features M --seq_len 96 --label_len 48 --is_training 1 \
     --seed 4021 --num_experts 3 --use_quantile_loss --do_cqr_calibration"

run weather_cqr $CQR --batch_size 8 \
    --root_path ./data/long_term_forecast/weather/ --data_path weather.csv \
    --enc_in 21 --dec_in 21 --c_out 21

run exchange_cqr $CQR --batch_size 8 \
    --root_path ./data/long_term_forecast/exchange_rate/ --data_path exchange_rate.csv \
    --enc_in 8 --dec_in 8 --c_out 8

run illness_cqr $CQR --batch_size 8 --pred_len 24 \
    --root_path ./data/long_term_forecast/illness/ --data_path national_illness.csv \
    --enc_in 7 --dec_in 7 --c_out 7

run electricity_cqr $CQR --batch_size 4 \
    --root_path ./data/long_term_forecast/electricity/ --data_path electricity.csv \
    --enc_in 321 --dec_in 321 --c_out 321

run traffic_cqr $CQR --batch_size 4 \
    --root_path ./data/long_term_forecast/traffic/ --data_path traffic.csv \
    --enc_in 862 --dec_in 862 --c_out 862

echo "[$(date +%T)] all gap cells attempted"
