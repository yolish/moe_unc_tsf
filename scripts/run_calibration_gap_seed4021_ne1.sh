#!/bin/bash
# Fill the missing cells of the seed-4021 / iTransformer / ne=1 / features=M headline
# table (docs/table_seed4021_ne1_iTransformer.csv), pred_len 96 except national-illness=24.
#
# The columns are the same eight methods as the ne=3 table: CP, CPVS, CPVS-aleatoric,
# CP-MOG, ACI g=0.01, ACI g=0.001, CQR quantile, CQR retrain. CQR retrain is NOT run here
# -- the ne=3 table has it for the four ETT datasets only, and ne=1 already matches that.
#
# Three kinds of job, cheapest first so the table fills in from the top:
#   (A) calibration-only on an existing MOG checkpoint (--prob_expert, ug=0, model_id=test).
#   (B) train the MOG checkpoint (weather/electricity/traffic have none at ne=1) and
#       calibrate in the same process.
#   (C) CQR quantile: a separate pinball-loss model (model_id=cqr, no --prob_expert),
#       trained then calibrated.
#
# Hyperparameters follow scripts/run_campaign_25h.py -- the current seed-4021 trainer --
# i.e. run.py defaults (lr 1e-4, 10 epochs, patience 3) with batch_size 4 on the wide
# datasets (electricity 321ch, traffic 862ch) and 8 elsewhere.
#
# Strictly serial: both A100s are ~90% occupied by the 25h campaign and another tenant,
# so extra lanes only produce OOMs.
#
#   bash scripts/run_calibration_gap_seed4021_ne1.sh [gpu]
#
# Afterwards: python scripts/collect_calibration_results.py
#             python scripts/build_headline_table.py --num-experts 1
set -u
cd "$(dirname "$0")/.."
GPU=${1:-1}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/gap_seed4021_ne1
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Take the matching claims out of the 25h campaign's queue so it does not repeat this work
# (the claim is a plain O_EXCL file keyed by "<kind>_<dataset>_ne<N>_<variant>").
CLAIMS=logs/campaign/claims
if [ -d "$CLAIMS" ]; then
    for tag in calib_ETTm2_ne1_MOG calib_exchange-rate_ne1_MOG calib_national-illness_ne1_MOG \
               train_weather_ne1_MOG train_electricity_ne1_MOG train_traffic_ne1_MOG \
               calib_weather_ne1_MOG calib_electricity_ne1_MOG calib_traffic_ne1_MOG \
               cqr_exchange-rate_ne1_MOE cqr_weather_ne1_MOE cqr_electricity_ne1_MOE \
               cqr_traffic_ne1_MOE; do
        [ -e "$CLAIMS/$tag" ] || echo "gap_ne1 $$" > "$CLAIMS/$tag"
    done
fi

run() {   # tag, then full arg list for run.py
    local tag=$1; shift
    local log="logs/gap_seed4021_ne1/${tag}_${STAMP}.log"
    echo "[$(date +%T)] START $tag -> $log"
    local t0=$SECONDS
    $PY -u run.py "$@" > "$log" 2>&1
    echo "[$(date +%T)] END   $tag rc=$? ($((SECONDS - t0))s)"
}

COMMON="--task_name long_term_forecast --model iTransformer --features M \
        --seq_len 96 --label_len 48 --seed 4021 --num_experts 1"
MOG_CAL="--model_id test --prob_expert $COMMON"
# The six MOG columns of the table, in column order.
ALL_MOG="--do_cp_calibration --do_cpvs_calibration --do_aleatoric_only_calibration \
         --do_aleatoric_scale_calibration --do_aci_aleatoric_scale_calibration \
         --do_aci_aleatoric_scale_g001_calibration"

# ------------------------------------------------- (A) calibration on existing checkpoints
run ettm2_mog $MOG_CAL --is_training 0 --data ETTm2 --batch_size 8 --pred_len 96 \
    --root_path ./data/long_term_forecast/ETT/ --data_path ETTm2.csv \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --do_aci_aleatoric_scale_calibration --do_aci_aleatoric_scale_g001_calibration

run illness_mog $MOG_CAL --is_training 0 --data custom --batch_size 8 --pred_len 24 \
    --root_path ./data/long_term_forecast/illness/ --data_path national_illness.csv \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --do_aci_aleatoric_scale_calibration --do_aci_aleatoric_scale_g001_calibration

# exchange-rate has CP + CPVS only.
run exchange_mog $MOG_CAL --is_training 0 --data custom --batch_size 8 --pred_len 96 \
    --root_path ./data/long_term_forecast/exchange_rate/ --data_path exchange_rate.csv \
    --enc_in 8 --dec_in 8 --c_out 8 \
    --do_aleatoric_only_calibration --do_aleatoric_scale_calibration \
    --do_aci_aleatoric_scale_calibration --do_aci_aleatoric_scale_g001_calibration

# ------------------------------------- (B) train the missing MOG checkpoints, then calibrate
run weather_mog $MOG_CAL --is_training 1 --data custom --batch_size 8 --pred_len 96 \
    --root_path ./data/long_term_forecast/weather/ --data_path weather.csv \
    --enc_in 21 --dec_in 21 --c_out 21 $ALL_MOG

run electricity_mog $MOG_CAL --is_training 1 --data custom --batch_size 4 --pred_len 96 \
    --root_path ./data/long_term_forecast/electricity/ --data_path electricity.csv \
    --enc_in 321 --dec_in 321 --c_out 321 $ALL_MOG

run traffic_mog $MOG_CAL --is_training 1 --data custom --batch_size 4 --pred_len 96 \
    --root_path ./data/long_term_forecast/traffic/ --data_path traffic.csv \
    --enc_in 862 --dec_in 862 --c_out 862 $ALL_MOG

# ----------------------------------------------------------------- (C) CQR quantile models
CQR="--model_id cqr --is_training 1 --use_quantile_loss --do_cqr_calibration $COMMON"

run exchange_cqr $CQR --data custom --batch_size 8 --pred_len 96 \
    --root_path ./data/long_term_forecast/exchange_rate/ --data_path exchange_rate.csv \
    --enc_in 8 --dec_in 8 --c_out 8

run weather_cqr $CQR --data custom --batch_size 8 --pred_len 96 \
    --root_path ./data/long_term_forecast/weather/ --data_path weather.csv \
    --enc_in 21 --dec_in 21 --c_out 21

run electricity_cqr $CQR --data custom --batch_size 4 --pred_len 96 \
    --root_path ./data/long_term_forecast/electricity/ --data_path electricity.csv \
    --enc_in 321 --dec_in 321 --c_out 321

run traffic_cqr $CQR --data custom --batch_size 4 --pred_len 96 \
    --root_path ./data/long_term_forecast/traffic/ --data_path traffic.csv \
    --enc_in 862 --dec_in 862 --c_out 862

echo "[$(date +%T)] all ne=1 gap cells attempted"
