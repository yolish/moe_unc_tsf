#!/bin/bash
# ACI at ne1 (MOG, 1-expert, --prob_expert), CQR-retrain split out (run_aci_cqr_retrain_all.sh
# runs it last, across every pending config, so it never blocks a faster method behind it).
#
#   bash scripts/run_aci_ne1_mog_others.sh [gpu]
#
# CP/CPVS/Aleatoric-only + CQR-quantile run on all 9 datasets. MoECP runs on the 7 that are
# channel-count feasible (skips electricity/traffic, same as ne3), on every one of those 7
# regardless of whether a base MoECP row exists there (confirmed with the user earlier).
set -u
cd "$(dirname "$0")/.."
GPU=${1:-0}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/aci_ne1_mog
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
    grep -E 'Coverage:|Avg Width|Unbounded|Final alpha_t' "$log" | tail -5
}

MOG="--task_name long_term_forecast --model_id test --model iTransformer \
     --features M --seq_len 96 --label_len 48 --seed 4021 --num_experts 1 \
     --prob_expert --is_training 0 --aci_gamma 0.001 --aci_alpha 0.1"
MOG_SMALL="$MOG --do_aci_cp_calibration --do_aci_cpvs_calibration --do_aci_aleatoric_only_calibration --do_aci_moecp_calibration --moecp_temperature 1.0"
MOG_LARGE="$MOG --do_aci_cp_calibration --do_aci_cpvs_calibration --do_aci_aleatoric_only_calibration"

CQR_QUANTILE="--task_name long_term_forecast --model_id cqr --model iTransformer \
     --features M --seq_len 96 --label_len 48 --seed 4021 --num_experts 1 \
     --is_training 0 --use_quantile_loss --aci_gamma 0.001 --aci_alpha 0.1 \
     --do_aci_cqr_calibration"

ETT="./data/long_term_forecast/ETT/"

for ds in ETTh1 ETTh2 ETTm1 ETTm2; do
    run "${ds,,}_mog" $MOG_SMALL --pred_len 96 --batch_size 8 \
        --root_path $ETT --data_path ${ds}.csv --data $ds \
        --enc_in 7 --dec_in 7 --c_out 7
    run "${ds,,}_cqr" $CQR_QUANTILE --pred_len 96 --batch_size 8 \
        --root_path $ETT --data_path ${ds}.csv --data $ds \
        --enc_in 7 --dec_in 7 --c_out 7
done

run weather_mog $MOG_SMALL --pred_len 96 --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/weather/ --data_path weather.csv \
    --enc_in 21 --dec_in 21 --c_out 21
run weather_cqr $CQR_QUANTILE --pred_len 96 --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/weather/ --data_path weather.csv \
    --enc_in 21 --dec_in 21 --c_out 21

run exchange_mog $MOG_SMALL --pred_len 96 --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/exchange_rate/ --data_path exchange_rate.csv \
    --enc_in 8 --dec_in 8 --c_out 8
run exchange_cqr $CQR_QUANTILE --pred_len 96 --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/exchange_rate/ --data_path exchange_rate.csv \
    --enc_in 8 --dec_in 8 --c_out 8

run illness_mog $MOG_SMALL --pred_len 24 --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/illness/ --data_path national_illness.csv \
    --enc_in 7 --dec_in 7 --c_out 7
run illness_cqr $CQR_QUANTILE --pred_len 24 --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/illness/ --data_path national_illness.csv \
    --enc_in 7 --dec_in 7 --c_out 7

run electricity_mog $MOG_LARGE --pred_len 96 --batch_size 4 --data custom \
    --root_path ./data/long_term_forecast/electricity/ --data_path electricity.csv \
    --enc_in 321 --dec_in 321 --c_out 321
run electricity_cqr $CQR_QUANTILE --pred_len 96 --batch_size 4 --data custom \
    --root_path ./data/long_term_forecast/electricity/ --data_path electricity.csv \
    --enc_in 321 --dec_in 321 --c_out 321

run traffic_mog $MOG_LARGE --pred_len 96 --batch_size 4 --data custom \
    --root_path ./data/long_term_forecast/traffic/ --data_path traffic.csv \
    --enc_in 862 --dec_in 862 --c_out 862
run traffic_cqr $CQR_QUANTILE --pred_len 96 --batch_size 4 --data custom \
    --root_path ./data/long_term_forecast/traffic/ --data_path traffic.csv \
    --enc_in 862 --dec_in 862 --c_out 862

echo "[$(date +%T)] ne1 MOG (non-retrain) done for all 9 datasets"
