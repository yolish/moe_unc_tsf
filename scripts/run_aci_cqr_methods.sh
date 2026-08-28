#!/bin/bash
# ACI variants of the two CQR calibrators.
#
#   bash scripts/run_aci_cqr_methods.sh [gpu]
#
# Separate from run_aci_base_methods.sh because these need --use_quantile_loss, which is
# incompatible with --prob_expert: the pinball-loss head replaces the MoG variance output,
# and run.py prints an explicit skip if both are set. So these run against the --model_id cqr
# checkpoints (pe0), matching where the base cqr_quantile / cqr_retrain numbers come from.
# Same 3 experts, just not the MoG configuration -- do not table these next to the
# --prob_expert results as if they were one grid.
#
# gamma = 0.001, per the 0.1/pred_len rule of thumb at pred_len=96. See run_aci_base_methods.sh.
#
# Retrained CQR refits the model ~29 times per dataset (~6 min each on ETTh1); expect this
# script to take substantially longer than the base-methods one.
set -u
cd "$(dirname "$0")/.."
GPU=${1:-0}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/aci_cqr_ne3
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

BASE="--task_name long_term_forecast --model_id cqr --model iTransformer \
      --features M --seq_len 96 --label_len 48 --seed 4021 --num_experts 3 \
      --is_training 0 --use_quantile_loss --aci_gamma 0.001 --aci_alpha 0.1"

FLAGS="--do_aci_cqr_calibration --do_aci_cqr_retrain_calibration"

ETT="./data/long_term_forecast/ETT/"

for ds in ETTh1 ETTh2 ETTm1 ETTm2; do
    run "${ds,,}" $BASE $FLAGS --batch_size 8 \
        --root_path $ETT --data_path ${ds}.csv --data $ds \
        --enc_in 7 --dec_in 7 --c_out 7
done

run exchange $BASE $FLAGS --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/exchange_rate/ --data_path exchange_rate.csv \
    --enc_in 8 --dec_in 8 --c_out 8

run illness $BASE $FLAGS --batch_size 8 --data custom --pred_len 24 \
    --root_path ./data/long_term_forecast/illness/ --data_path national_illness.csv \
    --enc_in 7 --dec_in 7 --c_out 7

run weather $BASE $FLAGS --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/weather/ --data_path weather.csv \
    --enc_in 21 --dec_in 21 --c_out 21

echo "[$(date +%T)] ACI CQR grid done"
echo "Now regenerate the report:  $PY scripts/collect_calibration_results.py"
