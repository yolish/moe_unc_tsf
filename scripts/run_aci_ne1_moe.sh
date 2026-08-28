#!/bin/bash
# ACI CP at ne1 (MOE variant: 1 expert, no --prob_expert). The only ACI method that
# applies here -- CPVS/Aleatoric-only/MoECP all need --prob_expert for a MoG variance
# decomposition or gating distribution, neither of which exists on this variant, so
# build_headline_table.py correctly reports them as n/a rather than missing. The CQR
# columns on this table are shared with run_aci_ne1_mog.sh's CQR runs (CQR never depends
# on --prob_expert), so they are not repeated here.
#
#   bash scripts/run_aci_ne1_moe.sh [gpu]
#
# electricity/traffic have no ne1/pe0 checkpoint at all (base CP itself was never trained
# there for this variant -- confirmed before writing this script), so they are skipped
# here, not just their ACI variant.
set -u
cd "$(dirname "$0")/.."
GPU=${1:-1}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/aci_ne1_moe
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
    grep -E 'Coverage:|Avg Width|Final alpha_t' "$log" | tail -3
}

MOE="--task_name long_term_forecast --model_id test --model iTransformer \
     --features M --seq_len 96 --label_len 48 --seed 4021 --num_experts 1 \
     --is_training 0 --aci_gamma 0.001 --aci_alpha 0.1 --do_aci_cp_calibration"

ETT="./data/long_term_forecast/ETT/"

for ds in ETTh1 ETTh2 ETTm1 ETTm2; do
    run "${ds,,}" $MOE --pred_len 96 --batch_size 8 \
        --root_path $ETT --data_path ${ds}.csv --data $ds \
        --enc_in 7 --dec_in 7 --c_out 7
done

run weather $MOE --pred_len 96 --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/weather/ --data_path weather.csv \
    --enc_in 21 --dec_in 21 --c_out 21

run exchange $MOE --pred_len 96 --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/exchange_rate/ --data_path exchange_rate.csv \
    --enc_in 8 --dec_in 8 --c_out 8

run illness $MOE --pred_len 24 --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/illness/ --data_path national_illness.csv \
    --enc_in 7 --dec_in 7 --c_out 7

echo "[$(date +%T)] ne1 MOE ACI-CP done for the 7 datasets with a pe0/ne1 checkpoint"
echo "Now regenerate the report:  $PY scripts/collect_calibration_results.py"
