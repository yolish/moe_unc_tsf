#!/bin/bash
# Two cells that no other script covers:
#
#   1. weather ACI, both gammas. Neither run_aci_gc_seed4021_ne3.sh (gamma=0.01) nor
#      run_aci_g001_seed4021_ne3.sh (gamma=0.001) includes weather in its 8-dataset grid,
#      so that cell has been blank in both ACI columns from the start. Cheap: 21 channels.
#   2. electricity MoECP. The job that would have produced it (electricity_mog in
#      run_calibration_gap_seed4021_ne3.sh) was SIGTERMed at 62 min while still in ACI,
#      after writing only its CP-MOG result, and nothing has picked it up since.
#
# Same config as everywhere else: iTransformer, ne=3, MOG (--prob_expert), ug=0,
# features=M, seed 4021, pred_len 96, calibration only (--is_training 0).
#
# Serial, batch_size 4 on electricity. Three wide-channel jobs have already been reaped by
# the OOM killer today with both A100s ~97% held by another tenant.
#
#   bash scripts/run_gap_weather_aci_and_elec_moecp.sh [gpu]
set -u
cd "$(dirname "$0")/.."
GPU=${1:-1}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/gap_weather_aci
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
    grep -E 'Coverage:|Avg Width' "$log" | tail -4
}

BASE="--task_name long_term_forecast --model_id test --model iTransformer \
      --features M --seq_len 96 --label_len 48 --seed 4021 --num_experts 3 \
      --prob_expert --is_training 0"

# weather, both ACI gammas in one pass (separate flags -> separate result files).
run weather_aci_both $BASE --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/weather/ --data_path weather.csv \
    --enc_in 21 --dec_in 21 --c_out 21 \
    --do_aci_aleatoric_scale_calibration --do_aci_aleatoric_scale_g001_calibration

run electricity_moecp $BASE --batch_size 4 --data custom \
    --root_path ./data/long_term_forecast/electricity/ --data_path electricity.csv \
    --enc_in 321 --dec_in 321 --c_out 321 \
    --do_moecp_calibration

echo "[$(date +%T)] done"
