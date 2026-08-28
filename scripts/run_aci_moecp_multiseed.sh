#!/bin/bash
# ACI MoECP under the clip-to-window-max rule (2026-08-14 patch to
# ACIMoECPCalibrator._read_off), gamma = 0.001, MoGE / 3 experts.
#
#   bash scripts/run_aci_moecp_multiseed.sh [gpu]
#
# Two jobs in one:
#   1. ETT x seeds 4021-4025 -- the new multi-seed cells for the ACI headline table.
#   2. weather + exchange-rate at seed 4021 -- NOT new cells. Those rows already exist
#      but were produced under the old +inf rule, so leaving them would mix two clipping
#      semantics inside tab:other_aci. Re-run so every MoECP + ACI number in the report
#      comes from the same rule.
# Seed 4021 is re-run for the same reason: its stored value predates the patch. The
# collector keeps the last line in the result file, so these supersede the old ones (and
# will set rerun_disagrees=1, which is correct -- the rule changed, not the run).
#
# ne1 (Single Gaussian) is deliberately absent: MoECP is no longer reported there (with one
# expert the gate distribution is constant, so localisation degenerates to CP-fixed), see
# SG_COLS_ACI in scripts/build_results_tex.py.
set -u
cd "$(dirname "$0")/.."
GPU=${1:-0}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/aci_moecp_multiseed
mkdir -p "$LOGDIR"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
    local tag=$1
    local log=$2
    shift 2
    echo "[$(date +%T)] START $tag -> $log"
    local t0=$SECONDS
    $PY -u run.py "$@" > "$log" 2>&1
    local rc=$?
    echo "[$(date +%T)] END   $tag rc=$rc ($((SECONDS - t0))s)"
    grep -E 'Coverage:|Avg Width|Unbounded|Final alpha_t' "$log" | tail -4
}

ETT="./data/long_term_forecast/ETT/"

# tau=1.0: the collector drops any other tau, since coverage counts a clipped interval as a
# hit while width is a finite mean, so taus are not comparable across runs.
BASE="--task_name long_term_forecast --model_id test --model iTransformer \
      --features M --seq_len 96 --label_len 48 --pred_len 96 \
      --is_training 0 --num_experts 3 --prob_expert \
      --aci_gamma 0.001 --aci_alpha 0.1 --moecp_temperature 1.0 \
      --do_aci_moecp_calibration --batch_size 8"

for seed in 4021 4022 4023 4024 4025; do
  for ds in ETTh1 ETTh2 ETTm1 ETTm2; do
    run "${ds,,}_s${seed}_moecp" "${LOGDIR}/${ds,,}_s${seed}_${STAMP}.log" \
        $BASE --seed $seed \
        --root_path $ETT --data_path ${ds}.csv --data $ds \
        --enc_in 7 --dec_in 7 --c_out 7
  done
done

# --- consistency re-runs at seed 4021 only (old +inf rule -> clipped rule) ---
run "weather_s4021_moecp" "${LOGDIR}/weather_s4021_${STAMP}.log" \
    $BASE --seed 4021 --data custom \
    --root_path ./data/long_term_forecast/weather/ --data_path weather.csv \
    --enc_in 21 --dec_in 21 --c_out 21

run "exchange_s4021_moecp" "${LOGDIR}/exchange_s4021_${STAMP}.log" \
    $BASE --seed 4021 --data custom \
    --root_path ./data/long_term_forecast/exchange_rate/ --data_path exchange_rate.csv \
    --enc_in 8 --dec_in 8 --c_out 8

echo "[$(date +%T)] ACI MoECP multi-seed done"
echo "Now regenerate:  $PY scripts/collect_calibration_results.py"
