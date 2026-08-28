#!/bin/bash
# Rerun MoECP (base method only, not ACI-MoECP) across the full grid needed for
# docs/results_tables.tex, after the clip-to-window-max fix in
# calibration/moecp_calibration.py (2026-08-13): unreached target levels now cap at the
# window's own max residual per cell instead of returning +inf, so width and coverage are
# computed over the same population as every other calibrator here.
#
# Grid: iTransformer, MOG variant (--prob_expert, ug0), seed 4021, pl96, is_training 0
# (checkpoints already exist for every cell below -- calibration only, no training).
#   - ETTh1/ETTh2/ETTm1/ETTm2: num_experts 1,2,3,4,5   (headline Single Gaussian/MoGE rows
#     + the full expert sweep)
#   - weather/electricity/traffic/exchange-rate: num_experts 1,3 only (Single Gaussian +
#     MoGE headline rows; these datasets are never in the expert sweep)
#
# Old (pre-fix) MoECP results were archived to
# archive/result_calibration_moecp_preclip.removed_20260813.txt and
# result_calibration_moecp_tsf.txt was truncated, so every line this run appends is fresh.
#
# GPU0 only (GPU1 is shared with other users' jobs and has ~10GB free). Electricity and
# traffic get --moecp_workers 16 each (O(W log W * H * C) per origin, unusable serial on
# 321/862 channels -- see scripts/run_elec_moecp_workers.sh). 128 cores on this box, so
# running both large-dataset workers alongside the ETT/weather/exchange streams is fine.
#
#   bash scripts/run_moecp_clipfix_full_grid.sh
set -u
cd "$(dirname "$0")/.."
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
GPU=0
mkdir -p logs/moecp_clipfix

run_cell() {   # root_path, data_path, data, extra_args...
    local root=$1 file=$2 data=$3; shift 3
    local name="${data}_$(echo "$*" | tr -d ' -')"
    local log="logs/moecp_clipfix/${data}_$(date +%s%N)_${STAMP}.log"
    echo "[$(date +%T)] START $data $* -> $log"
    CUDA_VISIBLE_DEVICES=$GPU $PY -u run.py \
        --task_name long_term_forecast --is_training 0 \
        --root_path "$root" --data_path "$file" --data "$data" \
        --model_id test --model iTransformer --features M \
        --seq_len 96 --label_len 48 --pred_len 96 \
        --seed 4021 --prob_expert --do_moecp_calibration \
        "$@" > "$log" 2>&1
    local rc=$?
    echo "[$(date +%T)] END   $data $* rc=$rc"
}

ETT_ROOT=./data/long_term_forecast/ETT/

# Stream A: ETTh1, then ETTh2 -- ne 1..5 each, batch_size 8.
{ for ne in 1 2 3 4 5; do run_cell "$ETT_ROOT" ETTh1.csv ETTh1 --batch_size 8 --num_experts "$ne"; done
  for ne in 1 2 3 4 5; do run_cell "$ETT_ROOT" ETTh2.csv ETTh2 --batch_size 8 --num_experts "$ne"; done; } &

# Stream B: ETTm1, then ETTm2 -- ne 1..5 each, batch_size 8.
{ for ne in 1 2 3 4 5; do run_cell "$ETT_ROOT" ETTm1.csv ETTm1 --batch_size 8 --num_experts "$ne"; done
  for ne in 1 2 3 4 5; do run_cell "$ETT_ROOT" ETTm2.csv ETTm2 --batch_size 8 --num_experts "$ne"; done; } &

# Stream C: weather, then exchange-rate -- ne 1 and 3 only, batch_size 8.
{ for ne in 1 3; do run_cell ./data/long_term_forecast/weather/ weather.csv custom \
      --batch_size 8 --num_experts "$ne"; done
  for ne in 1 3; do run_cell ./data/long_term_forecast/exchange_rate/ exchange_rate.csv custom \
      --batch_size 8 --num_experts "$ne"; done; } &

# Stream D: electricity -- ne 1 and 3, batch_size 4, 16 workers (321 channels).
{ for ne in 1 3; do run_cell ./data/long_term_forecast/electricity/ electricity.csv custom \
      --batch_size 4 --num_experts "$ne" --moecp_workers 16; done; } &

# Stream E: traffic -- ne 1 and 3, batch_size 4, 16 workers (862 channels, the heaviest cell).
{ for ne in 1 3; do run_cell ./data/long_term_forecast/traffic/ traffic.csv custom \
      --batch_size 4 --num_experts "$ne" --moecp_workers 16; done; } &

wait
echo "[$(date +%T)] all 28 MoECP clip-fix cells attempted"
