#!/bin/bash
# electricity + traffic half of the weather/traffic/electricity calibration gap.
#
# Same cells as scripts/run_calibration_gap_large_mog.sh (iTransformer, ne=3, MOG
# i.e. --prob_expert with unc_gating=0, pl=96, seed 4021), same methods (Aleatoric
# Scale CP = "CP-MOG", and MoECP at the default tau=100 / alpha=0.1), but MoECP runs
# channel-parallel via --moecp_workers.
#
# MoECP is O(W log W * H * C) per forecast origin, so it scales with channel count:
# measured 0.23 s/origin on weather (21 ch), 4.72 on electricity (321), 13.13 on
# traffic (862) -- ~6.6 h and ~12.1 h serially. The channel axis is separable, and the
# parent retains the RNG and scatters pi~, so the parallel result is bit-identical to
# --moecp_workers 1 (verified against the serial path over both recompute_every=1 and
# 3, including the unbounded-interval cells).
#
# weather is deliberately NOT here: it is already running on the serial path from the
# first script, which gives an independent end-to-end check on real data.
#
#   bash scripts/run_calibration_gap_large_mog_parallel.sh
set -u
cd "$(dirname "$0")/.."
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

run_cell() {   # gpu, dataset_dir, data_file, workers
    local gpu=$1 dir=$2 file=$3 nw=$4 rc=0
    local name=${file%.csv}
    local log="logs/calib_gap_${name}_mog_seed4021_par_${STAMP}.log"
    echo "[$(date +%T)] $name -> $log (${nw} workers)"
    CUDA_VISIBLE_DEVICES=$gpu $PY -u run.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --root_path "./data/long_term_forecast/${dir}/" \
        --data_path "$file" \
        --model_id test \
        --model iTransformer \
        --data custom \
        --features M \
        --seq_len 96 --label_len 48 --pred_len 96 \
        --batch_size 8 \
        --seed 4021 \
        --num_experts 3 \
        --prob_expert \
        --moecp_workers "$nw" \
        --do_aleatoric_scale_calibration \
        --do_moecp_calibration > "$log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[$(date +%T)] $name FAILED (rc=$rc) -- last lines:"
        tail -3 "$log"
        return "$rc"
    fi
    echo "[$(date +%T)] $name done"
}

# 128 cores, load ~10, 459G free. Workers are numpy-only (forked after GPU inference),
# so the two datasets run concurrently.
#
# GPU choice is contention, not capacity: only the one-off inference pass touches the
# GPU (~1 GB), everything after is numpy. Both cards on this box are routinely occupied
# by other jobs -- a first attempt on GPU 0 died in exp.test() with CUDA OOM after an
# unrelated process took 76 GB of it -- so pass a card that currently has headroom.
GPU=${1:-1}
run_cell "$GPU" electricity electricity.csv 16 &
E=$!
run_cell "$GPU" traffic     traffic.csv     24 &
T=$!
wait $E; rc_e=$?
wait $T; rc_t=$?
echo "[$(date +%T)] electricity rc=$rc_e, traffic rc=$rc_t"
exit $(( rc_e | rc_t ))
