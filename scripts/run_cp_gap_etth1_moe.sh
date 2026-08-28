#!/bin/bash
# Fill the missing Standard-CP cells for ETTh1 / iTransformer / MoE / pl96.
#
# The width-vs-experts figure had ne=1 and ne=3 on seed 4021 only, while ne=2/4/5 had
# all five seeds. Every checkpoint below already exists, so this is calibration only
# (--is_training 0) -- no retraining, ~2-4 min per cell.
#
#   bash scripts/run_cp_gap_etth1_moe.sh
#
# Splits the 8 cells over the two GPUs and waits. Results land in
# result_calibration_mse_cp.txt; re-run scripts/plot_width_vs_experts.py afterwards.
set -u
cd "$(dirname "$0")/.."
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)

run_cell() {   # gpu, num_experts, seed
    local gpu=$1 ne=$2 seed=$3
    local log="logs/cp_gap_etth1_moe_ne${ne}_seed${seed}_${STAMP}.log"
    echo "[$(date +%T)] gpu$gpu  ne=$ne seed=$seed -> $log"
    CUDA_VISIBLE_DEVICES=$gpu $PY -u run.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --root_path ./data/long_term_forecast/ETT/ \
        --data_path ETTh1.csv \
        --model_id test \
        --model iTransformer \
        --data ETTh1 \
        --features M \
        --seq_len 96 --label_len 48 --pred_len 96 \
        --batch_size 8 \
        --seed "$seed" \
        --num_experts "$ne" \
        --do_cp_calibration > "$log" 2>&1
    echo "[$(date +%T)] gpu$gpu  ne=$ne seed=$seed done (rc=$?)"
}

# One worker per GPU; the MoE variant is the default (no --prob_expert / --unc_gating).
{ for seed in 4022 4023 4024 4025; do run_cell 0 1 "$seed"; done; } &
{ for seed in 4022 4023 4024 4025; do run_cell 1 3 "$seed"; done; } &
wait
echo "[$(date +%T)] all 8 CP gap cells attempted"
