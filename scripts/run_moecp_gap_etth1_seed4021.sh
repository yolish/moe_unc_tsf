#!/bin/bash
# Fill the missing MoECP cells for ETTh1 / iTransformer / MoG / pl96: seed 4021 at
# ne=1,2,4,5 (ne=3 already had it). The earlier MoECP sweep covered seeds 4022-4025.
#
# Checkpoints already exist, so this is calibration only (--is_training 0).
#
#   bash scripts/run_moecp_gap_etth1_seed4021.sh
#
# Results land in result_calibration_moecp_tsf.txt; re-run
# scripts/plot_width_vs_experts.py afterwards.
set -u
cd "$(dirname "$0")/.."
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)

run_cell() {   # gpu, num_experts
    local gpu=$1 ne=$2
    local log="logs/moecp_gap_etth1_ne${ne}_seed4021_${STAMP}.log"
    echo "[$(date +%T)] gpu$gpu  ne=$ne seed=4021 -> $log"
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
        --seed 4021 \
        --num_experts "$ne" \
        --prob_expert \
        --do_moecp_calibration > "$log" 2>&1
    echo "[$(date +%T)] gpu$gpu  ne=$ne seed=4021 done (rc=$?)"
}

# --prob_expert selects the MoG variant MoECP needs (pe1 ug0).
{ for ne in 1 4; do run_cell 0 "$ne"; done; } &
{ for ne in 2 5; do run_cell 1 "$ne"; done; } &
wait
echo "[$(date +%T)] all 4 MoECP gap cells attempted"
