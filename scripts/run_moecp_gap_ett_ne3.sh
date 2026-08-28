#!/bin/bash
# Fill the missing MoECP cells for iTransformer / MoG / ne=3 / pl96: ETTh2, ETTm1,
# ETTm2 at seeds 4021-4025 (ETTh1 was already covered by the earlier sweep).
#
# Checkpoints already exist, so this is calibration only (--is_training 0).
#
#   bash scripts/run_moecp_gap_ett_ne3.sh
#
# Results land in result_calibration_moecp_tsf.txt; re-run
# scripts/collect_calibration_results.py afterwards.
set -u
cd "$(dirname "$0")/.."
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
GPU=0   # gpu1 is occupied by another job

run_cell() {   # dataset, seed
    local ds=$1 seed=$2
    local log="logs/moecp_gap_${ds}_ne3_seed${seed}_${STAMP}.log"
    echo "[$(date +%T)] $ds seed=$seed -> $log"
    CUDA_VISIBLE_DEVICES=$GPU $PY -u run.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --root_path ./data/long_term_forecast/ETT/ \
        --data_path "${ds}.csv" \
        --model_id test \
        --model iTransformer \
        --data "$ds" \
        --features M \
        --seq_len 96 --label_len 48 --pred_len 96 \
        --batch_size 8 \
        --seed "$seed" \
        --num_experts 3 \
        --prob_expert \
        --do_moecp_calibration > "$log" 2>&1
    echo "[$(date +%T)] $ds seed=$seed done (rc=$?)"
}

# --prob_expert selects the MoG variant MoECP needs (pe1 ug0).
# Two streams on the same GPU: plenty of headroom, and it halves wall time.
{ for s in 4021 4022 4023 4024 4025; do run_cell ETTh2 "$s"; done
  for s in 4021 4022 4023 4024 4025; do run_cell ETTm2 "$s"; done; } &
{ for s in 4021 4022 4023 4024 4025; do run_cell ETTm1 "$s"; done; } &
wait
echo "[$(date +%T)] all 15 MoECP ne=3 gap cells attempted"
