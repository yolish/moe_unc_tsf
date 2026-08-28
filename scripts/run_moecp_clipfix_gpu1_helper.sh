#!/bin/bash
# Companion to run_moecp_clipfix_full_grid.sh. Stream C (weather + exchange-rate) already
# finished on its own -- only ETTm2 ne=3,4,5 are still unstarted on GPU0 (ne=1,2 already
# done/running there). Take those three onto GPU1 in parallel.
set -u
cd "$(dirname "$0")/.."
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
GPU=1
mkdir -p logs/moecp_clipfix

run_cell() {
    local root=$1 file=$2 data=$3; shift 3
    local log="logs/moecp_clipfix/${data}_$(date +%s%N)_${STAMP}.log"
    echo "[$(date +%T)] START $data $* -> $log" >> logs/moecp_clipfix_full_grid_driver.log
    CUDA_VISIBLE_DEVICES=$GPU $PY -u run.py \
        --task_name long_term_forecast --is_training 0 \
        --root_path "$root" --data_path "$file" --data "$data" \
        --model_id test --model iTransformer --features M \
        --seq_len 96 --label_len 48 --pred_len 96 \
        --seed 4021 --prob_expert --do_moecp_calibration \
        "$@" > "$log" 2>&1
    local rc=$?
    echo "[$(date +%T)] END   $data $* rc=$rc" >> logs/moecp_clipfix_full_grid_driver.log
}

ETT_ROOT=./data/long_term_forecast/ETT/
for ne in 3 4 5; do run_cell "$ETT_ROOT" ETTm2.csv ETTm2 --batch_size 8 --num_experts "$ne"; done
echo "[$(date +%T)] gpu1 helper (ETTm2 ne3-5) attempted" >> logs/moecp_clipfix_full_grid_driver.log
