#!/bin/bash
# electricity MoECP, the last MoG cell of the seed-4021 headline table.
#
# Must pass --moecp_workers: it defaults to 1 (run.py:226), and MoECP recomputes an H x C
# grid of weighted quantiles at every forecast origin -- O(W log W * H * C) per origin. The
# first attempt ran serial and produced nothing in 111 minutes. Traffic (862 channels, more
# work than electricity's 321) finished in 43 minutes with 16 workers.
#
#   bash scripts/run_elec_moecp_workers.sh [gpu]
set -u
cd "$(dirname "$0")/.."
GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

./unc_moe/bin/python -u run.py \
    --task_name long_term_forecast --is_training 0 \
    --root_path ./data/long_term_forecast/electricity/ --data_path electricity.csv \
    --model_id test --model iTransformer --data custom --features M \
    --seq_len 96 --label_len 48 --pred_len 96 --batch_size 4 \
    --enc_in 321 --dec_in 321 --c_out 321 \
    --seed 4021 --num_experts 3 --prob_expert --moecp_workers 16 \
    --do_moecp_calibration
