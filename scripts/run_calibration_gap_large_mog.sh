#!/bin/bash
# Fill the missing calibration cells for weather / traffic / electricity.
#
# Scope: the MOG variant only (--prob_expert, unc_gating=0), iTransformer, ne=3,
# pl=96, seed 4021 -- i.e. exactly the three checkpoints that already exist:
#   long_term_forecast_test_iTransformer_{weather,traffic,electricity}_ne3_pe1_ug0_..._seed4021
#
# Methods added here: Aleatoric Scale CP ("CP-MOG") and MoECP. Standard CP,
# Adaptive CPVS, Aleatoric MoG and Aleatoric only were already recorded for these
# cells. CQR quantile / CQR retrain are NOT here: they need a pinball-loss
# (--use_quantile_loss, non-prob_expert) model, and no such checkpoint exists for
# these three datasets. CP-VS Aleatoric has no long-term-forecast implementation
# at all -- it only exists on the tabular_regression path.
#
# Checkpoints already exist, so this is calibration only (--is_training 0).
#
#   bash scripts/run_calibration_gap_large_mog.sh
#
# Results land in result_calibration_aleatoric_scale_tsf.txt and
# result_calibration_moecp_tsf.txt; re-run scripts/collect_calibration_results.py
# afterwards to refresh docs/calibration_results_tsf.md.
set -u
cd "$(dirname "$0")/.."
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs

run_cell() {   # gpu, dataset_dir, data_file
    local gpu=$1 dir=$2 file=$3
    local name=${file%.csv}
    local log="logs/calib_gap_${name}_mog_seed4021_${STAMP}.log"
    echo "[$(date +%T)] gpu$gpu  $name -> $log"
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
        --do_aleatoric_scale_calibration \
        --do_moecp_calibration > "$log" 2>&1
    echo "[$(date +%T)] gpu$gpu  $name done (rc=$?)"
}

# gpu1 is busy (~72G used), so weather+electricity share gpu0 and traffic --
# the heavy one, 862 channels -- gets its own lane on gpu0 too, run serially.
run_cell 0 weather     weather.csv
run_cell 0 electricity electricity.csv
run_cell 0 traffic     traffic.csv
echo "[$(date +%T)] all 3 calibration gap cells attempted"
