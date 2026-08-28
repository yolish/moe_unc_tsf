#!/bin/bash
# Reproduce Table 1 (ACI-adapted calibration intervals) for traffic (862 channels).
#
#   bash scripts/run_aci_traffic.sh [gpu]
#
# Same three trunks as scripts/run_aci_ett.sh; see that file for the trunk-to-method
# mapping. Batch size is 4 rather than 8 because of the channel count.
set -u
cd "$(dirname "$0")/.."
GPU=${1:-0}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/aci_traffic; mkdir -p "$LOGDIR"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEEDS=(4021 4022 4023 4024 4025)

COMMON="--task_name long_term_forecast --model iTransformer --features M \
        --seq_len 96 --label_len 48 --pred_len 96 --batch_size 4 \
        --data custom --root_path ./data/long_term_forecast/traffic/ --data_path traffic.csv \
        --enc_in 862 --dec_in 862 --c_out 862 --learning_rate 0.001 \
        --is_training 1 --aci_gamma 0.001 --aci_alpha 0.1"

run() {
    local tag=$1; shift
    local log="${LOGDIR}/${tag}_${STAMP}.log"
    echo "[$(date +%T)] START $tag -> $log"
    $PY -u run.py "$@" > "$log" 2>&1
    echo "[$(date +%T)] END   $tag rc=$?"
}

for seed in "${SEEDS[@]}"; do
    run "traffic_s${seed}_mog" $COMMON --seed $seed --model_id test --num_experts 3 --prob_expert \
        --do_aci_cpvs_calibration --do_aci_aleatoric_scale_g001_calibration --do_aci_cp_calibration

    run "traffic_s${seed}_sg" $COMMON --seed $seed --model_id test --num_experts 1 --prob_expert \
        --do_aci_cpvs_calibration

    run "traffic_s${seed}_cqr" $COMMON --seed $seed --model_id cqr --num_experts 1 --use_quantile_loss \
        --do_aci_cqr_calibration --do_aci_cqr_retrain_calibration
done
