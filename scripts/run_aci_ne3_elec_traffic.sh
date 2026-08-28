#!/bin/bash
# Fill the last two ne3 (MOG, 3-expert) gaps: electricity and traffic.
#
#   bash scripts/run_aci_ne3_elec_traffic.sh [gpu]
#
# ACI-MoECP is deliberately excluded: it runs serially and is impractical at 321/862
# channels (confirmed with the user). The other 5 ACI methods have no such blocker.
set -u
cd "$(dirname "$0")/.."
GPU=${1:-1}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/aci_ne3_elec_traffic
mkdir -p "$LOGDIR"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
    local tag=$1; shift
    local log="${LOGDIR}/${tag}_${STAMP}.log"
    echo "[$(date +%T)] START $tag -> $log"
    local t0=$SECONDS
    $PY -u run.py "$@" > "$log" 2>&1
    local rc=$?
    echo "[$(date +%T)] END   $tag rc=$rc ($((SECONDS - t0))s)"
    grep -E 'Coverage:|Avg Width|Retrains:|Final alpha_t' "$log" | tail -4
}

MOG="--task_name long_term_forecast --model_id test --model iTransformer \
     --features M --seq_len 96 --label_len 48 --pred_len 96 --seed 4021 --num_experts 3 \
     --prob_expert --is_training 0 --aci_gamma 0.001 --aci_alpha 0.1 \
     --do_aci_cp_calibration --do_aci_cpvs_calibration --do_aci_aleatoric_only_calibration"

CQR="--task_name long_term_forecast --model_id cqr --model iTransformer \
     --features M --seq_len 96 --label_len 48 --pred_len 96 --seed 4021 --num_experts 3 \
     --is_training 0 --use_quantile_loss --aci_gamma 0.001 --aci_alpha 0.1 \
     --do_aci_cqr_calibration --do_aci_cqr_retrain_calibration"

run electricity_mog $MOG --batch_size 4 --data custom \
    --root_path ./data/long_term_forecast/electricity/ --data_path electricity.csv \
    --enc_in 321 --dec_in 321 --c_out 321

run electricity_cqr $CQR --batch_size 4 --data custom \
    --root_path ./data/long_term_forecast/electricity/ --data_path electricity.csv \
    --enc_in 321 --dec_in 321 --c_out 321

run traffic_mog $MOG --batch_size 4 --data custom \
    --root_path ./data/long_term_forecast/traffic/ --data_path traffic.csv \
    --enc_in 862 --dec_in 862 --c_out 862

run traffic_cqr $CQR --batch_size 4 --data custom \
    --root_path ./data/long_term_forecast/traffic/ --data_path traffic.csv \
    --enc_in 862 --dec_in 862 --c_out 862

echo "[$(date +%T)] ne3 electricity/traffic ACI gap-fill done"
echo "Now regenerate the report:  $PY scripts/collect_calibration_results.py"
