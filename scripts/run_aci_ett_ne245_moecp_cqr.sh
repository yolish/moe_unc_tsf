#!/bin/bash
# ACI MoECP + ACI CQR (quantile and retrain) at ne in {2,4,5} on the four ETT datasets.
# Fills the sweep-table gap that scripts/run_aci_ett_ne245.sh left open: that script
# backfilled aci_cp/aci_cpvs/aci_aleatoric_only at ne 2/4/5, but aci_moecp and the two
# ACI-CQR rows were only run at {1, 3} experts (this project's headline grids) -- see
# docs/results_tables.tex Table sweep_aci, whose ne=2/4/5 rows are dashed for exactly
# these three methods.
#
# Checkpoints for ne 2/4/5 on all four ETT datasets already exist (trained during the
# ne 1-5 backbone sweep: MoG ones for MoECP, model_id=cqr ones for the CQR pair), so
# this is calibration-only (--is_training 0), same pattern as run_aci_ett_ne245.sh.
# The MoECP grid uses --prob_expert (the MoG trunk); the CQR grid uses model_id=cqr
# with --use_quantile_loss instead, which is incompatible with --prob_expert (run.py
# skips if both are set) -- same split as run_aci_cqr_methods.sh vs run_aci_base_methods.sh.
#
#   bash scripts/run_aci_ett_ne245_moecp_cqr.sh [gpu]
set -u
cd "$(dirname "$0")/.."
GPU=${1:-1}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/aci_ett_ne245_moecp_cqr
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
    grep -E 'Coverage:|Avg Width|Unbounded|Retrains:|Final alpha_t' "$log" | tail -5
}

MOG_BASE="--task_name long_term_forecast --model_id test --model iTransformer \
     --features M --seq_len 96 --label_len 48 --seed 4021 \
     --prob_expert --is_training 0 --aci_gamma 0.001 --aci_alpha 0.1 \
     --do_aci_moecp_calibration --moecp_temperature 1.0"

CQR_BASE="--task_name long_term_forecast --model_id cqr --model iTransformer \
     --features M --seq_len 96 --label_len 48 --seed 4021 \
     --is_training 0 --use_quantile_loss --aci_gamma 0.001 --aci_alpha 0.1 \
     --do_aci_cqr_calibration --do_aci_cqr_retrain_calibration"

ETT="./data/long_term_forecast/ETT/"

for ne in 2 4 5; do
    for ds in ETTh1 ETTh2 ETTm1 ETTm2; do
        run "${ds,,}_ne${ne}_moecp" $MOG_BASE --num_experts $ne \
            --pred_len 96 --batch_size 8 \
            --root_path $ETT --data_path ${ds}.csv --data $ds \
            --enc_in 7 --dec_in 7 --c_out 7
        run "${ds,,}_ne${ne}_cqr" $CQR_BASE --num_experts $ne \
            --pred_len 96 --batch_size 8 \
            --root_path $ETT --data_path ${ds}.csv --data $ds \
            --enc_in 7 --dec_in 7 --c_out 7
    done
done

echo "[$(date +%T)] ETT ne={2,4,5} ACI MoECP/CQR grid done"
echo "Now regenerate: $PY scripts/collect_calibration_results.py && $PY scripts/build_results_tex.py"
