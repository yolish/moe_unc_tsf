#!/bin/bash
# ACI CP / CPVS / aleatoric-only at ne in {2,4,5} on the four ETT datasets (MOG, --prob_expert).
# Fills the sweep-table gap: aci_aleatoric_scale_g001 (CP-MoG + ACI) already has the full
# 1-5 expert sweep, but aci_cp/aci_cpvs/aci_aleatoric_only were only run at {1, 3} experts
# (this project's headline grids) -- see docs/results_tables.tex Table 7 dashes. Checkpoints
# for ne 2/4/5 on all four ETT datasets already exist (trained during the ne 1-5 backbone
# sweep), so this is calibration-only (--is_training 0), same pattern as scripts/run_aci_ne1_mog.sh.
#
#   bash scripts/run_aci_ett_ne245.sh [gpu]
set -u
cd "$(dirname "$0")/.."
GPU=${1:-1}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/aci_ett_ne245
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
    grep -E 'Coverage:|Avg Width|Unbounded|Final alpha_t' "$log" | tail -5
}

MOG_BASE="--task_name long_term_forecast --model_id test --model iTransformer \
     --features M --seq_len 96 --label_len 48 --seed 4021 \
     --prob_expert --is_training 0 --aci_gamma 0.001 --aci_alpha 0.1 \
     --do_aci_cp_calibration --do_aci_cpvs_calibration --do_aci_aleatoric_only_calibration"

ETT="./data/long_term_forecast/ETT/"

for ne in 2 4 5; do
    for ds in ETTh1 ETTh2 ETTm1 ETTm2; do
        run "${ds,,}_ne${ne}_mog" $MOG_BASE --num_experts $ne \
            --pred_len 96 --batch_size 8 \
            --root_path $ETT --data_path ${ds}.csv --data $ds \
            --enc_in 7 --dec_in 7 --c_out 7
    done
done

echo "[$(date +%T)] ETT ne={2,4,5} ACI CP/CPVS/aleatoric-only grid done"
echo "Now regenerate: $PY scripts/collect_calibration_results.py && $PY scripts/build_results_tex.py"
