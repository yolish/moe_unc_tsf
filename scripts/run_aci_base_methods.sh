#!/bin/bash
# ACI variants of the fixed-alpha calibrators, on the MoG (--prob_expert) grid at 3 experts.
#
#   bash scripts/run_aci_base_methods.sh [gpu]
#
# gamma = 0.001, not the --aci_gamma default of 0.01: the delayed-update protocol feeds back
# errors correlated across a whole horizon, so the effective step is ~pred_len*gamma, and the
# 0.1/pred_len rule of thumb in --aci_gamma's help lands on 0.001 at pred_len=96. gamma=0.01
# drove 20-39% unbounded rates in scripts/run_aci_gc_seed4021_ne3.sh.
#
# Only four of the six ACI methods appear here. ACI CQR and ACI Retrained CQR need
# --use_quantile_loss, which is incompatible with --prob_expert (run.py prints an explicit
# skip), so they run on the separate --model_id cqr / pe0 checkpoints instead -- see
# run_aci_cqr_methods.sh. Reporting the two groups together would imply a comparison that
# is not valid.
set -u
cd "$(dirname "$0")/.."
GPU=${1:-1}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/aci_mog_ne3
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
    grep -E 'Coverage:|Avg Width|Unbounded|Final alpha_t' "$log" | tail -4
}

# ug0 only (unc_gating=0), seed 4021, 3 experts, eval-only against existing checkpoints.
BASE="--task_name long_term_forecast --model_id test --model iTransformer \
      --features M --seq_len 96 --label_len 48 --seed 4021 --num_experts 3 \
      --prob_expert --is_training 0 --aci_gamma 0.001 --aci_alpha 0.1"

# MoECP is kept at tau=1: the collector drops any other tau (coverage counts an unbounded
# interval as a hit while width is a finite-only mean, so higher tau is not comparable).
FLAGS="--do_aci_cp_calibration --do_aci_cpvs_calibration \
       --do_aci_aleatoric_only_calibration --do_aci_moecp_calibration --moecp_temperature 1.0"

ETT="./data/long_term_forecast/ETT/"

for ds in ETTh1 ETTh2 ETTm1 ETTm2; do
    run "${ds,,}" $BASE $FLAGS --batch_size 8 \
        --root_path $ETT --data_path ${ds}.csv --data $ds \
        --enc_in 7 --dec_in 7 --c_out 7
done

run exchange $BASE $FLAGS --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/exchange_rate/ --data_path exchange_rate.csv \
    --enc_in 8 --dec_in 8 --c_out 8

run illness $BASE $FLAGS --batch_size 8 --data custom --pred_len 24 \
    --root_path ./data/long_term_forecast/illness/ --data_path national_illness.csv \
    --enc_in 7 --dec_in 7 --c_out 7

# 21 channels: serial ACI MoECP is slower here than on the 7-channel sets but still
# tractable, unlike electricity (321) and traffic (862).
run weather $BASE $FLAGS --batch_size 8 --data custom \
    --root_path ./data/long_term_forecast/weather/ --data_path weather.csv \
    --enc_in 21 --dec_in 21 --c_out 21

echo "[$(date +%T)] ACI MoG ne3 grid done for the 7 small/medium datasets"
echo "[$(date +%T)] electricity + traffic skipped: ACI MoECP runs serially (--moecp_workers"
echo "               is not supported yet), which is impractical at 321/862 channels."
echo "Now regenerate the report:  $PY scripts/collect_calibration_results.py"
