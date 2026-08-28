#!/bin/bash
# Reproduce Table 1 (ACI-adapted calibration intervals) for the four ETT datasets.
#
#   bash scripts/run_aci_ett.sh [gpu]
#
# Three trunks are needed because the calibrators consume different model heads:
#   MOG (ne3, --prob_expert)  -> CP-MoG, CP-MoG-a, CP-fixed
#   SG  (ne1, --prob_expert)  -> CP-G
#   CQR (ne1, --use_quantile_loss, model_id=cqr) -> CQR, CQR (retrained)
# CP-fixed is emitted by the MOG trunk since it ignores sigma(x) entirely.
set -u
cd "$(dirname "$0")/.."
GPU=${1:-0}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/aci_ett; mkdir -p "$LOGDIR"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEEDS=(4021 4022 4023 4024 4025)
DATASETS=(ETTh1 ETTh2 ETTm1 ETTm2)

# Paper config: look-back 96, horizon 96, alpha=0.1, delta=0.001, window 1000.
COMMON="--task_name long_term_forecast --model iTransformer --features M \
        --seq_len 96 --label_len 48 --pred_len 96 --batch_size 8 \
        --enc_in 7 --dec_in 7 --c_out 7 --learning_rate 0.001 \
        --is_training 1 --aci_gamma 0.001 --aci_alpha 0.1"

run() {
    local tag=$1; shift
    local log="${LOGDIR}/${tag}_${STAMP}.log"
    echo "[$(date +%T)] START $tag -> $log"
    $PY -u run.py "$@" > "$log" 2>&1
    echo "[$(date +%T)] END   $tag rc=$?"
}

for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    D="--data $ds --root_path ./data/long_term_forecast/ETT/ --data_path ${ds}.csv --seed $seed"

    # CP-MoG, CP-MoG-a, CP-fixed
    run "${ds}_s${seed}_mog" $COMMON $D --model_id test --num_experts 3 --prob_expert \
        --do_aci_cpvs_calibration --do_aci_aleatoric_scale_g001_calibration --do_aci_cp_calibration

    # CP-G
    run "${ds}_s${seed}_sg" $COMMON $D --model_id test --num_experts 1 --prob_expert \
        --do_aci_cpvs_calibration

    # CQR, CQR (retrained)
    run "${ds}_s${seed}_cqr" $COMMON $D --model_id cqr --num_experts 1 --use_quantile_loss \
        --do_aci_cqr_calibration --do_aci_cqr_retrain_calibration
  done
done
