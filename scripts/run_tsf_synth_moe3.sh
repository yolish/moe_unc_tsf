#!/usr/bin/env bash
# Every time-series calibration method on SynthMoE3, the 3-regime synthetic series from
# scripts/make_synth_moe3.py (generate it first; verify with scripts/verify_synth_moe3.py).
#
# Usage: bash scripts/run_tsf_synth_moe3.sh <gpu> [num_experts...] [-- extra run.py args]
#   bash scripts/run_tsf_synth_moe3.sh 0          # ne=3 only (the DGP's true expert count)
#   bash scripts/run_tsf_synth_moe3.sh 0 1 2 3 5  # sweep, to see the ne=3 elbow
#
# The 14 methods split into three invocations because of two hard constraints in run.py:
#
#   Stage 1  --prob_expert   12 methods. All the aleatoric/epistemic ones need the MoG
#                            variance decomposition, which only exists under prob_expert.
#   Stage 2  --use_quantile_loss  CQR. Pinball loss is mutually exclusive with prob_expert
#                            (run.py exits on the combination), and the quantile head has
#                            a different output width, so it needs its own model_id or it
#                            collides with stage 1's checkpoint.
#   Stage 3  --use_quantile_loss, is_training=0   CQR retrain. Shares stage 2's setting
#                            string exactly, so with is_training=1 run.py would find
#                            stage 2's results/ dir and silently skip. Evaluate-only
#                            reuses that checkpoint and always runs the retrain path.
#
# ug is left at 0 throughout (no --unc_gating): the repo reports ug0.
export CUDA_VISIBLE_DEVICES=$1
shift

num_experts=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
  num_experts+=("$1"); shift
done
[[ "$1" == "--" ]] && shift
extra_args=("$@")
[[ ${#num_experts[@]} -eq 0 ]] && num_experts=(3)

PY=${PYTHON_BIN:-./unc_moe/bin/python}

model_name="iTransformer"
dataset="SynthMoE3"
root_path="./data/long_term_forecast/synthetic/"
data_path="synth_moe3.csv"
model_id="test"
cqr_model_id="cqr"
features="M"
seq_len=96
label_len=48
pred_len=96
batch_size=8
n_channels=7
seeds=(${SEEDS:-4021})

log_dir="logs/synth_moe3"
mkdir -p "$log_dir"
stamp=$(date +%Y%m%d_%H%M%S)

common=(
  --task_name long_term_forecast
  --root_path "$root_path"
  --data_path "$data_path"
  --model "$model_name"
  --data "$dataset"
  --features "$features"
  --seq_len "$seq_len"
  --label_len "$label_len"
  --pred_len "$pred_len"
  --batch_size "$batch_size"
  --enc_in "$n_channels"
  --dec_in "$n_channels"
  --c_out "$n_channels"
  --freq h
)

for seed in "${seeds[@]}"; do
  for ne in "${num_experts[@]}"; do

    echo "=== SynthMoE3 ne$ne seed$seed :: stage 1 (prob_expert, 12 methods) ==="
    $PY -u run.py "${common[@]}" \
      --is_training 1 \
      --model_id "$model_id" \
      --seed "$seed" \
      --num_experts "$ne" \
      --prob_expert \
      --do_cp_calibration \
      --do_cpvs_calibration \
      --do_aleatoric_mog_calibration \
      --do_aleatoric_mog_calibration_second_option \
      --do_aleatoric_only_calibration \
      --do_aleatoric_scale_calibration \
      --do_aci_aleatoric_scale_calibration \
      --do_aci_aleatoric_scale_g001_calibration \
      --do_adaptive_variance_calibration \
      --do_adaptive_window_calibration \
      --do_cp_dvs_calibration \
      --do_moecp_calibration \
      "${extra_args[@]}" \
      > "$log_dir/stage1_ne${ne}_seed${seed}_${stamp}.log" 2>&1
    echo "    -> $log_dir/stage1_ne${ne}_seed${seed}_${stamp}.log (exit $?)"

    echo "=== SynthMoE3 ne$ne seed$seed :: stage 2 (quantile head, CQR) ==="
    $PY -u run.py "${common[@]}" \
      --is_training 1 \
      --model_id "$cqr_model_id" \
      --seed "$seed" \
      --num_experts "$ne" \
      --use_quantile_loss \
      --do_cqr_calibration \
      "${extra_args[@]}" \
      > "$log_dir/stage2_cqr_ne${ne}_seed${seed}_${stamp}.log" 2>&1
    echo "    -> $log_dir/stage2_cqr_ne${ne}_seed${seed}_${stamp}.log (exit $?)"

    echo "=== SynthMoE3 ne$ne seed$seed :: stage 3 (CQR retrain, reuses stage 2 ckpt) ==="
    $PY -u run.py "${common[@]}" \
      --is_training 0 \
      --model_id "$cqr_model_id" \
      --seed "$seed" \
      --num_experts "$ne" \
      --use_quantile_loss \
      --do_cqr_retrain_calibration \
      "${extra_args[@]}" \
      > "$log_dir/stage3_cqrretrain_ne${ne}_seed${seed}_${stamp}.log" 2>&1
    echo "    -> $log_dir/stage3_cqrretrain_ne${ne}_seed${seed}_${stamp}.log (exit $?)"

  done
done
