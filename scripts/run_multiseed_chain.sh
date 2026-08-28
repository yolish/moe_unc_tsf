#!/bin/bash
# Unattended continuation of the multi-seed ACI campaign, so the pipeline finishes even
# with no one logged in.
#
#   bash scripts/run_multiseed_chain.sh
#
# Phases, in order:
#   0. wait for the two in-flight drivers (CP/CPVS/CP-MoG on GPU1, MoECP on GPU0)
#   1. ACI-CQR on ETT x 4022-4025          (GPU1)  \ in parallel
#      CPVS-aleatoric on ETT x 4022-4025   (GPU0)  /
#   2. regenerate the report  <-- ETT tables are complete and usable from here on
#   3. electricity: train the 12 missing backbones, calibrate each
#   4. regenerate again, now including electricity
#
# Phase 2 sits before electricity deliberately. Electricity is the long pole (12 trainings
# from scratch) and the ETT results do not depend on it, so the report reaches a correct,
# complete-for-ETT state early rather than being held hostage to the slowest phase.
#
# Every step shells out to a resumable script or to run.py against an existing checkpoint,
# so re-running this file after an interruption repeats only outstanding work.
set -u
cd "$(dirname "$0")/.."
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/chain
CHAIN=logs/chain/chain_${STAMP}.log

say() { echo "[$(date +'%F %T')] $*" | tee -a "$CHAIN"; }

regen() {
    say "regen: collector -> multiseed CSVs -> tex"
    $PY scripts/collect_calibration_results.py            >> "$CHAIN" 2>&1
    for v in "3 MOG" "1 MOG" "1 MOE"; do
        set -- $v
        $PY scripts/build_headline_table_multiseed.py \
            --num-experts $1 --variant $2 --seeds 4021,4022,4023,4024,4025 >> "$CHAIN" 2>&1
        $PY scripts/build_headline_table.py \
            --num-experts $1 --variant $2 --seed 4021     >> "$CHAIN" 2>&1
    done
    $PY scripts/build_results_tex.py                      >> "$CHAIN" 2>&1
    $PY scripts/build_results_tex.py --multiseed \
        --out docs/results_tables_multiseed.tex           >> "$CHAIN" 2>&1
    say "regen done"
}

# ---------------------------------------------------------------- 0. wait for in-flight
say "phase 0: waiting for in-flight drivers"
while pgrep -f 'run_aci_multiseed_ett\.sh|run_aci_moecp_multiseed\.sh' > /dev/null; do
    sleep 120
done
say "phase 0 done"

# ---------------------------------------------------------------- 1. CQR + aleatoric
say "phase 1: ACI-CQR (GPU1) and CPVS-aleatoric (GPU0), in parallel"

(
  export CUDA_VISIBLE_DEVICES=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ETT="./data/long_term_forecast/ETT/"
  for seed in 4022 4023 4024 4025; do
    for ds in ETTh1 ETTh2 ETTm1 ETTm2; do
      SET="long_term_forecast_cqr_iTransformer_${ds}_ne1_pe0_ug0_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_0_seed${seed}"
      F=""
      grep -qF "$SET " result_calibration_aci_cqr_quantile_tsf.txt 2>/dev/null || F="$F --do_aci_cqr_calibration"
      grep -qF "$SET " result_calibration_aci_cqr_retrain_tsf.txt  2>/dev/null || F="$F --do_aci_cqr_retrain_calibration"
      [ -z "$F" ] && { echo "SKIP cqr ${ds} ${seed}"; continue; }
      echo "[$(date +%T)] cqr ${ds} s${seed}:$F"
      $PY -u run.py --task_name long_term_forecast --model_id cqr --model iTransformer \
          --features M --seq_len 96 --label_len 48 --pred_len 96 --seed $seed \
          --num_experts 1 --is_training 0 --use_quantile_loss \
          --aci_gamma 0.001 --aci_alpha 0.1 --batch_size 8 \
          --root_path $ETT --data_path ${ds}.csv --data $ds \
          --enc_in 7 --dec_in 7 --c_out 7 $F \
          > logs/chain/cqr_${ds,,}_s${seed}_${STAMP}.log 2>&1
    done
  done
) >> "$CHAIN" 2>&1 &
PID_CQR=$!

bash scripts/run_aci_multiseed_ett.sh 0 aleatoric_only 4022,4023,4024,4025 \
    >> "$CHAIN" 2>&1 &
PID_ALE=$!

wait $PID_CQR; say "phase 1: CQR finished"
wait $PID_ALE; say "phase 1: CPVS-aleatoric finished"

# ---------------------------------------------------------------- 2. regen (ETT complete)
say "phase 2: regen with ETT complete"
regen

# ---------------------------------------------------------------- 3. electricity
say "phase 3: electricity -- 12 backbones, train + calibrate"
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ELEC="--data custom --root_path ./data/long_term_forecast/electricity/ \
      --data_path electricity.csv --enc_in 321 --dec_in 321 --c_out 321 --batch_size 4"
# aci_moecp is absent on purpose: it runs serially (no --moecp_workers in the ACI driver),
# which is impractical at 321 channels. See FOOTNOTE_ACI in build_results_tex.py.
MOG_FLAGS="--do_aci_cp_calibration --do_aci_cpvs_calibration \
           --do_aci_aleatoric_only_calibration --do_aci_aleatoric_scale_g001_calibration"

for seed in 4022 4023 4024 4025; do
  for cfg in ne3mog ne1mog ne1moe; do
    case $cfg in
      ne3mog) ne=3; pe=1; extra="--num_experts 3 --prob_expert"; flags="$MOG_FLAGS" ;;
      ne1mog) ne=1; pe=1; extra="--num_experts 1 --prob_expert"; flags="$MOG_FLAGS" ;;
      ne1moe) ne=1; pe=0; extra="--num_experts 1";               flags="--do_aci_cp_calibration" ;;
    esac
    CK="checkpoints/long_term_forecast_test_iTransformer_electricity_ne${ne}_pe${pe}_ug0_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_0_seed${seed}"
    # Train only when the checkpoint is absent; otherwise calibrate against it.
    if [ -d "$CK" ]; then TR=0; else TR=1; fi
    say "electricity ${cfg} s${seed} (is_training=$TR)"
    $PY -u run.py --task_name long_term_forecast --model_id test --model iTransformer \
        --features M --seq_len 96 --label_len 48 --pred_len 96 --seed $seed \
        --is_training $TR --aci_gamma 0.001 --aci_alpha 0.1 \
        $ELEC $extra $flags \
        > logs/chain/elec_${cfg}_s${seed}_${STAMP}.log 2>&1
    say "  rc=$?"
  done
done

# ---------------------------------------------------------------- 4. final regen
say "phase 4: final regen"
regen
say "CHAIN COMPLETE"
