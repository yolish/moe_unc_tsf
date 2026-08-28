#!/bin/bash
# ACI calibration across seeds on the four ETT datasets, to turn the single-seed (4021)
# ACI tables in docs/results_tables.tex into mean +/- std over 5 seeds.
#
#   bash scripts/run_aci_multiseed_ett.sh [gpu] [methods] [seeds]
#
#     methods  comma-separated subset of: cp,cpvs,aleatoric_only,aleatoric_scale_g001
#              default: cp,cpvs,aleatoric_scale_g001   (aleatoric_only is deliberately
#              NOT in the default -- it is deferred to a final pass by request)
#     seeds    comma-separated, default 4022,4023,4024,4025
#
# Resumable: before each invocation the script asks the result files which of the
# requested methods this exact setting already has, passes flags only for the missing
# ones, and skips the invocation entirely when none are missing. So re-running after an
# interruption costs only the work that is actually outstanding, and adding a method
# later (e.g. the deferred aleatoric_only pass) does not recompute the others.
#
# MoECP + ACI is not here -- it runs from run_aci_moecp_multiseed.sh, since it needs the
# 2026-08-14 clip-to-window-max rule and a re-run of seed 4021 that the others do not.
# CQR is not here either: separate --model_id cqr trunk, and it runs last by request.
set -u
cd "$(dirname "$0")/.."
GPU=${1:-1}
METHODS=${2:-cp,cpvs,aleatoric_scale_g001}
SEEDS=${3:-4022,4023,4024,4025}
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=logs/aci_multiseed_ett
mkdir -p "$LOGDIR"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# method -> result file that records it, for the has-it-already check.
resfile() {
    case $1 in
        cp)                  echo result_calibration_aci_cp_tsf.txt ;;
        cpvs)                echo result_calibration_aci_cpvs_tsf.txt ;;
        aleatoric_only)      echo result_calibration_aci_aleatoric_only_tsf.txt ;;
        aleatoric_scale_g001) echo result_calibration_aci_aleatoric_scale_g001_tsf.txt ;;
    esac
}

# method -> run.py flag.
flagfor() {
    case $1 in
        cp)                  echo --do_aci_cp_calibration ;;
        cpvs)                echo --do_aci_cpvs_calibration ;;
        aleatoric_only)      echo --do_aci_aleatoric_only_calibration ;;
        aleatoric_scale_g001) echo --do_aci_aleatoric_scale_g001_calibration ;;
    esac
}

ETT="./data/long_term_forecast/ETT/"

for seed in ${SEEDS//,/ }; do
  for ds in ETTh1 ETTh2 ETTm1 ETTm2; do
    for cfg in ne3mog ne1mog ne1moe; do

      case $cfg in
        ne3mog) ne=3; pe=1; extra="--num_experts 3 --prob_expert" ;;
        ne1mog) ne=1; pe=1; extra="--num_experts 1 --prob_expert" ;;
        ne1moe) ne=1; pe=0; extra="--num_experts 1" ;;
      esac

      SETTING="long_term_forecast_test_iTransformer_${ds}_ne${ne}_pe${pe}_ug0_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_0_seed${seed}"

      # Without --prob_expert only aci_cp is defined; run.py would skip the rest anyway,
      # so asking for them here would just produce confusing "skipped" noise in the log.
      if [ "$cfg" = "ne1moe" ]; then WANT="cp"; else WANT="$METHODS"; fi

      FLAGS=""; MISSING=""
      for m in ${WANT//,/ }; do
        f=$(resfile "$m")
        if [ -f "$f" ] && grep -qF "$SETTING " "$f"; then continue; fi
        FLAGS="$FLAGS $(flagfor "$m")"; MISSING="$MISSING,$m"
      done

      if [ -z "$FLAGS" ]; then
        echo "[$(date +%T)] SKIP  ${ds,,}_s${seed}_${cfg} (all requested methods present)"
        continue
      fi

      tag="${ds,,}_s${seed}_${cfg}"
      log="${LOGDIR}/${tag}_${STAMP}.log"
      echo "[$(date +%T)] START $tag missing=${MISSING#,} -> $log"
      t0=$SECONDS
      $PY -u run.py \
          --task_name long_term_forecast --model_id test --model iTransformer \
          --features M --seq_len 96 --label_len 48 --pred_len 96 --seed $seed \
          --is_training 0 --aci_gamma 0.001 --aci_alpha 0.1 --batch_size 8 \
          --root_path $ETT --data_path ${ds}.csv --data $ds \
          --enc_in 7 --dec_in 7 --c_out 7 \
          $extra $FLAGS > "$log" 2>&1
      rc=$?
      echo "[$(date +%T)] END   $tag rc=$rc ($((SECONDS - t0))s)"
      grep -E 'Coverage:|Avg Width|Final alpha_t' "$log" | tail -6

    done
  done
done

echo "[$(date +%T)] done: methods=$METHODS seeds=$SEEDS"
echo "Now regenerate:  $PY scripts/collect_calibration_results.py"
