#!/bin/bash
# Finish the MoECP ne=3 / pl96 gap cells across BOTH GPUs.
#
# Supersedes run_moecp_gap_ett_ne3.sh, which ran GPU0 only. Cells already in
# result_calibration_moecp_tsf.txt (ETTh2 4021-4023) and the one still in flight
# from the old script (ETTm1 4021) are not queued here.
#
#   bash scripts/run_moecp_gap_ett_ne3_2gpu.sh
#
# Checkpoints exist, so this is calibration only (--is_training 0).
# Results land in result_calibration_moecp_tsf.txt.
set -u
cd "$(dirname "$0")/.."
PY=./unc_moe/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)

QUEUE=$(mktemp);  LOCK=$(mktemp)
trap 'rm -f "$QUEUE" "$LOCK"' EXIT

# Longest-processing-time-first: the m-datasets have 11425 test samples vs 2785
# for ETTh2, so front-loading them leaves cheap cells to even out the tail.
cat > "$QUEUE" <<'EOF'
ETTm2 4021
ETTm2 4022
ETTm2 4023
ETTm2 4024
ETTm2 4025
ETTm1 4022
ETTm1 4023
ETTm1 4024
ETTm1 4025
ETTh2 4024
ETTh2 4025
EOF

# Pop one line atomically so the four workers share one queue and self-balance,
# whatever the per-GPU contention turns out to be.
pop() {
    flock 9
    local line
    line=$(head -1 "$QUEUE")
    [ -n "$line" ] && sed -i 1d "$QUEUE"
    echo "$line"
} 9<"$LOCK"

worker() {   # gpu, worker-id
    local gpu=$1 wid=$2 line ds seed log
    while :; do
        line=$(pop)
        [ -z "$line" ] && break
        ds=${line% *}; seed=${line#* }
        log="logs/moecp_gap_${ds}_ne3_seed${seed}_${STAMP}.log"
        echo "[$(date +%T)] w$wid gpu$gpu  $ds seed=$seed -> $log"
        CUDA_VISIBLE_DEVICES=$gpu $PY -u run.py \
            --task_name long_term_forecast \
            --is_training 0 \
            --root_path ./data/long_term_forecast/ETT/ \
            --data_path "${ds}.csv" \
            --model_id test \
            --model iTransformer \
            --data "$ds" \
            --features M \
            --seq_len 96 --label_len 48 --pred_len 96 \
            --batch_size 8 \
            --seed "$seed" \
            --num_experts 3 \
            --prob_expert \
            --do_moecp_calibration > "$log" 2>&1
        echo "[$(date +%T)] w$wid gpu$gpu  $ds seed=$seed done (rc=$?)"
    done
    echo "[$(date +%T)] w$wid gpu$gpu  drained"
}

# gpu1 is shared with another job (~71GB resident, ~89% util); these runs need
# only ~1-2GB each, and the shared queue keeps it from becoming the bottleneck.
worker 0 1 & worker 0 2 & worker 1 3 & worker 1 4 &
wait
echo "[$(date +%T)] queue drained - all MoECP ne=3 gap cells attempted"
