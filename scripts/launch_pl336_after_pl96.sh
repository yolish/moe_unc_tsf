#!/bin/bash
# Waits for the in-flight pred_len=96 run_tsf_ett.sh shards to exit, then starts the
# two queued grids interleaved -- one grid per GPU, each grid covering all 4 ETT
# datasets on its own card, so neither blocks the other.
#
# GPU 0 -> run_tsf_ett_pl336.sh      : iTransformer, pred_len=336, 400 runs
# GPU 1 -> run_tsf_ett_backbones.sh  : DLinear/PatchTST/TimeMixer, pred_len=96, 1200 runs
#
# One shell per grid. (The pl96 run accidentally launched its GPU-0 shard twice,
# which duplicated every ETTh2/ETTm2 result and raced on a shared checkpoint.pth.)

set -u
cd /home/dsi/giladaviv/moe_unc_tsf
export PATH=/home/dsi/giladaviv/moe_unc_tsf/unc_moe/bin:$PATH

PL96_PIDS=(1402100 1403623 1404416)
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=/home/dsi/giladaviv/moe_unc_tsf/logs
mkdir -p "$LOGDIR"
WAITLOG="$LOGDIR/grid_launcher_${STAMP}.log"

echo "[$(date)] waiting for pl96 shards: ${PL96_PIDS[*]}" >> "$WAITLOG"
for pid in "${PL96_PIDS[@]}"; do
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
    done
    echo "[$(date)] pl96 shard $pid exited" >> "$WAITLOG"
done

# Nothing from the pl96 grid should still be holding a GPU.
sleep 30
echo "[$(date)] all pl96 shards done; launching both grids" >> "$WAITLOG"

# pred_len=336 grid, iTransformer, all 4 datasets on GPU 0.
setsid nohup bash scripts/run_tsf_ett_pl336.sh 0 "0 1 2 3" \
    > "$LOGDIR/run_tsf_ett_pl336_gpu0_${STAMP}.log" 2>&1 < /dev/null &
echo "[$(date)] GPU0 pl336 grid (iTransformer, 400 runs) pid $!" >> "$WAITLOG"

# Backbone grid, pred_len=96, all 4 datasets on GPU 1.
setsid nohup bash scripts/run_tsf_ett_backbones.sh 1 "0 1 2 3" \
    > "$LOGDIR/run_tsf_ett_backbones_gpu1_${STAMP}.log" 2>&1 < /dev/null &
echo "[$(date)] GPU1 backbone grid (DLinear/PatchTST/TimeMixer, 1200 runs) pid $!" >> "$WAITLOG"

wait
echo "[$(date)] launcher finished" >> "$WAITLOG"
