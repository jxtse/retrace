#!/usr/bin/env bash
# --- connection/paths via env; defaults are placeholders ---
: "${RETRACE_SHARE:=/path/to/retrace-share}"
: "${RETRACE_DATA_ROOT:=/path/to/data-root}"
: "${RETRACE_OUTPUTS:=/path/to/outputs}"
: "${RETRACE_HOST:=gpu-host}"
# S1 v2 rejection-sampling pilot chain.
# 1. wait for the running S1-swift test40@medium eval (both shards' exit receipts), then stop its vLLM :8000/:8002
# 2. start two base Qwen3.8-27B TP2 services (projection_batch_v1/serve.sh argv, no adapter) on :18740/:18741
# 3. pilot: 200 shuffled train questions x K=4, student prompt only, medium effort, S1 sampling; unchanged S1 gates
set -uo pipefail
RS=$RETRACE_SHARE/stage1-sampling
T=$RETRACE_SHARE/eval-stage1-legacy
PB=$RETRACE_SHARE/stage1-selfdistill/projection_batch_v1
log() { echo "$(date -u +%FT%TZ) $*" >> $RS/chain.log; }
log "chain start; waiting for test40@medium receipts"
while [ ! -f $T/arms/s1s/rollout_h0/exit_receipt.json ] || [ ! -f $T/arms/s1s/rollout_h1/exit_receipt.json ]; do sleep 60; done
log "test40@medium receipts present"
for p in 8000 8002; do
  pid=$(cat $T/serving/vllm-$p.pid); pgid=$(ps -o pgid= -p $pid | tr -d ' ')
  [ -n "$pgid" ] && kill -TERM -- -$pgid && log "TERM vllm $p pgid $pgid"
done
echo "{\"stopped_at\": \"$(date -u +%FT%TZ)\", \"reason\": \"eval complete; GPUs handed to S1 v2 rejection sampling\"}" > $T/serving/STOPPED.json
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l)
  [ "$used" = 0 ] && break; sleep 10
done
log "gpus: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
cd $PB && MAXSEQS=32 bash serve.sh
log "base services starting (MAXSEQS=32)"
for i in $(seq 1 90); do
  ok=0; for p in 18740 18741; do curl -s -m 10 http://127.0.0.1:$p/v1/models | grep -q qwen3.8-27b-thinking-selfdistill && ok=$((ok+1)); done
  [ $ok = 2 ] && break; sleep 20
done
[ $ok = 2 ] || { log "base services not ready after 30 min"; exit 1; }
log "base services ready; pilot start"
cd $RS
PY=$RETRACE_SHARE/retrace-continual/venv/bin/python
export HF_HUB_OFFLINE=1 RS_CONC=32
$PY rs_gen.py http://127.0.0.1:18740 0 2 4 200 > pilot_s0.log 2>&1 &
p0=$!
$PY rs_gen.py http://127.0.0.1:18741 1 2 4 200 > pilot_s1.log 2>&1 &
p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
log "pilot finished rc=$r0/$r1 rows=$(ls rows | wc -l) (base services left up for the full run)"
