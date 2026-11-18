#!/usr/bin/env bash
# --- connection/paths via env; defaults are placeholders ---
: "${RETRACE_SHARE:=/path/to/retrace-share}"
: "${RETRACE_DATA_ROOT:=/path/to/data-root}"
: "${RETRACE_OUTPUTS:=/path/to/outputs}"
: "${RETRACE_HOST:=gpu-host}"
# Stage 1 then stage 2 in one chain: train S1 on the smaller dataset first, then continue into A1.
# 1. stop base vLLM services (:18740/:18741) used for rejection sampling
# 2. S1 v2: smoke (2 steps) -> full (fresh LoRA from base, 819 audited rows, medium) -> final-adapter + checks
# 3. A1 v2: A1-medium data (167 rows), continued from S1 v2 adapter, ZeRO-3 + SP4 (same recipe as A1-swift) -> checks
set -uo pipefail
S=$RETRACE_SHARE/stage1-train
A=$RETRACE_SHARE/stage2-train
PB=$RETRACE_SHARE/stage1-selfdistill/projection_batch_v1
LOG=$S/chain_v2.log
log() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
check_adapter() {  # dir ref_dir label
  $RETRACE_SHARE/retrace-continual/venv/bin/python - "$1" "$2" "$3" >> $LOG 2>&1 <<'EOF'
import json, sys, torch
from safetensors.torch import load_file
new, ref, label = sys.argv[1:4]
n, r = load_file(new + '/adapter_model.safetensors'), load_file(ref + '/adapter_model.safetensors')
assert set(n) == set(r) and all(n[k].shape == r[k].shape for k in n), 'keys/shapes differ from reference adapter'
nb = sum('lora_B' in k for k in n); nz = sum(int(n[k].abs().sum() > 0) for k in n if 'lora_B' in k)
assert nz == nb, f'lora_B nonzero {nz}/{nb}'
c = json.load(open(new + '/adapter_config.json')); assert c['r'] == 8 and c['lora_alpha'] == 16
nm = lambda d: float(sum((v.float() ** 2).sum() for v in d.values())) ** .5
print(label, 'adapter ok: tensors', len(n), 'lora_B nonzero', nz, '/', nb, '|new|', round(nm(n), 3), '|ref|', round(nm(r), 3),
      '|new-ref|', round(float(sum(((n[k].float() - r[k].float()) ** 2).sum() for k in n)) ** .5, 3))
EOF
}
export_last() {  # out_dir
  local ck; ck=$(readlink -f $1/last-checkpoint 2>/dev/null || ls -d $1/checkpoint-* | sort -t- -k2 -n | tail -1)
  mkdir -p $1/final-adapter && cp $ck/adapter_model.safetensors $ck/adapter_config.json $1/final-adapter/ && log "final-adapter <- $ck"
}
log "chain start"
for p in 18740 18741; do
  pid=$(cat $PB/logs/vllm-$p.pid 2>/dev/null); pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  [ -n "$pgid" ] && kill -TERM -- -$pgid && log "TERM base vllm $p pgid $pgid"
done
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000' | wc -l); [ "$used" = 0 ] && break; sleep 10
done
log "gpus: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
cd $S
bash run_s1v2.sh smoke
n=$(grep -c "{'loss'" v2_smoke_zero3/train.log)
grep -q '^exit 0' v2_smoke_zero3/train.log && [ "$n" -ge 2 ] || { log "S1v2 SMOKE FAILED (loss lines $n)"; exit 1; }
log "S1v2 smoke ok; full start"
bash run_s1v2.sh full
grep -q '^exit 0' v2_full_zero3/train.log || { log "S1v2 FULL FAILED: $(tail -1 v2_full_zero3/train.log)"; exit 1; }
log "S1v2 full done: $(grep -o -E "'train_loss': '[0-9.]+'" v2_full_zero3/train.log | tail -1) steps $(grep -c "{'loss'" v2_full_zero3/train.log)"
export_last $S/v2_full_zero3
check_adapter $S/v2_full_zero3/final-adapter $PB/sft_full/final-adapter S1v2 || { log "S1v2 adapter check FAILED"; exit 1; }
cd $A
bash run_a1v2.sh zero3
grep -q '^exit 0' a1v2_full_sp4_zero3/train.log || { log "A1v2 FAILED: $(grep -E 'Error|exit' a1v2_full_sp4_zero3/train.log | tail -2)"; exit 1; }
log "A1v2 done: $(grep -o -E "'train_loss': '[0-9.]+'" a1v2_full_sp4_zero3/train.log | tail -1) steps $(grep -c "{'loss'" a1v2_full_sp4_zero3/train.log)"
export_last $A/a1v2_full_sp4_zero3
check_adapter $A/a1v2_full_sp4_zero3/final-adapter $S/v2_full_zero3/final-adapter A1v2 || { log "A1v2 adapter check FAILED"; exit 1; }
log "CHAIN DONE"
