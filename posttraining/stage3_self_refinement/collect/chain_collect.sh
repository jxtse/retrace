#!/usr/bin/env bash
# --- connection/paths via env; defaults are placeholders ---
: "${RETRACE_SHARE:=/path/to/retrace-share}"
: "${RETRACE_DATA_ROOT:=/path/to/data-root}"
: "${RETRACE_OUTPUTS:=/path/to/outputs}"
: "${RETRACE_HOST:=gpu-host}"
# S3 collection chain per replica: rep1 then rep2.
# Usage: bash chain_collect.sh <c0|c1> <workers>
set -uo pipefail
E=$RETRACE_SHARE/stage3-collect
PY=$RETRACE_SHARE/retrace-continual/venv/bin/python
arm=$1; w=$2
log() { echo "$(date -u +%FT%TZ) $arm $*" >> $E/chain.log; }
for rep in rep1 rep2; do
  [ -f $E/STOP_AFTER_REP1 ] && [ $rep = rep2 ] && { log "STOP_AFTER_REP1 present, not starting rep2"; break; }
  log "start $rep workers=$w"
  $PY $E/launch_collect.py $arm $rep $w >> $E/chain.log 2>&1
  log "$rep exit $? results $(ls $E/arms/$arm/$rep/runs/*.json 2>/dev/null | wc -l)"
done
log "chain done"
