#!/bin/bash
# --- connection/paths via env; defaults are placeholders ---
: "${RETRACE_SHARE:=/path/to/retrace-share}"
: "${RETRACE_DATA_ROOT:=/path/to/data-root}"
: "${RETRACE_OUTPUTS:=/path/to/outputs}"
: "${RETRACE_HOST:=gpu-host}"
# Launch the OPSD driver detached (setsid nohup). The driver itself holds the flock and
# writes s3/driver/driver.pid, so a second launch exits with "driver already running".
# usage: bash s3/scripts/opsd_launch.sh [from_step]
set -uo pipefail
W=$RETRACE_SHARE/online-arms
PY=$W/venv-mcore/bin/python
mkdir -p "$W/s3/driver"
cd "$W" || exit 1
if [ "${1:-}" = "" ]; then
  setsid nohup "$PY" s3/driver/opsd_driver.py run >> "$W/s3/driver/driver.log" 2>&1 < /dev/null &
else
  setsid nohup "$PY" s3/driver/opsd_driver.py run --from-step "$1" >> "$W/s3/driver/driver.log" 2>&1 < /dev/null &
fi
sleep 3
echo "DRIVER_PID=$(cat "$W/s3/driver/driver.pid" 2>/dev/null) LOG=$W/s3/driver/driver.log"
