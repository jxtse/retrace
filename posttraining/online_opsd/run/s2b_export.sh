#!/bin/bash
# --- connection/paths via env; defaults are placeholders ---
: "${RETRACE_SHARE:=/path/to/retrace-share}"
: "${RETRACE_DATA_ROOT:=/path/to/data-root}"
: "${RETRACE_OUTPUTS:=/path/to/outputs}"
: "${RETRACE_HOST:=gpu-host}"
# S2b export: mcore LoRA checkpoint -> HF/PEFT adapter (no merge), timing recorded.
# usage: bash scripts/s2b_export.sh <ckpt_dir> <out_dir>
# NOTE: the checkpoint args.json records --template retrace_opsd_ptok_v1, so the export
# run must import the same external plugin or argument validation fails on the template.
set -uo pipefail
W=$RETRACE_SHARE/online-arms
CKPT=$1; OUT=$2
export CUDA_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export PYTHONPATH=$W/pylibs
export MODELSCOPE_CACHE=$W/.mscache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
T0=$(date +%s)
$W/venv-mcore/bin/megatron export \
  --adapters "$CKPT" \
  --to_hf true --merge_lora false \
  --tensor_model_parallel_size 2 \
  --torch_dtype bfloat16 \
  --output_dir "$OUT" \
  --model $RETRACE_SHARE/models/Qwen/Qwen3.8-27B \
  --external_plugins "$W/scripts/s2b_template.py"
RC=$?
T1=$(date +%s)
echo "EXPORT_SECONDS=$((T1 - T0)) EXPORT_RC=$RC"
find "$OUT" -maxdepth 2 | head -20
