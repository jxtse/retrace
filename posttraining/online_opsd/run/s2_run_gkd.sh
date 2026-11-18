#!/bin/bash
# --- connection/paths via env; defaults are placeholders ---
: "${RETRACE_SHARE:=/path/to/retrace-share}"
: "${RETRACE_DATA_ROOT:=/path/to/data-root}"
: "${RETRACE_OUTPUTS:=/path/to/outputs}"
: "${RETRACE_HOST:=gpu-host}"
# S2 OPSD smoke: one Megatron-SWIFT gkd step on one ReTRACE row.
# usage: bash scripts/s2_run_gkd.sh <layout> <dataset> <outdir> [train_iters] [extra argv...]
#   layout: tp4sp | tp2cp2 | cp4 | tp1
#   dataset: shortest | longest  (data/s2_ds_<dataset>.jsonl)
set -uo pipefail
W=$RETRACE_SHARE/online-arms
MODEL=$RETRACE_SHARE/models/Qwen/Qwen3.8-27B
LAY=$1; DS=$2; OUT=$3; ITERS=${4:-1}; shift 4 || true

case $LAY in
  tp4sp)   PAR="--tensor_model_parallel_size 4 --sequence_parallel true" ;;
  tp2cp2)  PAR="--tensor_model_parallel_size 2 --context_parallel_size 2 --sequence_parallel true" ;;
  cp4)     PAR="--context_parallel_size 4" ;;
  tp1)     PAR="--tensor_model_parallel_size 1" ;;
  *) echo "unknown layout $LAY"; exit 2 ;;
esac
case $LAY in
  tp4sp|tp1) DP_SIZE=1 ;;
  tp2cp2|cp4) DP_SIZE=1 ;;
esac
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
export PYTHONPATH=$W/pylibs
export MODELSCOPE_CACHE=$W/.mscache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

$W/venv-mcore/bin/megatron rlhf \
  --rlhf_type gkd \
  --model "$MODEL" \
  --teacher_model "$MODEL" \
  --dataset "$W/data/s2_ds_${DS}_x4.jsonl" \
  --tuner_type lora \
  --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 \
  --target_modules 'all-linear' \
  --torch_dtype bfloat16 \
  --micro_batch_size 1 --global_batch_size 4 --num_generations 4 \
  $PAR \
  --train_iters "$ITERS" \
  --lmbda 0 --beta 0.5 --sft_alpha 0 --temperature "${TEMP:-1.0}" \
  --max_length ${MAXLEN:-131072} --max_completion_length 8 \
  --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
  --gradient_accumulation_fusion false \
  --attention_backend flash \
  --save_safetensors true --merge_lora false \
  --save_steps "${SAVE_STEPS:-9999}" \
  --no_use_vllm \
  --logging_steps 1 --seed 42 \
  --output_dir "$OUT" \
  --dataloader_num_workers 0 --dataset_num_proc 1 \
  ${LOAD_OPT:-} \
  "$@"
