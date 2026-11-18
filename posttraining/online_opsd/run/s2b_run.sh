#!/bin/bash
# --- connection/paths via env; defaults are placeholders ---
: "${RETRACE_SHARE:=/path/to/retrace-share}"
: "${RETRACE_DATA_ROOT:=/path/to/data-root}"
: "${RETRACE_OUTPUTS:=/path/to/outputs}"
: "${RETRACE_HOST:=gpu-host}"
# S2b: one Megatron-SWIFT GKD step on one frozen row, with top-k teacher logits.
# usage: bash scripts/s2b_run.sh <tag p22k|p56k|p75k|p100k> <topk> <outdir> [iters] [extra argv...]
set -uo pipefail
W=$RETRACE_SHARE/online-arms
MODEL=$RETRACE_SHARE/models/Qwen/Qwen3.8-27B
TAG=$1; TOPK=$2; OUT=$3; ITERS=${4:-1}; shift 4 || true

# LoRA target list, Megatron module naming (dumped with RETRACE_DBG_KEYS=1,
# see s2b/logs/module_keys.txt).  Explicit full module names for all 64 decoder layers
# only, so the MTP head (mtp.fc, mtp.layers.0.*) never gets a LoRA tensor (that is what
# broke the vLLM 0.29 load of the S2 adapter).  Module set == the teacher-SFT arm:
# attention q/k/v/o (linear_qkv, linear_proj), linear-attn in_proj_qkv/z/b/a + out_proj
# (in_proj, out_proj), mlp gate/up/down (linear_fc1, linear_fc2).
# A bare `--target_modules` regex is accepted at train time but rejected at save time
# (swift/megatron/init.py:307 asserts target_modules must not be a str), hence a list;
# the flag is nargs='+' (space separated, NOT comma joined - a comma string is kept
# verbatim as one element and then matches nothing).
TARGET=$("$W/venv-mcore/bin/python" - <<'PYEOF'
import json
cfg = json.load(open('$RETRACE_SHARE/models/Qwen/Qwen3.8-27B/config.json'))['text_config']
out = []
for i, t in enumerate(cfg['layer_types']):
    pre = f'language_model.decoder.layers.{i}.'
    if t == 'full_attention':
        out += [pre + 'self_attention.linear_qkv', pre + 'self_attention.linear_proj']
    else:
        out += [pre + 'self_attention.in_proj', pre + 'self_attention.out_proj']
    out += [pre + 'mlp.linear_fc1', pre + 'mlp.linear_fc2']
print(' '.join(out))
PYEOF
)

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
  --dataset "$W/s2b/data/ds_${TAG}.jsonl" \
  --template retrace_opsd_ptok_v1 \
  --external_plugins "$W/scripts/s2b_template.py" \
  --tuner_type lora \
  --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 \
  --target_modules $TARGET \
  --torch_dtype bfloat16 \
  --micro_batch_size 1 --global_batch_size 4 --num_generations 4 \
  --tensor_model_parallel_size 4 --sequence_parallel true \
  --train_iters "$ITERS" \
  --lmbda 0 --beta 0.5 --sft_alpha 0 --temperature "${TEMP:-1.0}" \
  --gkd_logits_topk "$TOPK" \
  --loss_scale default \
  --max_length ${MAXLEN:-131072} --max_completion_length 8 \
  --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
  --gradient_accumulation_fusion false \
  --attention_backend flash \
  --save_safetensors true --merge_lora false \
  --save_steps ${SAVE_STEPS:-9999} \
  --no_use_vllm \
  --logging_steps 1 --seed 42 \
  --output_dir "$OUT" \
  --dataloader_num_workers 0 --dataset_num_proc 1 \
  "$@"
