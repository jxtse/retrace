#!/bin/bash
# --- connection/paths via env; defaults are placeholders ---
: "${RETRACE_SHARE:=/path/to/retrace-share}"
: "${RETRACE_DATA_ROOT:=/path/to/data-root}"
: "${RETRACE_OUTPUTS:=/path/to/outputs}"
: "${RETRACE_HOST:=gpu-host}"
# OPSD (GKD fixed self-teacher) training step launcher.
#
# usage: opsd_run_train.sh <dataset.jsonl> <index.json> <outdir> <train_iters> [resume_ckpt]
#
# Reuses the S2b recipe unchanged (scripts/s2b_run.sh) with three deltas:
#   * --global_batch_size 16 (one optimizer step per 16-row batch),
#   * --save_steps 1 (every step is checkpointed so the next step can resume),
#   * optional --mcore_adapter <ckpt> to continue optimizer state + step counter.
# Template/index/env are ours; no framework file is edited.
set -uo pipefail
W=$RETRACE_SHARE/online-arms
MODEL=$RETRACE_SHARE/models/Qwen/Qwen3.8-27B
DATASET=$1; INDEX=$2; OUT=$3; ITERS=$4; RESUME=${5:-} EXTRAS=("${@:6}")
GBS=${GBS:-16}
TOPK=${TOPK:-64}

# 192 explicit mcore LoRA module names (same set as the S2b run and the teacher-SFT adapter):
# 64 decoder layers x (qkv+proj | in_proj+out_proj) + (fc1+fc2); MTP head excluded.
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
export RETRACE_INDEX="$INDEX"
export RETRACE_VERIFY=1
echo "TRAIN_START $(date -u +%FT%TZ) iters=$ITERS gbs=$GBS topk=$TOPK resume=${RESUME:-none}"

RESUME_ARG=()
if [ -n "$RESUME" ]; then
  # --no_finetune is required for the mcore backend: MegatronArguments.finetune defaults
  # to True, which makes _load_checkpoint skip both the checkpoint iteration counter and
  # the optimizer-state restore. With --no_finetune the run continues at iteration s
  # (verified on the p22k row: resumed 2 -> logs 3/3 and takes exactly one step).
  RESUME_ARG=(--mcore_adapter "$RESUME" --no_finetune)
fi

"$W/venv-mcore/bin/megatron" rlhf \
  --rlhf_type gkd \
  --model "$MODEL" \
  --teacher_model "$MODEL" \
  --dataset "$DATASET" \
  --template retrace_opsd_ptok_v1 \
  --external_plugins "$W/scripts/s2b_template.py" \
  --tuner_type lora \
  --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 \
  --target_modules $TARGET \
  --torch_dtype bfloat16 \
  --micro_batch_size 1 --global_batch_size "$GBS" \
  --tensor_model_parallel_size 4 --sequence_parallel true \
  --train_iters "$ITERS" \
  --lmbda 0 --beta 0.5 --sft_alpha 0 --temperature 1.0 \
  --gkd_logits_topk "$TOPK" \
  --loss_scale default \
  --max_length 131072 --max_completion_length 8 \
  --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
  --gradient_accumulation_fusion false \
  --attention_backend flash \
  --save_safetensors true --merge_lora false \
  --save_steps 1 \
  --no_use_vllm \
  --logging_steps 1 --seed 42 \
  --output_dir "$OUT" \
  --dataloader_num_workers 0 --dataset_num_proc 1 \
  --optimizer adam --lr 2e-5 --min_lr 2e-5 --lr_decay_style constant \
  --lr_warmup_iters 0 --weight_decay 0.0 --clip_grad 1.0 \
  "${RESUME_ARG[@]}" "${EXTRAS[@]}"
RC=$?
echo "TRAIN_EXIT $RC $(date -u +%FT%TZ)"
exit $RC
