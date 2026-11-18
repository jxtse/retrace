#!/usr/bin/env bash
# --- connection/paths via env; defaults are placeholders ---
: "${RETRACE_SHARE:=/path/to/retrace-share}"
: "${RETRACE_DATA_ROOT:=/path/to/data-root}"
: "${RETRACE_OUTPUTS:=/path/to/outputs}"
: "${RETRACE_HOST:=gpu-host}"
# S1 v2: same as run_s1.sh except data = train_s1v2_swift.jsonl (819 audited pass/minor rows, medium) and out dir v2_*.
# S1 is re-trained on ms-swift from the BASE model.
# Data: train_s1v2_swift.jsonl = 819 rows (788 audited pass/minor S1 rows + 31 audited base rejection samples), build_s1v2.py
#       (convert_s1.py; mask identical except swift also supervises '<think>' '\n' after the assistant header).
# Recipe = S1 (projection_batch_v1/train.py): fresh LoRA r8/a16/dropout0.05, same target regex, LR 2e-5 constant,
#       adamw_torch, wd 0, max_grad_norm 1, 1 epoch, bf16, grad ckpt, per-device 1 x 4 ranks x GA1 = global batch 4.
# Framework settings = A1-swift run (template qwen3_8, loss_scale ignore_empty_think, add_non_thinking_prefix,
#       flash_attn, liger off). Memory: rows <= 7977 tokens, so plain data parallel with DeepSpeed ZeRO-3 (no SP).
# Usage: bash run_s1v2.sh <smoke|full>
set -euo pipefail
B=$RETRACE_SHARE/stage1-train
V=$RETRACE_SHARE/swift-env/venv-swift
STAGE=${1:?smoke|full}
cd $B
OUT=$B/v2_${STAGE}_zero3
mkdir -p $OUT
if [ "$STAGE" = smoke ]; then
  [ -f v2_smoke_rows.jsonl ] || $V/bin/python - <<'EOF'
import json
rows=open('train_s1v2_swift.jsonl').readlines()
L=sorted(range(len(rows)),key=lambda i:-len(rows[i]))[:8]
open('v2_smoke_rows.jsonl','w').writelines(rows[i] for i in L)
EOF
  DATA=$B/v2_smoke_rows.jsonl; EXTRA=(--max_steps 2 --save_strategy no)
else
  DATA=$B/train_s1v2_swift.jsonl; EXTRA=(--save_strategy steps --save_steps 50 --save_total_limit 2 --save_only_model true)
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 MASTER_PORT=29575
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$RETRACE_SHARE/swift-env/tmp HF_HUB_OFFLINE=1 MODELSCOPE_CACHE=$RETRACE_SHARE/swift-env/msc
set +e
timeout 14400 $V/bin/swift sft \
  --model $RETRACE_SHARE/models/Qwen/Qwen3.8-27B --template qwen3_8 --torch_dtype bfloat16 \
  --tuner_type lora --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 --lora_bias none \
  --target_regex '.*\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|in_proj_qkvz|in_proj_ba|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))' \
  --dataset $DATA --split_dataset_ratio 0 \
  --loss_scale ignore_empty_think --add_non_thinking_prefix true \
  --max_length 16384 --truncation_strategy delete --attn_impl flash_attn \
  --learning_rate 2e-5 --lr_scheduler_type constant --warmup_ratio 0 --weight_decay 0 --max_grad_norm 1.0 \
  --optim adamw_torch --adam_beta2 0.999 \
  --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --gradient_checkpointing true --use_liger_kernel false --bf16 true --seed 42 --data_seed 42 \
  --logging_steps 1 --report_to none --dataloader_num_workers 0 --add_version false \
  --deepspeed zero3 --output_dir $OUT "${EXTRA[@]}" > $OUT/train.log 2>&1
echo "exit $?" >> $OUT/train.log
