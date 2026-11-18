#!/usr/bin/env bash
# --- connection/paths via env; defaults are placeholders ---
: "${RETRACE_SHARE:=/path/to/retrace-share}"
: "${RETRACE_DATA_ROOT:=/path/to/data-root}"
: "${RETRACE_OUTPUTS:=/path/to/outputs}"
: "${RETRACE_HOST:=gpu-host}"
# S1 GKD long-row attempt: DP=4 ZeRO-3 + gkd_logits_topk 64 (SP unsupported for gkd).
set -euo pipefail
W=$RETRACE_SHARE/online-arms
cd $W
OUT=$W/s1/gkd_dp4_topk
mkdir -p $OUT
export PYTHONPATH=$W/pylibs
export RETRACE_MANIFEST=$W/data/manifest.json RETRACE_ASSERT=warn
export CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 MASTER_PORT=29611
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$W/tmp
export HF_HUB_OFFLINE=1 MODELSCOPE_CACHE=$RETRACE_SHARE/swift-env/msc
timeout 3600 $RETRACE_SHARE/swift-env/venv-swift/bin/swift rlhf \
  --rlhf_type gkd --model $RETRACE_SHARE/models/Qwen/Qwen3.8-27B \
  --teacher_model $RETRACE_SHARE/models/Qwen/Qwen3.8-27B \
  --template retrace_opsd_replay_v1 --custom_register_path $W/scripts/retrace_template.py \
  --torch_dtype bfloat16 --tuner_type lora --lora_rank 8 --lora_alpha 16 --lora_dropout 0.05 --lora_bias none \
  --target_regex '.*\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|in_proj_qkvz|in_proj_ba|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))' \
  --dataset $W/data/rows_gkd_quad_v2.jsonl --dataset_shuffle false --split_dataset_ratio 0 \
  --lmbda 0 --gkd_logits_topk 64 --max_length 131072 --truncation_strategy delete \
  --attn_impl flash_attn --padding_free true \
  --learning_rate 2e-5 --lr_scheduler_type constant --warmup_ratio 0 --weight_decay 0 --max_grad_norm 1.0 \
  --max_steps 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --steps_per_generation 1 \
  --gradient_checkpointing true --use_liger_kernel false --sequence_parallel_size 1 \
  --optim adamw_torch --adam_beta2 0.999 --bf16 true --seed 42 --data_seed 42 \
  --logging_steps 1 --save_strategy no --report_to none --dataloader_num_workers 0 --add_version false \
  --output_dir $OUT --deepspeed zero3 > $OUT/train.log 2>&1
echo "exit $?" >> $OUT/train.log
