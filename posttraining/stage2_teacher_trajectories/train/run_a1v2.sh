#!/usr/bin/env bash
# --- connection/paths via env; defaults are placeholders ---
: "${RETRACE_SHARE:=/path/to/retrace-share}"
: "${RETRACE_DATA_ROOT:=/path/to/data-root}"
: "${RETRACE_OUTPUTS:=/path/to/outputs}"
: "${RETRACE_HOST:=gpu-host}"
# A1 v2: A1-medium data, continued from the S1 v2 (ms-swift) adapter instead of the TRL S1 adapter.
# ms-swift A1 smoke: 2 optimizer steps on the 8 longest rows (worst-case memory), native swift options only.
# Usage: bash run_a1v2.sh <zero3|zero3_offload|fsdp2|fsdp2_offload>
set -euo pipefail
B=$RETRACE_SHARE/swift-env
N=$RETRACE_SHARE/stage2-train
V=${1:-ddp}
cd $B
OUT=$N/a1v2_full_sp4_$V
mkdir -p $OUT
# longest 8 rows (2 steps x 4 ranks), same selection rule as A1 smoke
[ -f smoke_rows.jsonl ] || venv-swift/bin/python - <<'EOF'
import json
rows=[l for l in open('train_swift.jsonl')]
L=[(len(json.dumps(json.loads(l)['messages'])),i) for i,l in enumerate(rows)]
keep=[i for _,i in sorted(L,reverse=True)[:8]]
open('smoke_rows.jsonl','w').writelines(rows[i] for i in keep)
print('smoke rows',len(keep))
EOF
EXTRA=()
case $V in
  zero3) EXTRA=(--deepspeed zero3) ;;
  zero3_offload) EXTRA=(--deepspeed zero3_offload) ;;
  fsdp2) EXTRA=(--fsdp fsdp2) ;;
  fsdp2_offload) python3 - <<'EOF'
import json
c=json.load(open('venv-swift/lib/python3.12/site-packages/swift/config/fsdp2.json'))
c['fsdp_config']['activation_cpu_offload']=True
json.dump({'fsdp':c['fsdp'],'fsdp_config':c['fsdp_config']},open('fsdp2_offload.json','w'),indent=1)
EOF
     EXTRA=(--fsdp $B/fsdp2_offload.json) ;;
esac
export CELOSS_PARALLEL_SIZE=2048 CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 MASTER_PORT=29579
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$B/tmp HF_HUB_OFFLINE=1 MODELSCOPE_CACHE=$B/msc
timeout 36000 venv-swift/bin/swift sft \
  --model $RETRACE_SHARE/models/Qwen/Qwen3.8-27B --template qwen3_8 --torch_dtype bfloat16 \
  --adapters $RETRACE_SHARE/stage1-train/v2_full_zero3/final-adapter \
  --dataset $N/train_a1_medium.jsonl --dataset_shuffle false --split_dataset_ratio 0 \
  --loss_scale ignore_empty_think --add_non_thinking_prefix true \
  --max_length 131072 --truncation_strategy delete --attn_impl flash_attn --padding_free true \
  --learning_rate 2e-5 --lr_scheduler_type constant --warmup_ratio 0 --weight_decay 0 --max_grad_norm 1.0 \
  --num_train_epochs 1 --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --gradient_checkpointing true --use_liger_kernel false --sequence_parallel_size 4 --optim adamw_torch --adam_beta2 0.999 --bf16 true --seed 42 --data_seed 42 \
  --logging_steps 1 --save_strategy steps --save_steps 10 --save_total_limit 3 --save_only_model true --report_to none --dataloader_num_workers 0 --add_version false \
  --output_dir $OUT "${EXTRA[@]}" > $OUT/train.log 2>&1
echo "exit $?" >> $OUT/train.log
