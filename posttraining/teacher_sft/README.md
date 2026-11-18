# `teacher_sft/`

The direct teacher-SFT comparison arm: the same 167 teacher-trajectory rows as
stage 2, but a fresh LoRA from the base model instead of continuing from the
stage-1 adapter. `run_teacher_sft_base.sh` takes the DeepSpeed layout as an
argument (`zero3`, `zero3_offload`, `fsdp2`, `fsdp2_offload`).
