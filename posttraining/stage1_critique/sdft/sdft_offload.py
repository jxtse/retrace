"""Official TRL activation offloading wrapped around the unmodified SDFTTrainer.

TRL 1.13 exposes `activation_offloading` only on SFTConfig/SFTTrainer
(sft_trainer.py: `training_step` runs under `self.maybe_activation_offload_context`).
SDFTTrainer has no such flag. This module reproduces that exact pattern for
SDFTTrainer without touching sampling, teacher construction, loss, optimizer,
or checkpointing: the official `get_act_offloading_ctx_manager` from
`trl.models.activation_offloading` is entered around `training_step`.

Everything else (`_prepare_inputs`, generation, `compute_loss`) is the
installed SDFTTrainer code path.
"""
import contextlib


def make_offload_sdft_trainer_class(base_cls):
    class OffloadSDFTTrainer(base_cls):
        def __init__(self, *args, activation_offloading=False, **kwargs):
            super().__init__(*args, **kwargs)
            if activation_offloading:
                from trl.models.activation_offloading import get_act_offloading_ctx_manager
                self.maybe_activation_offload_context = get_act_offloading_ctx_manager(
                    model=self.model, warn_if_no_head=False)
            else:
                self.maybe_activation_offload_context = contextlib.nullcontext()
            self.activation_offloading = bool(activation_offloading)

        def training_step(self, *args, **kwargs):
            with self.maybe_activation_offload_context:
                return super().training_step(*args, **kwargs)

    OffloadSDFTTrainer.__name__ = f'Offload{base_cls.__name__}'
    return OffloadSDFTTrainer
