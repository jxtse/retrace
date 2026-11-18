"""S2b external plugin: pretokenized replay template + runtime probe.

Our own code, imported via `--external_plugins`; no framework file is edited.

1. Registers template `retrace_opsd_ptok_v1`: `encode()` ignores the row's text and
   returns the frozen ids/labels from s2b/frozen/index.json, so ms-swift cannot
   re-render (no dropped `` wrappers, no merged user turns, no tool-call
   re-formatting).  Fail-closed: unknown uid, ambiguous view, or a length that does
   not match the frozen record raises.
2. Wraps the Megatron GKD trainer to log, per microbatch, the loss-mask size, the
   padded length, teacher/student grad state and the fixed-teacher mode.
"""
import hashlib
import json
import os

import torch

from swift.template import TEMPLATE_MAPPING, register_template


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
TEMPLATE = 'retrace_opsd_ptok_v1'
INDEX_PATH = RETRACE_SHARE + '/online-arms/s2b/frozen/index.json'
DUMP_DIR = RETRACE_SHARE + '/online-arms/s2b'
Base = TEMPLATE_MAPPING['qwen3_8'].template_cls
_INDEX = None
_DUMPED = {'done': False}
_CALLS = {'n': 0, 'student': 0, 'teacher': 0}


def index():
    global _INDEX
    if _INDEX is None:
        _INDEX = json.loads(open(os.environ.get('RETRACE_INDEX', INDEX_PATH)).read())
    return _INDEX


def sha(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


class PtokTemplate(Base):

    def encode(self, inputs, return_template_inputs=False, return_length=False):
        if return_template_inputs:
            raise RuntimeError('pretokenized replay rows cannot be re-rendered')
        ctk = inputs.get('chat_template_kwargs') or {}
        uid = ctk.get('_retrace_uid')
        if uid is None:
            raise RuntimeError('row is missing chat_template_kwwards._retrace_uid')
        e = index()[uid]
        msgs = inputs.get('messages')
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        last_user = next((m for m in reversed(msgs or []) if m.get('role') == 'user'), None)
        lu_len = len((last_user or {}).get('content') or '')
        # Student and teacher views differ only by the appended privileged block; identify
        # the view by the last user message length (robust to any message re-formatting).
        if lu_len == e['last_user_len_teacher']:
            view = 'teacher'
        elif lu_len == e['last_user_len_student']:
            view = 'student'
        else:
            raise RuntimeError(f'uid={uid}: last user content length {lu_len} matches neither '
                               f'student {e["last_user_len_student"]} nor teacher '
                               f'{e["last_user_len_teacher"]}')
        ids = e[f'{view}_ids']
        labels = e[f'{view}_labels']
        if len(ids) != e[f'n_{view}']:
            raise RuntimeError(f'frozen {view} length mismatch for {uid}')
        _CALLS[view] += 1
        if os.environ.get('RETRACE_VERIFY', '1') == '1':
            if sha(ids) != e[f'{view}_ids_sha256']:
                raise RuntimeError(f'frozen {view} ids changed for {uid}')
        out = {'input_ids': list(ids), 'labels': list(labels)}
        if return_length:
            out['length'] = len(ids)
        return out


meta = TEMPLATE_MAPPING['qwen3_8']
meta.template_type = TEMPLATE
meta.template_cls = PtokTemplate
register_template(meta, exist_ok=True)
print(f'S2B plugin installed: template={TEMPLATE} index={INDEX_PATH}', flush=True)

# ---------------------------------------------------------------- runtime probe
from swift.megatron.trainers.gkd_trainer import MegatronGKDTrainer as _T  # noqa: E402

_orig_loss_func = _T.loss_func
_orig_prepare = _T._prepare_batch


def _prepare_batch(self, batch, vp_stage=None):
    labels = batch.get('labels')
    if labels is not None and hasattr(labels, 'shape'):
        n_valid = int((labels != -100).sum().item())
        seq = int(labels.shape[-1]) if labels.dim() > 1 else int(labels.numel())
        print(f'S2BPROBE prepare_batch seq_len={seq} labels_valid={n_valid}', flush=True)
        if not _DUMPED['done'] and (not torch.distributed.is_initialized()
                                    or torch.distributed.get_rank() == 0):
            _DUMPED['done'] = True
            try:
                payload = {k: v.detach().cpu() for k, v in batch.items()
                           if hasattr(v, 'detach') and k in ('input_ids', 'labels')}
                torch.save(payload, f'{DUMP_DIR}/probe_batch_rank0.pt')
                print('S2BPROBE dumped batch keys=' + ','.join(payload.keys()), flush=True)
            except Exception as e:
                print('S2BPROBE dump failed', e, flush=True)
    return _orig_prepare(self, batch, vp_stage=vp_stage)


def loss_func(self, output_tensor, *, labels, teacher_output, data_source=None):
    _CALLS['n'] += 1
    n_valid = int((labels != -100).sum().item()) if hasattr(labels, 'sum') else -1
    seq = int(labels.shape[-1]) if hasattr(labels, 'shape') and labels.dim() > 1 else -1
    t_full = getattr(teacher_output, 'full_logits', None)
    t_topk = getattr(teacher_output, 'topk_logprobs', None)
    t_rg = None
    if t_full is not None:
        t_rg = bool(t_full.requires_grad)
    elif t_topk is not None:
        t_rg = bool(t_topk.requires_grad)
    print(f'S2BPROBE loss_func call={_CALLS["n"]} seq_len={seq} labels_valid={n_valid} '
          f'teacher_requires_grad={t_rg} student_requires_grad={bool(output_tensor.requires_grad)} '
          f'teacher_mode_disable_adapter={getattr(self, "_teacher_use_disable_adapter", None)} '
          f'grad_enabled={torch.is_grad_enabled()} topk={getattr(self, "gkd_logits_topk", None)} '
          f'temperature={getattr(self, "temperature", None)} beta={getattr(self, "beta", None)}',
          flush=True)
    return _orig_loss_func(self, output_tensor, labels=labels, teacher_output=teacher_output,
                           data_source=data_source)


_T.loss_func = loss_func
_T._prepare_batch = _prepare_batch
print('S2B probe hooks installed', flush=True)

# ------------------------------------------- optional module-key dump (RETRACE_DBG_KEYS=1)
# Wraps peft's inject_adapter (the code path the Megatron bridge actually uses) so the
# real module keys can be written out; the LoRA regex must be written against those.
if os.environ.get('RETRACE_DBG_KEYS') == '1':
    import peft.tuners.tuners_utils as _TU

    _orig_inject = _TU.BaseTuner.inject_adapter
    _DBG = {'n': 0}

    def _inject_patched(self, model, *args, **kwargs):
        if _DBG['n'] == 0:
            _DBG['n'] = 1
            try:
                names = [k for k, _ in model.named_modules()]
                cand = [n for n in names if any(s in n for s in
                        ('self_attention', 'mlp', 'linear_attn', 'decoder', 'mtp',
                         'language_model', 'attention'))]
                with open(f'{DUMP_DIR}/logs/module_keys.txt', 'w') as f:
                    f.write(json.dumps({'n_modules': len(names), 'n_candidate': len(cand),
                                        'target_modules': str(
                                            getattr(model, '_s2b_tm', None)),
                                        'keys': cand[:600]}, indent=1))
                print(f'S2BDBG n_modules={len(names)} candidates={len(cand)} first={cand[:6]}',
                      flush=True)
            except Exception as e:
                print('S2BDBG dump failed', e, flush=True)
        return _orig_inject(self, model, *args, **kwargs)

    _TU.BaseTuner.inject_adapter = _inject_patched
    print('S2BDBG module-key dump installed', flush=True)
