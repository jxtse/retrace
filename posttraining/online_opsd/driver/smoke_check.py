#!/usr/bin/env python3
"""Step-1 smoke gate report for the OPSD production run (reads driver state + artifacts).

usage: python3 s3/driver/smoke_check.py [step]
Prints a JSON summary with one boolean per brief gate, plus the numbers to quote.
"""
import json
import sys
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
S3 = Path(RETRACE_SHARE + '/online-arms/s3')
step = int(sys.argv[1]) if len(sys.argv) > 1 else 1
st = json.loads((S3 / 'driver' / 'state.json').read_text())
d = st['steps'][str(step)]


def ok(phase):
    return bool(d.get(phase, {}).get('ok'))


ro = d.get('rollout', {})
inf = d.get('infra', {})
rows = d.get('rows', {})
par = d.get('parity', {})
tr = d.get('train', {})
ex = d.get('export', {})

served = d.get('serve', {})
gates = dict(
    served_model=(served.get('served'), ok('serve') and served.get('served') is not None),
    rollout_16=(f'{ro.get("summary", {}).get("success")}/{ro.get("summary", {}).get("n")}',
                ro.get('summary', {}).get('n') == 16),
    infra_rerun=(dict(n_rerun=inf.get('n_rerun'), ids=inf.get('rerun_ids'),
                      still_infra=inf.get('still_infra')), ok('infra')),
    parity=(par.get('status'), par.get('status') == 'PASS'),
    labels=(tr.get('labels'), bool(tr.get('labels', {}).get('ok'))),
    loss_grad=(dict(loss=tr.get('loss'), grad_norm=tr.get('grad_norm'),
                    peak_gib=tr.get('peak_gib_sampled'), reported_gib=tr.get('memory_gib_reported')),
               tr.get('loss') is not None and tr.get('grad_norm') is not None),
    checkpoint=(tr.get('checkpoint'), bool(tr.get('checkpoint'))),
    export=(dict(sha=ex.get('adapter_sha256'), keyset_equal=ex.get('keyset_equal'),
                 n_tensors=ex.get('n_tensors')), ok('export') and bool(ex.get('keyset_equal'))),
)
vllm_load = None
if st['steps'].get('2', {}).get('serve', {}).get('ok'):
    vllm_load = st['steps']['2']['serve']['halves']
elif st.get('eval', {}).get('eval_serve', {}).get('ok'):
    vllm_load = st['eval']['eval_serve']['halves']

times = {p: d.get(p, {}).get('seconds') for p in
         ('serve', 'rollout', 'infra', 'rows', 'parity', 'stop_serve', 'train', 'export') if ok(p)}
out = dict(step=step, gates={k: v[1] for k, v in gates.items()}, evidence={k: v[0] for k, v in gates.items()},
           phase_seconds=times, rollout_wall={k: ro.get('summary', {}).get('wall_seconds', {}).get(k)
                                              for k in ro.get('summary', {}).get('wall_seconds', {})},
           rows=dict(n_rows=rows.get('n_rows'), max_n_teacher=rows.get('max_n_teacher'),
                     total_loss_tokens=rows.get('total_loss_tokens'), padded=rows.get('padded'),
                     skipped=rows.get('skipped')),
           vllm_adapter_load=vllm_load,
           all_pass=all(v[1] for k, v in gates.items() if k != 'vllm_load'))
print(json.dumps(out, indent=1, ensure_ascii=False))
