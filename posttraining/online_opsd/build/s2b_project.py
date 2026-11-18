"""S2b wall-time projection for an OPSD loop over 128 targets.

Inputs are all measured, not assumed:
  * train s/step per row size and k  (s2b/runs/*/v*/logging.jsonl, 4 micro-batches per step)
  * export seconds                   (s2b/logs/export_p22k*.log)
  * trainer startup seconds          (job wall - step time, s2b_run_with_mem logs)
  * vLLM ready seconds               (s2b/vllm_8032_ready.json)
  * rollout wall per trajectory      (S3 collection wall_seconds over 260 trajectories,
                                      16 concurrent on 2 TP2 replicas)

Usage: venv-mcore/bin/python scripts/s2b_project.py
"""
import glob
import json
from math import ceil
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
W = Path(RETRACE_SHARE + '/online-arms')
TARGETS = 128
SLOTS = 16          # 8 workers x 2 TP2 replicas (S3 launch_collect)


def run_metrics():
    out = {}
    # sorted so the later (warm) run overwrites the cold one for the same tag
    for p in sorted(glob.glob(str(W / 's2b/runs/*/v*/logging.jsonl'))):
        tag = p.split('/runs/')[1].split('/')[0]
        for line in open(p):
            d = json.loads(line)
            if 'train_speed(s/it)' in d:
                out[tag] = {'s_per_step': round(d['train_speed(s/it)'], 1),
                            'mem_gib': d['memory(GiB)'], 'loss': d['loss'],
                            'grad_norm': d['grad_norm']}
    return out


def main():
    runs = run_metrics()
    # choose the last completed measurement per (tag, k) (warm runs are later)
    s_per_step_4mb = {'p22k': {}, 'p56k': {}, 'p75k': {}, 'p100k': {}}
    for tag, m in sorted(runs.items()):
        row, k = tag.split('_topk')
        if row in s_per_step_4mb:
            s_per_step_4mb[row][k] = m['s_per_step']
    chosen_k = 64
    per_row_train = {r: round(v[str(chosen_k)] / 4, 1) for r, v in s_per_step_4mb.items()
                     if str(chosen_k) in v}
    # median length row (p56k = 56,046 tokens) is the typical training row
    typical_row_s = per_row_train['p56k']

    export_s = 57
    startup_s = 85
    vllm_ready_s = 176
    roll = json.loads((W / 's2b/logs/rollout_wall.json').read_text())

    proj = {}
    for B in (8, 16, 32):
        n_steps = ceil(TARGETS / B)
        for stat in ('mean', 'median', 'p90'):
            w = roll[stat] * 3600
            waves = ceil(B / SLOTS)
            rollout_total = n_steps * waves * w
            train_total = TARGETS * typical_row_s + n_steps * startup_s
            turn_total = n_steps * (export_s + vllm_ready_s)
            total = rollout_total + train_total + turn_total
            proj[f'B{B}_{stat}'] = {
                'n_steps': n_steps, 'waves_per_step': waves,
                'rollout_h': round(rollout_total / 3600, 1),
                'train_h': round(train_total / 3600, 1),
                'turnaround_h': round(turn_total / 3600, 1),
                'total_h': round(total / 3600, 1),
            }
    out = {'per_row_train_s_at_topk64': per_row_train, 'typical_row_s': typical_row_s,
           'export_s': export_s, 'startup_s': startup_s, 'vllm_ready_s': vllm_ready_s,
           'rollout_wall_h': roll, 'projection': proj,
           'measured_s_per_step_4microbatches': s_per_step_4mb}
    (W / 's2b/logs/s2b_projection.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
