"""Convert offline replay rows to TITO rows.

The assistant turns keep *no* tool_calls field: their exact tokens travel in
``response_token_ids`` (per-turn ids, token-in-token-out), which is the supported ms-swift
mechanism. Without that, ms-swift's RLHF message normalization re-renders tool calls with
different formatting and breaks token parity (measured: -1609 and -13390 tokens).
"""
import copy
import json
import sys
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrace_replay as R  # noqa: E402

WORK = Path(RETRACE_SHARE + '/online-arms')


def to_tito(row):
    ids = row['input_ids']
    msgs = copy.deepcopy(row['messages'])
    spans, _ = R.assistant_spans(ids, msgs, row['tools'])
    assert len(spans) == sum(1 for m in msgs if m['role'] == 'assistant'), 'span count mismatch'
    out_msgs = []
    per_turn, per_mask = [], []
    for m in msgs:
        m2 = dict(m)
        if m2['role'] == 'assistant':
            m2.pop('tool_calls', None)
            m2.pop('reasoning_content', None)
            m2['content'] = ''
        out_msgs.append(m2)
    for _, s, e in sorted(spans):
        per_turn.append(ids[s:e])
        per_mask.append([1] * (e - s))
    row2 = dict(row, messages=out_msgs, response_token_ids=per_turn, response_loss_mask=per_mask)
    return row2


def main():
    for name in ('rows_gkd_pair.jsonl', 'rows_grpo_smoke.jsonl'):
        src = WORK / 'data' / name
        if not src.exists():
            continue
        rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
        dst = WORK / 'data' / name.replace('.jsonl', '_tito.jsonl')
        with dst.open('w') as f:
            for r in rows:
                r2 = to_tito(r)
                f.write(json.dumps(r2) + '\n')
        print(name, '->', dst.name, 'rows', len(rows),
              'turns', [len(r['response_token_ids']) for r in rows])


if __name__ == '__main__':
    main()
