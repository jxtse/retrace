"""Build the fail-closed manifest for the replay template (ids sha256 + loss counts)."""
import copy
import hashlib
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


def sha(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


def entry_for(row):
    ids = row['input_ids']
    labels = row['labels']
    n_loss = sum(1 for x in labels if x != -100)
    ent = {'student': {'n_tokens': len(ids), 'n_loss': n_loss, 'ids_sha256': sha(ids)}}
    tp = row.get('teacher_prompt')
    if tp:
        msgs = copy.deepcopy(row['messages'])
        idx = max(i for i, m in enumerate(msgs) if m['role'] == 'user')
        block = tp[len(msgs[idx].get('content') or ''):]
        msgs[idx]['content'] = tp
        t_ids = R.render_ids(msgs, row['tools'], add_generation_prompt=False)
        t_spans, _ = R.assistant_spans(t_ids, msgs, row['tools'])
        t_labels = [-100] * len(t_ids)
        for _, s, e in t_spans:
            t_labels[s:e] = t_ids[s:e]
        ent['teacher'] = {'n_tokens': len(t_ids), 'n_loss': sum(1 for x in t_labels if x != -100),
                          'ids_sha256': sha(t_ids), 'block': block}
    return ent


def main():
    out = {}
    for name in ('rows_gkd_smoke_v2.jsonl', 'rows_gkd_pair_v2.jsonl', 'rows_grpo_smoke_v2.jsonl'):
        p = WORK / 'data' / name
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            out[row['uid']] = entry_for(row)
    (WORK / 'data' / 'manifest.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {'n_tokens': v['student']['n_tokens'], 'n_loss': v['student']['n_loss'],
                          'teacher': v.get('teacher', {}).get('n_tokens')} for k, v in out.items()},
                     indent=1))


if __name__ == '__main__':
    main()
