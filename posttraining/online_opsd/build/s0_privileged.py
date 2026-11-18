"""S0 item 4: privileged-context table for the 130 train_stage2 targets.

Source (read-only, frozen): stage2_data/route_audit_v1/audit_routes.json (286 rank-1 routes)
and verdicts.jsonl (Astra Audit v1 per-step verdicts). VALID is preferred over SUSPECT.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
D = Path(RETRACE_SHARE + '/stage2-agentic/stage2_data/route_audit_v1')
SPLIT = Path(RETRACE_SHARE + '/stage2-agentic/split/SPLIT_STAGE2.json')
OUT = Path(RETRACE_SHARE + '/online-arms/data/privileged_routes.jsonl')
REPORT = Path(RETRACE_SHARE + '/online-arms/logs/s0_privileged_report.json')

PREF = {'VALID': 3, 'SUSPECT': 2, 'ERROR': 1, 'FAKE': 0}


def block_for(route, verdict):
    steps_txt = []
    vsteps = {s.get('step_index'): s for s in verdict.get('steps', [])}
    for k, st in enumerate(route['steps'], 1):
        name = st.get('reaction_name') or 'unnamed'
        steps_txt.append(f'Step {k} [{name}]: {st.get("raw", "")}')
        vs = vsteps.get(st.get('i', k - 1)) or vsteps.get(k - 1) or {}
        reason = (vs.get('reason') or '').strip()
        v = vs.get('verdict') or 'UNKNOWN'
        dc = vs.get('defect_code') or 'NONE'
        if reason:
            steps_txt.append(f'  audit {v} ({dc}): {reason}')
    head = (f'Audited passing route for this target (audit verdict {verdict.get("route_verdict")}, '
            f'rank {verdict.get("rank")}, total_score {route.get("total_score")}, '
            f'worst defect {verdict.get("worst_defect")}).')
    body = '\n'.join(steps_txt)
    summ = (verdict.get('summary') or '').strip()
    txt = head + '\n' + body + (('\nAudit summary: ' + summ) if summ else '')
    return '\n\n<privileged_audit_context>\n' + txt + '\n</privileged_audit_context>\n'


def main():
    routes = json.loads((D / 'audit_routes.json').read_text())
    verdicts = [json.loads(l) for l in (D / 'verdicts.jsonl').read_text().splitlines() if l.strip()]
    vmap = {v['uid']: v for v in verdicts}
    split = json.loads(SPLIT.read_text())
    train = split['train_stage2']
    by_idx = defaultdict(list)
    for r in routes:
        v = vmap.get(r['uid']) or {}
        by_idx[r['idx']].append((PREF.get(v.get('route_verdict'), -1), r.get('total_score') or 0.0, r, v))

    rows, no_pass, no_valid, verdicts_used = [], [], [], Counter()
    for idx in train:
        cands = by_idx.get(idx, [])
        if not cands:
            no_pass.append(idx)
            continue
        cands.sort(key=lambda t: (t[0], t[1]), reverse=True)
        best = cands[0]
        verdict = best[3].get('route_verdict')
        if verdict not in ('VALID', 'SUSPECT'):
            no_pass.append(idx)
            continue
        if not any(c[0] == 3 for c in cands):
            no_valid.append(idx)
        verdicts_used[verdict] += 1
        route, v = best[2], best[3]
        rows.append({
            'idx': int(idx), 'uid': route['uid'], 'verdict': verdict, 'rank': route['rank'],
            'total_score': route.get('total_score'), 'n_steps': len(route['steps']),
            'label': route.get('route_label'),
            'block': block_for(route, v),
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    report = {
        'n_train_targets': len(train),
        'n_with_privileged_route': len(rows),
        'verdict_used': dict(verdicts_used),
        'targets_with_any_valid': len(train) - len(no_valid),
        'targets_without_valid': no_valid,
        'targets_without_passing_route': no_pass,
        'source': {'routes': str(D / 'audit_routes.json'), 'verdicts': str(D / 'verdicts.jsonl'),
                   'split': str(SPLIT)},
        'policy': 'VALID preferred over SUSPECT; rank-1 routes only; ERROR/FAKE never used as teacher context',
    }
    REPORT.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}, indent=1))
    print('no_valid_idx', no_valid)
    print('no_pass_idx', no_pass)


if __name__ == '__main__':
    main()
