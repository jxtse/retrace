"""OPSD infra scan: classify native-harness attempts for infrastructure failures.

Same rule as the base / teacher-SFT test40 pipelines (teachersft_scan.py):
  contaminated = SERVICE UNAVAILABLE (503) | ChemBART engine failure | connection
  refused / NewConnectionError | run wallclock budget hit | MolPort API key error,
  plus the empty-response-at-wallclock case (>=1 empty completion in the wire log).
Policy failures (no route found, iteration exhaustion, student mistakes) are NOT infra
and are kept as-is: OPSD trains on every student trajectory.
"""
import json
import re
from pathlib import Path

PATTERNS = {
    'svc503': 'SERVICE UNAVAILABLE',
    'chembart_fail': 'engine=chembart failed',
    'molport': 'Molport API key',
    'wallclock': 'Run wallclock budget',
}
CONN_RE = re.compile(r'Connection refused|NewConnectionError')


def scan_attempt(run_dir, result):
    """Per-attempt signals for one result JSON inside run_dir."""
    run_dir = Path(run_dir)
    log = run_dir / 'logs' / f'{Path(result["path"]).stem}.log'
    txt = log.read_text(errors='ignore') if log.exists() else ''
    sig = {k: txt.count(v) for k, v in PATTERNS.items()}
    sig['conn_refused'] = len(CONN_RE.findall(txt))
    sig['wallclock'] = 'Run wallclock budget' in txt
    empty = 0
    wl = Path(result.get('session_dir') or '/nonexistent') / 'conversation/wire_log.jsonl'
    if wl.exists():
        for line in wl.read_text(errors='ignore').splitlines():
            if not line.strip():
                continue
            x = json.loads(line)
            m = ((x.get('response') or {}).get('choices') or [{}])[0].get('message') or {}
            if not (m.get('content') or '').strip() and not m.get('tool_calls'):
                empty += 1
    sig['empty_responses'] = empty
    sig['contaminated'] = (sig['svc503'] > 0 or sig['chembart_fail'] > 0
                           or sig['conn_refused'] > 0 or bool(sig['wallclock'])
                           or sig['molport'] > 0
                           or (empty > 0 and bool(sig['wallclock'])))
    return sig


def collect(run_dir, tag='001'):
    """idx -> latest attempt record for one harness run directory."""
    run_dir = Path(run_dir)
    out = {}
    for p in sorted(run_dir.glob('runs/*.json')):
        r = json.loads(p.read_text())
        i = int(r['idx'])
        rec = dict(idx=i, tag=tag, success=bool(r.get('success')), iterations=r.get('iterations'),
                   wall_seconds=r.get('wall_seconds'), error=r.get('error'), path=str(p),
                   session_dir=r.get('session_dir'), run_dir=str(run_dir))
        rec.update(scan_attempt(run_dir, rec))
        out[i] = rec
    return out


def summarise(attempts):
    """Aggregate attempt scan for the driver status report."""
    n = len(attempts)
    return dict(
        n=n,
        success=sum(1 for r in attempts.values() if r['success']),
        failed=sorted(i for i, r in attempts.items() if not r['success']),
        infra_contaminated=sorted(i for i, r in attempts.items() if r['contaminated'] and not r['success']),
        success_but_contaminated=sorted(i for i, r in attempts.items() if r['success'] and r['contaminated']),
        rerun_candidates=sorted(i for i, r in attempts.items() if not r['success'] and r['contaminated']),
        wall_seconds={i: r['wall_seconds'] for i, r in attempts.items()},
    )
