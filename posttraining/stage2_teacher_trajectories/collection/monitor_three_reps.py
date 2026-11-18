"""Read-only three-replicate status, with backward-compatible two-replicate defaults."""
import json
from pathlib import Path
import time

# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_WORKSPACE = os.environ.get("RETRACE_WORKSPACE", "/path/to/workspace")
RETRACE_ARCHIVE = os.environ.get("RETRACE_ARCHIVE", "/path/to/archive")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")

ROOT = Path(RETRACE_SHARE + '/stage2-agentic')


def process_owned(pid, entrypoint):
    try:
        return str(entrypoint).encode() in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\x00')
    except (FileNotFoundError, ProcessLookupError):
        return False


def collect_status(root=ROOT, owner_check=process_owned):
    root = Path(root)
    m = root/'migration'
    def load(name):
        if Path(name).name != name:
            raise ValueError('invalid receipt basename')
        p = m/name
        return json.loads(p.read_text()) if p.exists() else None
    plan = load('COLLECTION_PLAN_V3.json')
    split = json.loads((root/'split/SPLIT_STAGE2.json').read_text())
    ids = split['train_stage2']
    reps = plan['replicates'] if plan else ['rep1','rep2']
    if plan and (plan['target_ids'] != ids or reps != ['rep1','rep2','rep3']):
        raise ValueError('plan/target mismatch')
    expected = {(rep,i) for rep in reps for i in ids}
    start = load('CONTROLLER_START.json') or {}
    phase = load('ACTIVE_PHASE.json') or {'phase':'not_started'}
    queue = load('REP3_QUEUE_STATUS.json') if plan else None
    main_owned = bool(start.get('pid') and owner_check(start['pid'],start.get('entrypoint',str(m/'run_remote.py'))))
    queue_owned = bool(queue and owner_check(queue['pid'],queue['entrypoint']))
    failed = load('CONTROLLER_FAILED.json') or (load('REP3_FAILED.json') if plan else None)
    stopped = load('STOPPED.json') or (queue if queue and queue.get('status')=='stopped' else None)
    final_name = plan['final_receipt'] if plan else 'COLLECTION_FINISHED.json'
    finished = load(final_name)
    found = set()
    native = 0
    bad = []
    for rep in reps:
        for p in (root/'main_collection'/rep/'runs').glob('*.json'):
            try:
                d = json.loads(p.read_text())
                key = (rep,d['idx'])
                if type(d['idx']) is not int or key not in expected or key in found:
                    bad.append(str(p));continue
                if p.name != f"{d['idx']:03d}_r190_{d['idx']:03d}.json":
                    bad.append(str(p));continue
                found.add(key)
                native += d.get('success') is True
            except (ValueError, KeyError, TypeError):
                bad.append(str(p))
    logs = [p for rep in reps for p in (root/'main_collection'/rep/'logs').glob('*.log')]
    latest = max([p.stat().st_mtime for p in logs] or [time.time()])
    receipts = plan['all_phase_receipts'] if plan else (finished or start).get('native_phase_exit_receipts',[])
    exits_ok = bool(receipts) and all(load(x) and load(x)['exit_code']==0 for x in receipts)
    complete = bool(finished and finished.get('terminal_records')==len(expected)
                    and found==expected and not bad and exits_ok
                    and not main_owned and not queue_owned and not failed and not stopped)
    shown_phase = phase['phase']
    if queue_owned and not main_owned and queue['status']=='waiting_for_predecessor':
        shown_phase = 'rep3_predecessor_handoff'
    return {'controller_alive':main_owned or queue_owned,'phase':shown_phase,
            'failed':failed,'stopped':stopped,'complete':complete,
            'terminal':len(found),'expected':len(expected),'native_success':native,
            'last_trace_age_s':time.time()-latest,'record_errors':bad,
            'canary_verified':bool(load('CANARY_VERIFIED.json')),
            'queued_replicates':['rep3'] if queue_owned and queue['status']=='waiting_for_predecessor' else [],
            'replicate_counts':{rep:sum(k[0]==rep for k in found) for rep in reps}}


if __name__=='__main__':
    print(json.dumps(collect_status(),ensure_ascii=False))
