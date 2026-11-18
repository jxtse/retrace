"""Append one authorized train-only replicate after the unchanged two-replicate owner.
No requests while waiting: the existing controller.lock is acquired in blocking mode.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import run_remote as r

PLAN = r.M / 'COLLECTION_PLAN_V3.json'
QUEUE = 'REP3_QUEUE_STATUS.json'
owns_queue = False
owns_execution = False


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_plan():
    plan = json.loads(PLAN.read_text())
    split_path = r.ROOT / 'split/SPLIT_STAGE2.json'
    require(sha(split_path) == plan['split_sha256'], 'split changed')
    split = json.loads(split_path.read_text())
    ids = plan['target_ids']
    require(ids == split['train_stage2'], 'target membership or ordering changed')
    require(len(ids) == len(set(ids)) == 130, 'expected 130 unique train targets')
    require(not set(ids) & (set(split['dev']) | set(split['test40_eval_only'])), 'held-out target overlap')
    require(plan['replicates'] == ['rep1','rep2','rep3'] and plan['planned_total'] == 390, 'scope mismatch')
    require(plan['additional_trajectories'] == 130 and plan['workers'] == 6, 'unapproved expansion')
    require(plan['teacher'] == 'gpt-6-astra' and plan['scope'] == 'train_only', 'teacher/scope drift')
    require(sha(r.CONFIG) == plan['config_sha256'], 'runtime config drift')
    require(sha(Path(r.__file__)) == plan['base_runner_sha256'], 'native launch helper changed')
    return plan, ids


def scan_records(ids, reps):
    rows = []
    for rep in reps:
        folder = r.ROOT / 'main_collection' / rep / 'runs'
        expected = {f'{i:03d}_r190_{i:03d}.json' for i in ids}
        actual = {p.name for p in folder.glob('*.json')}
        require(actual == expected, f'{rep}: terminal filename coverage mismatch')
        for i in ids:
            path = r.record_path(rep, i)
            d = json.loads(path.read_text())
            require(type(d.get('idx')) is int and d['idx'] == i, f'{rep}/{i}: identity mismatch')
            rows.append({'rep': rep, 'idx': i, 'success': d.get('success'),
                         'iterations': d.get('iterations'), 'error': d.get('error'),
                         'skipped': d.get('skipped'), 'record_sha256': sha(path)})
    return rows


def check_predecessor(plan, ids):
    require(not (r.M / 'CONTROLLER_FAILED.json').exists(), 'predecessor failed')
    require(not (r.M / 'STOPPED.json').exists(), 'predecessor was stopped')
    p = r.M / 'COLLECTION_FINISHED.json'
    require(p.exists(), 'predecessor released lock without completion receipt')
    done = json.loads(p.read_text())
    require(done['terminal_records'] == 260, 'predecessor total mismatch')
    for name in plan['predecessor_phase_receipts']:
        item = r.M / name
        require(item.exists() and json.loads(item.read_text())['exit_code'] == 0,
                f'predecessor phase did not finish cleanly: {name}')
    return scan_records(ids, ['rep1','rep2'])


def update_queue(status, **extra):
    r.save(QUEUE, {'at': r.stamp(), 'pid': os.getpid(), 'entrypoint': str(Path(__file__).resolve()),
                  'status': status, 'plan_sha256': sha(PLAN), 'workers': 6,
                  'replicate': 'rep3', 'target_count': 130, **extra})


def stop_handler(signum, frame):
    raise SystemExit(128 + signum)


def main():
    global owns_queue, owns_execution
    require(sys.platform == 'linux', 'server-only entrypoint')
    guard = (r.M / 'rep3_queue.lock').open('a+')
    fcntl.flock(guard.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    owns_queue = True
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, stop_handler)
    plan, ids = read_plan()
    require(not (r.M / plan['final_receipt']).exists(), 'three-replicate collection already finished')
    prior = json.loads((r.M / 'CONTROLLER_START.json').read_text())
    update_queue('waiting_for_predecessor', predecessor_controller_pid=prior['pid'],
                 requests_started=0, additional_trajectories=130)
    lock = (r.M / 'controller.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    # Revalidate after waiting, because files may have changed while the predecessor ran.
    plan, ids = read_plan()
    previous_rows = check_predecessor(plan, ids)
    owns_execution = True
    existing = list((r.ROOT / 'main_collection/rep3/runs').glob('*.json'))
    require(not existing, 'rep3 outputs already exist; manual receipt-aware recovery required')
    r.save('REP3_PREDECESSOR_VERIFIED.json', {'at': r.stamp(), 'rows': previous_rows,
           'total': len(previous_rows), 'collection_receipt_sha256': sha(r.M / 'COLLECTION_FINISHED.json')})
    r.save('CONTROLLER_BEFORE_REP3.json', json.loads((r.M / 'CONTROLLER_START.json').read_text()))
    r.save('CONTROLLER_START.json', {'at': r.stamp(), 'pid': os.getpid(),
           'host': os.uname().nodename, 'entrypoint': str(Path(__file__).resolve()),
           'source_sha256': sha(Path(__file__)), 'plan_sha256': sha(PLAN),
           'bulk_workers': 6, 'planned_main_total': 390, 'gpu_execution': False,
           'native_phase_exit_receipts': plan['all_phase_receipts']})
    update_queue('running', requests_started='native driver governs per-case dispatch')
    r.run_phase('rep3_workers6', 'rep3', ids, 6)
    rows = scan_records(ids, plan['replicates'])
    require(len(rows) == 390, 'final coverage mismatch')
    require(rows[:260] == previous_rows, 'prior two-replicate records changed')
    r.save(plan['final_receipt'], {'at': r.stamp(), 'plan_sha256': sha(PLAN),
           'terminal_records': len(rows), 'native_success': sum(x['success'] is True for x in rows),
           'rows': rows, 'replicates': plan['replicates'],
           'native_phase_exit_receipts': plan['all_phase_receipts'],
           'legacy_quarantined_count': 12, 'legacy_included_in_main': False,
           'chemical_admission': 'pending quality screening and selective-loss preparation'})
    r.save('ACTIVE_PHASE.json', {'at': r.stamp(), 'phase': 'finished_3reps', 'workers': 0})
    update_queue('finished', terminal_records=390)


if __name__ == '__main__':
    try:
        main()
    except SystemExit as exc:
        if owns_execution:
            r.cleanup_after_stop(exc.code)
        if owns_queue:
            update_queue('stopped', exit_code=exc.code)
        raise
    except Exception as exc:
        if owns_execution and r.active is not None:
            r.cleanup_after_stop(1)
        if owns_queue:
            r.save('REP3_FAILED.json', {'at': r.stamp(), 'type': type(exc).__name__, 'error': str(exc)})
            update_queue('blocked', error=str(exc))
        raise
