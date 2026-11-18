"""Resume the authorized Astra-only CPU collection on the existing GPU host.
One remaining rep1 target is the full-protocol migration canary, not extra data.
Only after its real trace passes does the same controller admit four workers.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
ROOT = Path(RETRACE_SHARE + '/stage2-agentic')
PY = RETRACE_SHARE + '/retrace-continual/venv/bin/python'
M = ROOT / 'migration'
BENCH = 'dataset/retro-190_dataset/run_benchmark.py'
CONFIG = M / 'teacher_gpt6astra_remote.yaml'
WORKERS = 4
active = None


def save(name, value):
    p = M / name
    tmp = p.with_name(p.name + '.new')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(p)


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def record_path(rep, idx):
    return ROOT / 'main_collection' / rep / 'runs' / f'{idx:03d}_r190_{idx:03d}.json'


def validate_canary(rec, trace):
    assert rec.get('success') is True, 'canary native success missing'
    assert rec.get('iterations') == 40, 'canary did not finish the frozen 40 turns'
    assert not rec.get('error') and not rec.get('skipped'), 'canary error/skip'
    ms = trace['model_settings']['main']
    assert ms['model'] == 'gpt-6-astra' and ms['reasoning_effort'] == 'high'
    assert ms['max_tokens'] == 32768
    events = trace['events']
    calls = [x for x in events if x.get('event') == 'tool_call']
    results = [x for x in events if x.get('event') == 'tool_result']
    assert {x['tool_call_id'] for x in calls} == {x['tool_call_id'] for x in results}, 'unpaired tools'
    names = {x['name'] for x in results}
    assert {'neural_retro', 'score_route', 'save_best_route'} <= names, names
    assert rec.get('best_route'), 'no selected route artifact'
    assert trace.get('training_trace', {}).get('records'), 'missing training trace'
    return {'idx': rec['idx'], 'iterations': rec['iterations'], 'tool_calls': len(calls),
            'tool_results': len(results), 'tools': sorted(names), 'model_settings': ms,
            'native_wiring_pass': True, 'chemical_admission': 'not assessed by this gate'}


def run_phase(name, rep, ids, workers):
    global active
    if not ids:
        return
    env = dict(os.environ)
    for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy',
              'RETROPLANNER_HTTP_URL','RETROPLANNER_HTTP_URLS','PYTHONPATH']:
        env.pop(k, None)
    env.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
               AGENT_CONFIG=str(CONFIG), PYTHONUNBUFFERED='1', MEM_SKIP_GB='24',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    cmd = [PY, BENCH, '--only', ','.join(map(str, ids)), '--workers', str(workers),
           '--skip-existing', '--output-dir', str(ROOT / 'main_collection' / rep)]
    log = M / f'{name}.log'
    with log.open('xb') as out:
        active = subprocess.Popen(cmd, cwd=ROOT / 'harness', env=env, stdin=subprocess.DEVNULL,
                                  stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        save('ACTIVE_PHASE.json', {'at': stamp(), 'controller_pid': os.getpid(),
              'phase': name, 'driver_pid': active.pid, 'workers': workers, 'ids': ids,
              'argv': cmd, 'cuda_visible_devices': '', 'log': str(log)})
        code = active.wait()
        active = None
    save(f'{name}_EXIT.json', {'at': stamp(), 'exit_code': code, 'ids': ids})
    assert code == 0, f'{name} driver exit {code}'
    missing = [i for i in ids if not record_path(rep, i).is_file()]
    assert not missing, f'{name} missing terminal records: {missing}'


def terminate(signum, frame):
    # Do not call Popen.wait/poll from a signal handler: the interrupted wait
    # may already hold Popen's non-reentrant waitpid lock. Unwind it first.
    raise SystemExit(128 + signum)


def cleanup_after_stop(code):
    global active
    if active is not None and active.poll() is None:
        os.killpg(active.pid, signal.SIGTERM)
        try:
            active.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(active.pid, signal.SIGKILL)
            active.wait()
        active = None
    save('STOPPED.json', {'at': stamp(), 'exit_code': code,
                          'reason': 'operator/controller termination'})


def main():
    assert sys.platform == 'linux'
    lock = (M / 'controller.lock').open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, terminate)
    pre = json.loads((M / 'PREFLIGHT.json').read_text())
    assert pre['protocol_equal_except_transport'] and not pre['source_mismatches']
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == pre['config_sha256']
    split = json.loads((ROOT / 'split/SPLIT_STAGE2.json').read_text())
    ids = split['train_stage2']
    assert len(ids) == len(set(ids)) == 130
    done = [i for i in ids if record_path('rep1', i).exists()]
    assert set(done) == set(json.loads((M / 'LOCAL_DRAIN_RECEIPT.json').read_text())['started_ids'])
    pending = [i for i in ids if i not in done]
    canary = pending[0]
    save('CONTROLLER_START.json', {'at': stamp(), 'pid': os.getpid(), 'host': os.uname().nodename,
         'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'preserved_rep1_ids': done, 'canary_idx': canary, 'bulk_workers': WORKERS,
         'planned_total': 260, 'extra_sample_count': 0, 'gpu_execution': False})
    run_phase('canary', 'rep1', [canary], 1)
    rec = json.loads(record_path('rep1', canary).read_text())
    trace_path = Path(rec['transcript_json'])
    verdict = validate_canary(rec, json.loads(trace_path.read_text()))
    verdict.update(at=stamp(), transcript=str(trace_path),
                   transcript_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest())
    save('CANARY_VERIFIED.json', verdict)
    run_phase('rep1_remote', 'rep1', pending[1:], WORKERS)
    run_phase('rep2_remote', 'rep2', ids, WORKERS)
    rows = []
    for rep in ['rep1','rep2']:
        for i in ids:
            d = json.loads(record_path(rep, i).read_text())
            rows.append({'rep': rep, 'idx': i, 'success': d.get('success'),
                         'iterations': d.get('iterations'), 'error': d.get('error'), 'skipped': d.get('skipped')})
    assert len(rows) == 260
    save('COLLECTION_FINISHED.json', {'at': stamp(), 'terminal_records': len(rows),
         'native_success': sum(r['success'] is True for r in rows), 'rows': rows,
         'chemical_admission': 'pending semantic selection; native success is not chemistry validity'})
    save('ACTIVE_PHASE.json', {'at': stamp(), 'phase': 'finished', 'workers': 0})


if __name__ == '__main__':
    try:
        main()
    except SystemExit as exc:
        cleanup_after_stop(exc.code)
        raise
    except Exception as exc:
        if active is not None:
            cleanup_after_stop(1)
        save('CONTROLLER_FAILED.json', {'at': stamp(), 'type': type(exc).__name__, 'error': str(exc)})
        raise
