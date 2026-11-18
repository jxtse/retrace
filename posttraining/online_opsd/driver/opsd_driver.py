#!/usr/bin/env python3
"""OPSD production driver — resumable on-policy GKD (fixed self-teacher) over 8 steps,
# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_WORKSPACE = os.environ.get("RETRACE_WORKSPACE", "/path/to/workspace")
RETRACE_ARCHIVE = os.environ.get("RETRACE_ARCHIVE", "/path/to/archive")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")

then the frozen test40@medium eval of adapter 8.

State machine per step s = 1..8:
  serve -> rollout -> infra -> rows -> parity -> stop_serve -> train -> export
Phases are idempotent; progress lives in state.json (atomic) and events.jsonl (append).
Restart picks up at the first phase without ok:true.

Deviations from the brief's phase order are deliberate and documented in the release README (online_opsd): rows/parity run while the step's vLLM replicas are still up
(the /tokenize parity check needs the served tokenizer), i.e. before stop_serve.

Usage:
  opsd_driver.py run [--from-step N]
  opsd_driver.py phase <step> <phase>      # run exactly one phase (debug)
"""
import argparse
import datetime
import fcntl
import glob
import http.client
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import yaml
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
W = Path(RETRACE_SHARE + '/online-arms')
S3 = W / 's3'
LIB = S3 / 'lib'
SCR = S3 / 'scripts'
sys.path.insert(0, str(LIB))

MODEL = RETRACE_SHARE + '/models/Qwen/Qwen3.8-27B'
SERVING_PY = RETRACE_SHARE + '/stage3-endtoend/runtime/envs-attempt1/serving/bin/python'
HARNESS_PY = RETRACE_SHARE + '/stage3-endtoend/runtime/envs-attempt1/eval/bin/python'
COLLECT = Path(RETRACE_SHARE + '/stage3-collect')
PREPARED = Path(RETRACE_SHARE + '/stage3-endtoend/authorized-training/runtime')
S3RT = Path(RETRACE_SHARE + '/stage3-endtoend/runtime')
SRC_CFG = Path(RETRACE_SHARE + '/eval-stage2/arms/s2v2/rollout_h0/eval_config.yaml')
TEACHER_ADAPTER = Path(RETRACE_SHARE + '/teacher-sft/full_sp4_zero3/final-adapter')
TOOL_URL = os.environ.get('RETROPLANNER_HTTP_URL', 'http://localhost:8010/retroplanner/api/single_step')

HALVES = {'h0': dict(port=int(os.environ.get('RETRACE_H0_PORT', 8000)), cuda=os.environ.get('RETRACE_H0_CUDA', '0,1'), arm='c0'),
          'h1': dict(port=int(os.environ.get('RETRACE_H1_PORT', 8002)), cuda=os.environ.get('RETRACE_H1_CUDA', '2,3'), arm='c1')}
BASE_ALIAS = 'qwen3.8-27b-opsd-base'
LORA_ALIAS = 'qwen3.8-27b-opsd-step'
EVAL_ALIAS = 'qwen3.8-27b-opsd'
WORKERS_PER_HALF = 8
PROTOCOL = dict(temperature=0.7, reasoning_effort='medium', max_tokens=32768, max_iterations=40,
                wallclock_s=14400)
ROLLOUT_TIMEOUT = PROTOCOL['wallclock_s'] + 3600
TRAIN_TIMEOUT = 4 * 3600
N_STEPS = 8
PHASES = ['serve', 'rollout', 'infra', 'rows', 'parity', 'stop_serve', 'train', 'export']
EVAL_PHASES = ['eval_serve', 'eval', 'eval_infra', 'eval_stop']
STATE = S3 / 'driver' / 'state.json'
EVENTS = S3 / 'driver' / 'events.jsonl'
LOCKF = S3 / 'driver' / 'LOCK'
PIDF = S3 / 'driver' / 'driver.pid'
STATUS_MD = S3 / 'driver' / 'RUN_STATUS.md'

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ------------------------------------------------------------------ primitives
def now():
    return datetime.datetime.now(datetime.timezone.utc)


def iso(t=None):
    return (t or now()).isoformat(timespec='seconds')


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return dict(schema='opsd.driver.state.v1', created_at=iso(), step=1, phase=PHASES[0],
                steps={}, timings={}, abort=None, n_steps=N_STEPS, eval={}, started_at=iso())


def save_state(st):
    st['updated_at'] = iso()
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(st, indent=1, ensure_ascii=False))
    os.replace(tmp, STATE)


def event(step, phase, name, **detail):
    rec = dict(utc=iso(), step=step, phase=phase, event=name, detail=detail)
    with EVENTS.open('a') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(json.dumps(rec, ensure_ascii=False), flush=True)


def gpu_used():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    return {int(l.split(',')[0]): int(l.split(',')[1]) for l in out.strip().splitlines()}


def require_gpus_free(where):
    u = gpu_used()
    if any(v > 1000 for v in u.values()):
        raise RuntimeError(f'{where}: GPUs not free: {u}')
    return u


def port_open(port):
    s = socket.socket()
    s.settimeout(2)
    try:
        return s.connect_ex(('127.0.0.1', port)) == 0
    finally:
        s.close()


def http_json(url, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with _OPENER.open(req, timeout=timeout) as r:
        return json.load(r)


def alive(ident):
    try:
        st = Path(f'/proc/{ident["pid"]}/stat').read_text().rsplit(')', 1)[1].split()
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        return int(st[19]) == int(ident['start_ticks']) and st[0] != 'Z' and boot == ident.get('boot_id', boot)
    except Exception:
        return False


def pticks(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])


def run_cmd(argv, log, env=None, cwd=None, timeout=None):
    with open(log, 'a') as f:
        p = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             env=env, cwd=cwd, start_new_session=True)
        try:
            return p.wait(timeout=timeout), p.pid
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 15)
            return 124, p.pid


def wait_for(pred, timeout, what, poll=30, on_tick=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        if on_tick:
            on_tick()
        time.sleep(poll)
    raise TimeoutError(f'timeout {timeout}s waiting for {what}')


# ------------------------------------------------------------------ data
def targets():
    return json.loads((S3 / 'data' / 'targets.json').read_text())


def privileged():
    return {int(json.loads(l)['idx']): json.loads(l)
            for l in (S3 / 'data' / 'privileged_union.jsonl').read_text().splitlines() if l.strip()}


def batch_ids(s):
    return targets()['batches'][s - 1]


def half_split(ids):
    return {'h0': ids[0::2], 'h1': ids[1::2]}


def step_dir(*parts):
    return S3.joinpath(*parts)


# ------------------------------------------------------------------ phases
def ph_serve(st, s):
    """Serve base (s=1) or adapter s-1 on both replicas; ready probe + LoRA load probe."""
    require_gpus_free('serve')
    adapter = None
    if s > 1:
        adapter = str(step_dir('export', f'step{s - 1:02d}', 'adapter'))
        sha = (step_dir('export', f'step{s - 1:02d}', 'adapter.sha256').read_text().split()[0])
        exp = st['steps'][str(s - 1)]['export']['adapter_sha256']
        assert sha == exp, (sha, exp)
    d = step_dir('serving', f'step{s:02d}')
    for half, cfg in HALVES.items():
        hd = d / half
        hd.mkdir(parents=True, exist_ok=True)
        if (hd / 'OWNER.json').exists() and alive(json.loads((hd / 'OWNER.json').read_text())['identity']):
            continue
        argv = [sys.executable, str(SCR / 'opsd_serve.py'), '--half', half, '--port', str(cfg['port']),
                '--cuda', cfg['cuda'], '--dir', str(hd), '--base-alias', BASE_ALIAS]
        if adapter:
            argv += ['--lora-alias', LORA_ALIAS, '--adapter', adapter]
        out = run_cmd(argv, hd / 'launch.log')
        assert out[0] == 0, f'serve launch failed {half}'
        print((hd / 'launch.log').read_text().strip()[-300:], flush=True)
    served = LORA_ALIAS if adapter else BASE_ALIAS
    t0 = time.time()
    probes = {}
    for half, cfg in HALVES.items():
        url = f'http://127.0.0.1:{cfg["port"]}'
        models = None
        while time.time() - t0 < 1800:
            try:
                models = http_json(url + '/v1/models', timeout=5)
                break
            except (OSError, http.client.HTTPException):
                time.sleep(5)
        assert models, f'{half}: /v1/models never came up'
        ids = [m['id'] for m in models['data']]
        assert BASE_ALIAS in ids, (half, ids)
        if adapter:
            assert LORA_ALIAS in ids, (half, ids)
        ans = http_json(url + '/v1/chat/completions', dict(
            model=served, messages=[{'role': 'user', 'content': 'Reply with exactly: OK'}],
            max_tokens=16, temperature=0), timeout=300)
        assert ans['model'] == served and ans['usage']['completion_tokens'] > 0, (half, ans.get('model'))
        probes[half] = dict(model_ids=ids, served=served, probe_usage=ans['usage'],
                            probe_text=ans['choices'][0]['message'].get('content'),
                            ready_seconds=round(time.time() - t0, 1))
        (d / half / 'READY_PROBE.json').write_text(json.dumps(probes[half], indent=1))
    return dict(served=served, adapter=adapter, halves=probes)


def _rollout_half(st, s, tag, half, ids, workers):
    d = step_dir('rollouts', f'step{s:02d}', f'{half}-{tag}')
    d.mkdir(parents=True, exist_ok=True)
    port = HALVES[half]['port']
    harness = COLLECT / 'arms' / HALVES[half]['arm'] / 'native-isolated/harness'
    assert (harness / 'dataset/retro-190_dataset/run_benchmark.py').exists()
    served = LORA_ALIAS if s > 1 else BASE_ALIAS
    cfg = json.loads(json.dumps(__import__('yaml').safe_load(SRC_CFG.read_text())))
    cfg['llm']['base_url'] = f'http://127.0.0.1:{port}/v1'
    cfg['llm']['model'] = served
    cfg['llm']['temperature'] = PROTOCOL['temperature']
    cfg['llm']['reasoning_effort'] = PROTOCOL['reasoning_effort']
    cfg['llm']['max_tokens'] = PROTOCOL['max_tokens']
    cfg['agent']['max_iterations'] = PROTOCOL['max_iterations']
    (d / 'eval_config.yaml').write_text(__import__('yaml').safe_dump(cfg, sort_keys=False))
    runner = d / 'detached_runner.py'
    if not runner.exists():
        shutil.copy2(COLLECT.parent / 'eval-stage2/arms/s2v2/rollout_h0/detached_runner.py', runner)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', AGENT_CONFIG=str(d / 'eval_config.yaml'),
               RETROPLANNER_HTTP_URL=TOOL_URL, PYTHONUNBUFFERED='1', CHEMPRICE_DISABLE_MOLPORT='1',
               MPLCONFIGDIR=str(d / 'mpl'), NUMBA_CACHE_DIR=str(d / 'numba'), XDG_CACHE_HOME=str(d / 'cache'))
    for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy'):
        env.pop(k, None)
    cmd = [HARNESS_PY, str(runner), '--out', str(d), '--', HARNESS_PY,
           'dataset/retro-190_dataset/run_benchmark.py', '--only', ','.join(map(str, ids)),
           '--workers', str(workers), '--skip-existing', '--output-dir', str(d)]
    (d / 'launch_manifest.json').write_text(json.dumps(dict(
        step=s, tag=tag, half=half, port=port, indices=ids, n=len(ids), workers=workers,
        served_model=served, adapter=(st['steps'][str(s - 1)]['export']['adapter'] if s > 1 else None),
        protocol=PROTOCOL, config_sha256=__import__('hashlib').sha256((d / 'eval_config.yaml').read_bytes()).hexdigest(),
        created_at=iso()), indent=1))
    p = subprocess.Popen(cmd, cwd=str(harness), env=env, stdin=subprocess.DEVNULL,
                         stdout=open(d / 'driver.log', 'ab'), stderr=subprocess.STDOUT,
                         start_new_session=True, close_fds=True)
    (d / 'launch_receipt.json').write_text(json.dumps(dict(wrapper_pid=p.pid, start_ticks=pticks(p.pid),
                                                           argv=cmd, cwd=str(harness), utc=iso()), indent=1))
    event(s, 'rollout', 'launched', half=half, tag=tag, pid=p.pid, n=len(ids), workers=workers)
    return p, d


def _wait_rollout(p, d, s, tag, half, n):
    t0 = time.time()
    while p.poll() is None:
        if time.time() - t0 > ROLLOUT_TIMEOUT:
            os.killpg(p.pid, 15)
            raise TimeoutError(f'{half}-{tag}: harness exceeded {ROLLOUT_TIMEOUT}s')
        n_done = len(list(d.glob('runs/*.json')))
        event(s, 'rollout', 'heartbeat', half=half, tag=tag, n_done=n_done, n=n,
              elapsed_s=round(time.time() - t0, 1))
        time.sleep(300)
    (d / 'exit_receipt.json').write_text(json.dumps(dict(rc=p.returncode, utc=iso(),
                                                         seconds=round(time.time() - t0, 1)), indent=1))
    return p.returncode


def ph_rollout(st, s):
    ids = batch_ids(s)
    parts = half_split(ids)
    procs = {}
    done_already = all(
        len(list(step_dir('rollouts', f'step{s:02d}', f'{half}-001').glob('runs/*.json'))) >= len(parts[half])
        for half in ('h0', 'h1'))
    if done_already:
        event(s, 'rollout', 'skip_launch_all_results_present')
    for half in ('h0', 'h1'):
        if not done_already:
            procs[half] = _rollout_half(st, s, '001', half, parts[half], WORKERS_PER_HALF)
    rcs = {}
    for half in ('h0', 'h1'):
        if not done_already:
            p, d = procs[half]
            rcs[half] = _wait_rollout(p, d, s, '001', half, len(parts[half]))
    import opsd_infra
    att = {}
    for half in ('h0', 'h1'):
        for i, r in opsd_infra.collect(step_dir('rollouts', f'step{s:02d}', f'{half}-001')).items():
            r['half'] = half
            att[i] = r
    sm = opsd_infra.summarise(att)
    missing = sorted(set(ids) - set(att))
    if len(att) == 0:
        raise RuntimeError(f'step {s}: rollout produced no results')
    return dict(rc=rcs, summary=sm, missing=missing, attempts=att)


def ph_infra(st, s):
    """One infra-rerun round for failed attempts hit by harness infra faults."""
    import opsd_infra
    att = {int(k): v for k, v in st['steps'][str(s)]['rollout']['attempts'].items()}
    cand = sorted(i for i, r in att.items() if not r['success'] and r['contaminated'])
    rerun = {}
    if cand:
        parts = {}
        for i in cand:
            parts.setdefault(att[i]['half'], []).append(i)
        procs = {}
        for half, idxs in parts.items():
            procs[half] = _rollout_half(st, s, 'infra-rerun1', half, sorted(idxs),
                                        min(WORKERS_PER_HALF, len(idxs)))
        for half, (p, d) in procs.items():
            _wait_rollout(p, d, s, 'infra-rerun1', half, len(parts[half]))
            for i, r in opsd_infra.collect(d, tag='infra-rerun1').items():
                r['half'] = half
                rerun[i] = r
        att.update(rerun)
    degraded = sorted(i for i, r in att.items() if not r['success'] and r['contaminated'])
    return dict(n_rerun=len(cand), rerun_ids=cand,
                rerun_result={i: att[i]['success'] for i in sorted(rerun)},
                still_infra=degraded, degraded=degraded, attempts=att)


def ph_rows(st, s):
    import opsd_rows
    att = {int(k): v for k, v in st['steps'][str(s)]['infra']['attempts'].items()}
    priv = privileged()
    built, skipped = [], []
    for idx in sorted(att):
        r = att[idx]
        if not Path(r['path']).exists():
            skipped.append(dict(idx=idx, why='no result json'))
            continue
        try:
            e, row, ev = opsd_rows.build_one(f'step{s:02d}/{idx}', r['path'], priv[idx]['block'],
                                             meta=dict(success=r['success'], contaminated=r['contaminated'],
                                                       run_dir=r['run_dir'], tag=r['tag']))
            if not ev.get('bodies_equal', False):
                skipped.append(dict(idx=idx, why='teacher/student response bodies differ'))
                continue
            if ev.get('n_loss', 0) <= 0:
                skipped.append(dict(idx=idx, why='no loss tokens (tool-call/empty trajectory)'))
                continue
            built.append((e, row, ev))
        except Exception as ex:
            skipped.append(dict(idx=idx, why=f'{type(ex).__name__}: {ex}'))
    if not built:
        raise RuntimeError(f'step {s}: no rows built ({skipped})')
    out = step_dir('rows', f'step{s:02d}')
    rep = opsd_rows.write_step(out, built, gbs=len(batch_ids(s)))
    too_long = [e['uid'] for e, _, _ in built if max(e['n_student'], e['n_teacher']) > 131072]
    if too_long:
        raise RuntimeError(f'step {s}: rows over 131072 tokens: {too_long}')
    rep.update(skipped=skipped, too_long=[], index=str(out / 'index.json'), dataset=str(out / 'ds.jsonl'))
    ev = rep.pop('evidence', [])
    rep['per_row'] = [dict(uid=e['uid'], idx=e['idx'], n_loss=e['n_loss'], n_teacher=e['n_teacher'],
                           success=e['harness_success'], contaminated=e['meta'].get('contaminated'))
                      for e in ev]
    return rep


def ph_parity(st, s):
    """vLLM /tokenize parity on 2 random rows of this step (student view)."""
    rows = [json.loads(l) for l in (step_dir('rows', f'step{s:02d}') / 'ds.jsonl').read_text().splitlines()]
    index = json.loads((step_dir('rows', f'step{s:02d}') / 'index.json').read_text())
    rng = random.Random(42 + s)
    picks = rng.sample([r for r in rows if 'padding_copy_of' not in r], 2)
    served = LORA_ALIAS if s > 1 else BASE_ALIAS
    res = {}
    for r in picks:
        tok = http_json(f'http://127.0.0.1:{HALVES["h0"]["port"]}/tokenize',
                        dict(model=served, messages=r['messages'], tools=r['tools'],
                             add_generation_prompt=False,
                             chat_template_kwargs={'reasoning_effort': PROTOCOL['reasoning_effort']}),
                        timeout=300)
        got = tok.get('tokens') or tok.get('token_ids')
        exp = index[r['uid']]['student_ids']
        res[r['uid']] = dict(n_vllm=len(got), n_frozen=len(exp), identical=got == exp,
                             first_divergence=next((i for i in range(min(len(got), len(exp)))
                                                    if got[i] != exp[i]), None))
    (step_dir('rows', f'step{s:02d}') / 'parity.json').write_text(json.dumps(res, indent=1))
    if not all(v['identical'] for v in res.values()):
        raise RuntimeError(f'step {s}: tokenize parity FAIL {res}')
    return dict(rows=res, status='PASS')


def ph_stop_serve(st, s):
    d = step_dir('serving', f'step{s:02d}')
    rc, _ = run_cmd([sys.executable, str(SCR / 'opsd_stop_serve.py'), str(d / 'h0'), str(d / 'h1')],
                    d / 'stop.log', timeout=900)
    assert rc == 0, f'stop_serve rc={rc}'
    res = json.loads((d / 'stop.log').read_text().strip().splitlines()[-1])
    require_gpus_free('after stop_serve')
    return res


def ckpt_of(s):
    """Newest mcore checkpoint dir of training step s (contains iter_*/ + tracker)."""
    cands = sorted(glob.glob(str(step_dir('train', f'step{s:02d}') / '*' / f'checkpoint-{s}')),
                   key=lambda p: os.path.getmtime(p))
    return cands[-1] if cands else None


def ph_train(st, s):
    require_gpus_free('train')
    ds = step_dir('rows', f'step{s:02d}', 'ds.jsonl')
    idx = step_dir('rows', f'step{s:02d}', 'index.json')
    out = step_dir('train', f'step{s:02d}')
    out.mkdir(parents=True, exist_ok=True)
    resume = ckpt_of(s - 1) if s > 1 else None
    if s > 1:
        assert resume, f'step {s}: previous checkpoint missing'
    log = out / 'train.log'
    env = dict(os.environ, GBS=str(len(batch_ids(s))), TOPK='64',
               RETRACE_DUMP_DIR=str(out), RETRACE_INDEX=str(idx))
    peak = {'gib': 0.0}
    stop = {'f': False}

    def sampler():
        while not stop['f']:
            try:
                peak['gib'] = max(peak['gib'], max(gpu_used().values()) / 1024.0)
            except Exception:
                pass
            time.sleep(20)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.time()
    rc, _ = run_cmd(['bash', str(SCR / 'opsd_run_train.sh'), str(ds), str(idx), str(out), str(s), resume or ''],
                    log, env=env, timeout=TRAIN_TIMEOUT)
    stop['f'] = True
    oom = 'out of memory' in log.read_text(errors='ignore').lower()
    if rc != 0 and oom:
        event(s, 'train', 'oom_retry', rc=rc)
        t0 = time.time()
        stop['f'] = False
        th = threading.Thread(target=sampler, daemon=True)
        th.start()
        rc, _ = run_cmd(['bash', str(SCR / 'opsd_run_train.sh'), str(ds), str(idx), str(out), str(s), resume or ''],
                        log, env=env, timeout=TRAIN_TIMEOUT)
        stop['f'] = True
    if rc != 0:
        raise RuntimeError(f'step {s}: training rc={rc} (oom_retry={oom}) see {log}')
    ckpt = ckpt_of(s)
    assert ckpt, f'step {s}: no checkpoint-{s} written'
    rec = {}
    for lj in sorted(out.glob('*/logging.jsonl')):
        lines = [json.loads(l) for l in lj.read_text().splitlines() if l.strip()]
        if lines and 'loss' in lines[0]:
            rec = lines[0]
    assert rec, f'step {s}: no logging.jsonl record'
    # labels sanity: the pretokenized template reports seq_len / labels_valid per
    # microbatch; the distinct pairs must cover every row and sum to the built total.
    txt = log.read_text(errors='ignore')
    # The template logs one prepare_batch per view (student + teacher), so each row appears
    # twice with the SAME labels_valid but a different seq_len (teacher view is longer by the
    # privileged block). Compare the multiset against 2 x the built per-row n_loss.
    # Keep every logged record: distinct rows can share (seq_len, labels_valid).
    pairs = re.findall(r'S2BPROBE prepare_batch seq_len=(\d+) labels_valid=(\d+)', txt)
    vals = sorted(int(b) for _, b in pairs)
    built = st['steps'][str(s)].get('rows', {})
    per_row = built.get('per_row') or []
    # The trained dataset is padded to the batch size with copies of built rows
    # (write_step records their source uids in `padded`), so expect those too.
    n_loss = {r['uid']: int(r['n_loss']) for r in per_row}
    dataset_vals = list(n_loss.values()) + [n_loss[u] for u in built.get('padded') or []]
    exp_vals = sorted(dataset_vals * 2)   # student + teacher view, same labels
    labels = dict(n_pairs=len(pairs), n_rows=len(batch_ids(s)), n_built=len(per_row),
                  n_padded=len(built.get('padded') or []),
                  labels_valid_values=vals[:8], expected_values=exp_vals[:8],
                  sum_observed=sum(vals), sum_expected=sum(exp_vals),
                  expected_total_loss_tokens=built.get('total_loss_tokens'))
    ok = (len(exp_vals) == 2 * len(batch_ids(s)) and vals == exp_vals)
    labels['ok'] = bool(ok)
    if not ok:
        raise RuntimeError(f'step {s}: labels_valid drift {labels}')
    return dict(rc=rc, resume_from=resume, checkpoint=ckpt, train_seconds=round(time.time() - t0, 1),
                loss=rec.get('loss'), grad_norm=rec.get('grad_norm'), iteration=rec.get('iteration'),
                memory_gib_reported=rec.get('memory(GiB)'), train_speed=rec.get('train_speed(s/it)'),
                peak_gib_sampled=round(peak['gib'], 2), labels=labels, log=str(log))


def ph_export(st, s):
    ckpt = st['steps'][str(s)]['train']['checkpoint']
    out = step_dir('export', f'step{s:02d}', 'adapter')
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not any(out.iterdir()):
        out.rmdir()   # megatron export refuses an existing output_dir
    rc, _ = run_cmd(['bash', str(SCR / 'opsd_export.sh'), ckpt, str(out)],
                    step_dir('export', f'step{s:02d}', 'export.log'), timeout=3600)
    assert rc == 0, f'step {s}: export rc={rc}'
    import hashlib
    p = out / 'adapter_model.safetensors'
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    (step_dir('export', f'step{s:02d}') / 'adapter.sha256').write_text(sha + '\n')
    keys = subprocess.run([str(W / 'venv-mcore/bin/python'), str(SCR / 'opsd_keys.py'), str(p),
                           str(TEACHER_ADAPTER / 'adapter_model.safetensors')],
                          capture_output=True, text=True)
    assert keys.returncode == 0, keys.stderr[-500:]
    rep = json.loads(keys.stdout.strip().splitlines()[-1])
    (step_dir('export', f'step{s:02d}') / 'keyset.json').write_text(json.dumps(rep, indent=1))
    if not rep['equal']:
        raise RuntimeError(f'step {s}: adapter key set != teacher-SFT adapter: {rep}')
    return dict(adapter=str(out), adapter_sha256=sha, n_tensors=rep['n_ours'],
                keyset_equal=True, keyset_sha256=rep['keys_sha256'])


# ------------------------------------------------------------------ eval
def _eval_manifest(half):
    f = PREPARED / f'test40-s3-{half}-001' / 'launch_manifest.json'
    return json.loads(f.read_text())


def ph_eval_serve(st, s):
    require_gpus_free('eval serve')
    d = step_dir('eval', 'serving')
    adapter = st['steps'][str(N_STEPS)]['export']['adapter']
    sha = st['steps'][str(N_STEPS)]['export']['adapter_sha256']
    for half, cfg in HALVES.items():
        hd = d / half
        hd.mkdir(parents=True, exist_ok=True)
        if (hd / 'OWNER.json').exists() and alive(json.loads((hd / 'OWNER.json').read_text())['identity']):
            continue
        argv = [sys.executable, str(SCR / 'opsd_serve.py'), '--half', half, '--port', str(cfg['port']),
                '--cuda', cfg['cuda'], '--dir', str(hd), '--base-alias', BASE_ALIAS,
                '--lora-alias', EVAL_ALIAS, '--adapter', adapter]
        rc, _ = run_cmd(argv, hd / 'launch.log')
        assert rc == 0
    t0 = time.time()
    probes = {}
    for half, cfg in HALVES.items():
        url = f'http://127.0.0.1:{cfg["port"]}'
        models = None
        while time.time() - t0 < 1800:
            try:
                models = http_json(url + '/v1/models', timeout=5)
                break
            except (OSError, http.client.HTTPException):
                time.sleep(5)
        ids = [m['id'] for m in models['data']]
        assert BASE_ALIAS in ids and EVAL_ALIAS in ids, (half, ids)
        ans = http_json(url + '/v1/chat/completions', dict(
            model=EVAL_ALIAS, messages=[{'role': 'user', 'content': 'Reply with exactly: OK'}],
            max_tokens=16, temperature=0), timeout=300)
        assert ans['usage']['completion_tokens'] > 0
        probes[half] = dict(model_ids=ids, probe_usage=ans['usage'], ready_seconds=round(time.time() - t0, 1),
                            adapter_sha256=sha)
        (d / half / 'READY_PROBE.json').write_text(json.dumps(probes[half], indent=1))
    return dict(served=EVAL_ALIAS, adapter=adapter, adapter_sha256=sha, halves=probes)


def _eval_launch(half, tag, ids, workers, out, st):
    m = _eval_manifest(half)
    import yaml
    cfg = yaml.safe_load(Path(m['config']['path']).read_text())
    cfg['llm']['model'] = EVAL_ALIAS
    cpath = step_dir('eval', 'config', half, f'{"eval" if tag == "001" else tag}.yaml')
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_text(yaml.safe_dump(cfg, sort_keys=False))
    argv = list(m['argv'])
    argv[argv.index('--out') + 1] = str(out)
    argv[argv.index('--output-dir') + 1] = str(out)
    argv[argv.index('--only') + 1] = ','.join(map(str, ids))
    argv[argv.index('--workers') + 1] = str(workers)
    env = dict(m['env'])
    env.update(AGENT_MODEL=EVAL_ALIAS, AGENT_CONFIG=str(cpath), MPLCONFIGDIR=str(out / 'mpl'),
               NUMBA_CACHE_DIR=str(out / 'numba'), XDG_CACHE_HOME=str(out / 'cache'),
               CHEMPRICE_DISABLE_MOLPORT='1', CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1')
    f = open(out / 'driver.log', 'ab')
    p = subprocess.Popen(argv, cwd=m['cwd'], env=env, stdin=subprocess.DEVNULL, stdout=f,
                         stderr=subprocess.STDOUT, start_new_session=True)
    (out / 'launch_manifest.json').write_text(json.dumps(dict(
        schema='opsd.eval.v1', half=half, tag=tag, indices=ids, workers=workers, argv=argv,
        config=dict(path=str(cpath), sha256=__import__('hashlib').sha256(cpath.read_bytes()).hexdigest()),
        env=env, cwd=m['cwd'], alias=EVAL_ALIAS, source_manifest=str(PREPARED / f'test40-s3-{half}-001' / 'launch_manifest.json'),
        diff_vs_s3_test40=['llm.model / AGENT_MODEL = qwen3.8-27b-opsd (LoRA: opsd adapter 8)',
                           'serve --lora-modules -> qwen3.8-27b-opsd', 'CHEMPRICE_DISABLE_MOLPORT=1'],
        created_at=iso()), indent=1))
    (out / 'launch_receipt.json').write_text(json.dumps(dict(wrapper_pid=p.pid, start_ticks=pticks(p.pid), utc=iso()), indent=1))
    return p, out


def ph_eval(st, s):
    parts = {}
    for half in ('h0', 'h1'):
        m = _eval_manifest(half)
        parts[half] = sorted(m['indices'])
        assert len(parts[half]) == 20
    procs = {}
    for half in ('h0', 'h1'):
        out = step_dir('eval', f'run-{half}-001')
        out.mkdir(parents=True, exist_ok=True)
        procs[half] = _eval_launch(half, '001', parts[half], 4, out, st)
    rcs = {}
    for half, (p, out) in procs.items():
        t0 = time.time()
        while p.poll() is None:
            if time.time() - t0 > ROLLOUT_TIMEOUT:
                os.killpg(p.pid, 15)
                raise TimeoutError(f'eval {half}: exceeded {ROLLOUT_TIMEOUT}s')
            time.sleep(300)
        rcs[half] = p.returncode
        (out / 'exit_receipt.json').write_text(json.dumps(dict(rc=p.returncode, utc=iso()), indent=1))
    import opsd_infra
    att = {}
    for half in ('h0', 'h1'):
        att.update(opsd_infra.collect(step_dir('eval', f'run-{half}-001'), tag='001'))
    return dict(rc=rcs, summary=opsd_infra.summarise(att), attempts=att,
                full_test40=sorted(set(parts['h0']) | set(parts['h1'])))


def ph_eval_infra(st, s):
    import opsd_infra
    att = {int(k): v for k, v in st['eval']['eval']['attempts'].items()}
    cand = sorted(i for i, r in att.items() if not r['success'] and r['contaminated'])
    if cand:
        m0, m1 = _eval_manifest('h0'), _eval_manifest('h1')
        half_of = {i: h for h, m in (('h0', m0), ('h1', m1)) for i in m['indices']}
        parts = {}
        for i in cand:
            parts.setdefault(half_of[i], []).append(i)
        n = [len(parts.get('h0', [])), len(parts.get('h1', []))]
        w = [2, 1] if n[0] >= n[1] else [1, 2]
        if not n[0]:
            w = [0, 3]
        if not n[1]:
            w = [3, 0]
        w = [min(w[k], n[k]) for k in range(2)]
        procs = {}
        for k, half in enumerate(('h0', 'h1')):
            if not parts.get(half):
                continue
            out = step_dir('eval', f'run-{half}-infra-rerun1')
            out.mkdir(parents=True, exist_ok=True)
            procs[half] = _eval_launch(half, 'infra-rerun1', sorted(parts[half]), max(1, w[k]), out, st)
        rerun = {}
        for half, (p, out) in procs.items():
            p.wait(timeout=ROLLOUT_TIMEOUT)
            (out / 'exit_receipt.json').write_text(json.dumps(dict(rc=p.returncode, utc=iso()), indent=1))
            rerun.update(opsd_infra.collect(out, tag='infra-rerun1'))
        att.update(rerun)
    return dict(n_rerun=len(cand), rerun_ids=cand, still_infra=sorted(
        i for i, r in att.items() if not r['success'] and r['contaminated']), attempts=att)


def ph_eval_stop(st, s):
    d = step_dir('eval', 'serving')
    rc, _ = run_cmd([sys.executable, str(SCR / 'opsd_stop_serve.py'), str(d / 'h0'), str(d / 'h1')],
                    d / 'stop.log', timeout=900)
    assert rc == 0
    require_gpus_free('after eval stop')
    snap = dict(
        schema='opsd.eval.snapshot.v1', alias=EVAL_ALIAS,
        adapter=st['steps'][str(N_STEPS)]['export']['adapter'],
        adapter_sha256=st['steps'][str(N_STEPS)]['export']['adapter_sha256'],
        runs={h: str(step_dir('eval', f'run-{h}-001')) for h in ('h0', 'h1')},
        n_results={h: len(list(step_dir('eval', f'run-{h}-001').glob('runs/*.json'))) for h in ('h0', 'h1')},
        summary=st['eval']['eval']['summary'], infra=st['eval']['eval_infra'].get('n_rerun'),
        pull_command=('rsync -a ' + RETRACE_HOST + ':' + str(S3) + '/eval/ ' + RETRACE_ARCHIVE
                      + '/Archive/online-opsd/eval/'),
        created_at=iso())
    (step_dir('eval') / 'SNAPSHOT_READY.json').write_text(json.dumps(snap, indent=1))
    return dict(snapshot=str(step_dir('eval') / 'SNAPSHOT_READY.json'), **snap)


PHASE_FN = {'serve': ph_serve, 'rollout': ph_rollout, 'infra': ph_infra, 'rows': ph_rows,
            'parity': ph_parity, 'stop_serve': ph_stop_serve, 'train': ph_train, 'export': ph_export,
            'eval_serve': ph_eval_serve, 'eval': ph_eval, 'eval_infra': ph_eval_infra,
            'eval_stop': ph_eval_stop}


def run_phase(st, s, phase):
    rec = st['eval'] if phase.startswith('eval') else st['steps'].setdefault(str(s), {})
    t0 = time.time()
    event(s, phase, 'begin')
    try:
        res = PHASE_FN[phase](st, s)
    except Exception as ex:
        rec[phase] = dict(ok=False, error=f'{type(ex).__name__}: {ex}', seconds=round(time.time() - t0, 1),
                          traceback=traceback.format_exc()[-2000:])
        st['abort'] = dict(step=s, phase=phase, error=str(ex), utc=iso())
        save_state(st)
        event(s, phase, 'failed', error=str(ex))
        raise
    rec[phase] = dict(ok=True, seconds=round(time.time() - t0, 1), finished_at=iso(), **res)
    tm = st['timings'].setdefault(phase, [])
    tm.append(round(time.time() - t0, 1))
    st['abort'] = None
    save_state(st)
    event(s, phase, 'ok', seconds=round(time.time() - t0, 1))
    return res


# ------------------------------------------------------------------ status docs
def eta_seconds(st):
    def mean(phase, fallback):
        v = st['timings'].get(phase)
        return sum(v) / len(v) if v else fallback
    rem, cur = 0.0, st.get('step', 1)
    done_phases = set()
    for s in range(1, N_STEPS + 1):
        d = st['steps'].get(str(s), {})
        for p in PHASES:
            if d.get(p, {}).get('ok'):
                done_phases.add((s, p))
            else:
                rem += mean(p, dict(serve=420, rollout=12600, infra=1800, rows=240, parity=120,
                                    stop_serve=300, train=1500, export=240)[p])
    ev = st.get('eval', {})
    for p in EVAL_PHASES:
        if not ev.get(p, {}).get('ok'):
            rem += mean(p, dict(eval_serve=420, eval=13000, eval_infra=2400, eval_stop=300)[p])
    return rem


def write_status(st, alive_pid=None):
    lines = []
    ab = st.get('abort')
    lines.append('# OPSD 生产运行状态（the GPU host）\n')
    lines.append(f'- 更新时间：{iso()}')
    lines.append(f'- 驱动 PID：{alive_pid if alive_pid else "未运行"}')
    cur = st.get('step')
    if isinstance(cur, int) and st.get('phase') in ('rollout', 'infra'):
        d = step_dir('rollouts', f'step{cur:02d}')
        n = len(list(d.glob('*-*/runs/*.json'))) if d.exists() else 0
        logs = list(d.glob('*-*/driver.log')) if d.exists() else []
        idle = int(time.time() - max((p.stat().st_mtime for p in logs), default=time.time()))
        lines.append(f'- 实时 rollout：{n}/16 题已有结果，日志静默 {idle}s')
    if ab:
        lines.append(f'- **已中止**：step {ab["step"]} / {ab["phase"]}：`{ab["error"][:300]}`')
    else:
        lines.append(f'- 当前：step {st.get("step")} / {st.get("phase")}')
    rem = eta_seconds(st)
    lines.append(f'- 预计剩余：{rem / 3600:.1f} h（ETA {iso(now() + datetime.timedelta(seconds=rem))}）\n')
    lines.append('| step | rollout 成功 | infra 重跑 | 行数/最大token | parity | loss | grad_norm | 峰值GiB | 训练耗时 | 各阶段耗时(s) |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for s in range(1, N_STEPS + 1):
        d = st['steps'].get(str(s), {})
        ro = d.get('rollout', {}).get('summary', {})
        inf = d.get('infra', {})
        rows = d.get('rows', {})
        tr = d.get('train', {})
        ph = {p: d.get(p, {}).get('seconds') for p in PHASES if d.get(p, {}).get('ok')}
        lines.append('| {} | {}/{} | {} | {} / {} | {} | {} | {} | {} | {} | {} |'.format(
            s,
            ro.get('success', '—'), ro.get('n', '—'),
            inf.get('n_rerun', '—') if inf else '—',
            rows.get('n_rows', '—'), rows.get('max_n_teacher', '—'),
            d.get('parity', {}).get('status', '—'),
            f'{tr.get("loss"):.6f}' if tr.get('loss') is not None else '—',
            f'{tr.get("grad_norm"):.6f}' if tr.get('grad_norm') is not None else '—',
            (tr.get('peak_gib_sampled') or tr.get('memory_gib_reported') or '—'),
            f'{tr.get("train_seconds") or tr.get("seconds")}s' if (tr.get('train_seconds') or tr.get('seconds')) else '—',
            json.dumps(ph, ensure_ascii=False)))
        if inf and inf.get('still_infra'):
            lines.append(f'\n  - step {s} infra 重跑后仍受基础设施影响的题：{inf["still_infra"]}')
    ev = st.get('eval', {})
    if ev:
        lines.append('\n## test40 评测\n')
        if ev.get('eval_serve', {}).get('ok'):
            lines.append(f'- 服务：{EVAL_ALIAS}（adapter 8 sha {str(ev["eval_serve"].get("adapter_sha256"))[:12]}…）')
        if ev.get('eval', {}).get('ok'):
            sm = ev['eval']['summary']
            lines.append(f'- 结果：{sm["success"]}/{sm["n"]} 成功；infra 重跑 {ev.get("eval_infra", {}).get("n_rerun", "—")}')
        if ev.get('eval_stop', {}).get('ok'):
            lines.append(f'- 快照：`{ev["eval_stop"]["snapshot"]}`（Mac 侧执行 pull_command 落盘）')
    STATUS_MD.write_text('\n'.join(lines) + '\n')


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'phase', 'status'])
    ap.add_argument('args', nargs='*')
    ap.add_argument('--from-step', type=int, default=None)
    a = ap.parse_args()
    S3.joinpath('driver').mkdir(parents=True, exist_ok=True)
    st = load_state()
    if a.mode == 'status':
        write_status(st)
        print(json.dumps(dict(step=st.get('step'), phase=st.get('phase'), abort=st.get('abort'),
                              eta_h=round(eta_seconds(st) / 3600, 2)), ensure_ascii=False))
        return
    lk = LOCKF.open('w')
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('driver already running (lock held)', flush=True)
        return 0
    PIDF.write_text(str(os.getpid()))
    if a.mode == 'phase':
        s, phase = int(a.args[0]), a.args[1]
        run_phase(st, s, phase)
        write_status(st, os.getpid())
        return
    # full run
    step_start = a.from_step or st.get('step', 1)
    write_status(st, os.getpid())
    for s in range(step_start, N_STEPS + 1):
        st['step'] = s
        for p in PHASES:
            if st['steps'].get(str(s), {}).get(p, {}).get('ok'):
                continue
            st['phase'] = p
            save_state(st)
            run_phase(st, s, p)
            write_status(st, os.getpid())
    st['step'] = 'eval'
    for p in EVAL_PHASES:
        if st['eval'].get(p, {}).get('ok'):
            continue
        st['phase'] = p
        save_state(st)
        run_phase(st, 8, p)
        write_status(st, os.getpid())
    st['phase'] = 'done'
    st['finished_at'] = iso()
    save_state(st)
    event(8, 'done', 'complete')
    write_status(st, os.getpid())


if __name__ == '__main__':
    try:
        sys.exit(main() or 0)
    except Exception:
        traceback.print_exc()
        try:
            st = load_state()
            write_status(st)
        except Exception:
            pass
        sys.exit(1)
