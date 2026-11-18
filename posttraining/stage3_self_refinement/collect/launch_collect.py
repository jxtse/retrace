"""S3 data collection.
# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_WORKSPACE = os.environ.get("RETRACE_WORKSPACE", "/path/to/workspace")
RETRACE_ARCHIVE = os.environ.get("RETRACE_ARCHIVE", "/path/to/archive")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")

Student = S2 v2 adapter (S1v2 -> A1-medium, ms-swift, sha 370edde7...), serving argv identical to the S2 v2 test40 run.
Per-trajectory protocol = the frozen test40 medium protocol (same eval_config as S2 v2 test40, only base_url differs):
temperature 0.7, max_tokens 32768, 40 iterations, 14400 s/case, reasoning_effort medium, tools via :8010.
Targets: the 130 frozen train_stage2 targets (SPLIT_STAGE2.json); asserted disjoint from dev and test40.
Two rollouts per target: rep1 for all 130 first, then rep2. Replica c0 (:8000) gets sorted targets [0::2], c1 (:8002) [1::2].
Usage: python launch_collect.py <c0|c1> <rep1|rep2> <workers>"""
import datetime, hashlib, json, os, subprocess, sys, urllib.request
from pathlib import Path
import yaml


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
E = Path(RETRACE_SHARE + '/stage3-collect')
R = Path(RETRACE_SHARE + '/retrace-continual')
PY = R / 'venv/bin/python'
SPLIT = Path(RETRACE_SHARE + '/stage2-agentic/split/SPLIT_STAGE2.json')
SRC_CFG = Path(RETRACE_SHARE + '/eval-stage2/arms/s2v2/rollout_h0/eval_config.yaml')
RUNNER = Path(RETRACE_SHARE + '/eval-stage2/arms/s2v2/rollout_h0/detached_runner.py')
ADAPTER = Path(RETRACE_SHARE + '/stage2-train/a1v2_full_sp4_zero3/final-adapter')
ADAPTER_SHA = '370edde7b44e0c4ef3cedcc35492306e1a95773a67fafd65f6ef4e6c53405e02'
PORT = {'c0': 8000, 'c1': 8002}
HALF = {'c0': 0, 'c1': 1}
ALIAS = 'qwen3.8-27b-s2v2'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    arm, rep, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    assert arm in PORT and rep in ('rep1', 'rep2')
    port = PORT[arm]
    harness = E / 'arms' / arm / 'native-isolated/harness'
    assert json.loads((E / 'arms' / arm / 'ISOLATION_VERIFIED.json').read_text())['verified']
    split = json.loads(SPLIT.read_text())
    train = sorted(split['train_stage2'])
    held = set(split['dev']) | set(split['test40_eval_only'])
    test40 = json.loads((R / 'data/reference-v1/provenance/manifest.json').read_text())['split']['test']
    assert len(train) == 130 and not set(train) & held and not set(train) & set(test40)
    ids = train[HALF[arm]::2]
    assert sha(ADAPTER / 'adapter_model.safetensors') == ADAPTER_SHA
    models = {m['id'] for m in json.loads(urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=10).read())['data']}
    assert ALIAS in models, models
    run = E / 'arms' / arm / rep
    run.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(SRC_CFG.read_text())
    assert cfg['llm']['model'] == ALIAS and cfg['llm']['reasoning_effort'] == 'medium'
    cfg['llm']['base_url'] = f'http://127.0.0.1:{port}/v1'
    (run / 'eval_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    a, b = yaml.safe_load(SRC_CFG.read_text()), yaml.safe_load((run / 'eval_config.yaml').read_text())
    a['llm'].pop('base_url'); b['llm'].pop('base_url')
    assert a == b, 'config drift vs S2 v2 test40'
    (run / 'detached_runner.py').write_bytes(RUNNER.read_bytes())
    rec = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), arm=arm, rep=rep, port=port,
               served_model=ALIAS, adapter=str(ADAPTER), adapter_sha256=ADAPTER_SHA, indices=ids, n=len(ids),
               split=str(SPLIT), split_sha256=sha(SPLIT), purpose='S3 student self-sampled trajectories, train_stage2 only',
               approved_by='owner-authorized',
               protocol={'max_tokens': 32768, 'reasoning_effort': 'medium', 'temperature': 0.7, 'max_iterations': 40,
                         'workers': workers, 'wallclock_s': 14400}, config_sha256=sha(run / 'eval_config.yaml'),
               source_config=str(SRC_CFG))
    (run / 'launch_manifest.json').write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', AGENT_CONFIG=str(run / 'eval_config.yaml'),
               RETROPLANNER_HTTP_URL='http://localhost:8010/retroplanner/api/single_step', PYTHONUNBUFFERED='1')
    cmd = [str(PY), str(run / 'detached_runner.py'), '--out', str(run), '--', str(PY),
           'dataset/retro-190_dataset/run_benchmark.py', '--only', ','.join(map(str, ids)), '--workers', str(workers),
           '--skip-existing', '--output-dir', str(run)]
    p = subprocess.Popen(cmd, cwd=harness, env=env, stdin=subprocess.DEVNULL, stdout=open(run / 'driver.log', 'ab'),
                         stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    (run / 'launch_receipt.json').write_text(json.dumps({'wrapper_pid': p.pid, 'argv': cmd, 'cwd': str(harness),
                                                         'created_at': rec['created_at']}, indent=2))
    print('LAUNCHED', arm, rep, port, len(ids), p.pid, flush=True)
    sys.exit(p.wait())


if __name__ == '__main__':
    main()
