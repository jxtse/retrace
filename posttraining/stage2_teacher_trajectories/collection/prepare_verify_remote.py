"""Verify the isolated remote copy and build a transport-only configuration.
Run only on the existing authorized the GPU host CPU instance, never on the Mac.
"""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import yaml


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
ROOT = Path(RETRACE_SHARE + '/stage2-agentic')
assert ROOT.is_dir() and sys.platform == 'linux'
M = ROOT / 'migration'
manifest = json.loads((M / 'SOURCE_MANIFEST.json').read_text())

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

mismatches = []
for row in manifest['files']:
    p = ROOT / row['path']
    if not p.is_file() or p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
        mismatches.append(row['path'])
if mismatches:
    print(json.dumps({'source_mismatches': mismatches}, indent=2), flush=True)
    raise SystemExit(2)

original = ROOT / 'harness/configs/teacher_gpt6astra.yaml'
cfg = yaml.safe_load(original.read_text())
# Same upstream chemical service via a dedicated loopback-only SSH reverse forward.
chem = 'http://localhost:8011/retroplanner/api/'
cfg['neural_retro'].update({
    'retroplanner_http_url': chem + 'single_step',
    'retroplanner_http_urls': [chem + 'single_step'],
    'chembart_url': chem + 'single_step',
    'pistachio_ringbreaker_url': chem + 'single_step',
})
cfg['verifier']['feasibility']['rater']['http_url'] = [chem + 'reaction_rater']
assert cfg['llm']['model'] == 'gpt-6-astra'
assert cfg['llm']['base_url'] == 'http://localhost:8080/v1'
assert cfg['agent']['max_iterations'] == 40
assert not cfg['verifier']['feasibility']['judge']['enabled']
assert not cfg['neural_retro']['precursor_sanity_filter']['enabled']
out_config = M / 'teacher_gpt6astra_remote.yaml'
with out_config.open('x') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

sys.path.insert(0, str(ROOT / 'harness'))
from core.config import load_config
local_resolved = load_config(str(original))
remote_resolved = load_config(str(out_config))
# Exactly the same resolved protocol after undoing transport URL changes.
for k in ('retroplanner_http_url', 'retroplanner_http_urls', 'chembart_url', 'pistachio_ringbreaker_url'):
    remote_resolved['neural_retro'][k] = local_resolved['neural_retro'][k]
remote_resolved['verifier']['feasibility']['rater']['http_url'] = local_resolved['verifier']['feasibility']['rater']['http_url']
assert remote_resolved == local_resolved, 'Non-transport config drift'

# Validate all carried-over main records and full transcripts by exact hashes.
drain = json.loads((M / 'LOCAL_DRAIN_RECEIPT.json').read_text())
relocations = []
for row in drain['completed_records']:
    old = Path(row['path'])
    dst = ROOT / old.relative_to(manifest['local_root'])
    assert sha(dst) == row['sha256']
    rec = json.loads(dst.read_text())
    src_transcript = Path(rec['transcript_json'])
    dst_transcript = ROOT / src_transcript.relative_to(manifest['local_root'])
    events = json.loads(dst_transcript.read_text())
    relocations.append({'idx': rec['idx'], 'record': str(dst),
                        'original_transcript_json': str(src_transcript),
                        'relocated_transcript_json': str(dst_transcript),
                        'transcript_sha256': sha(dst_transcript),
                        'transcript_root_type': type(events).__name__})
with (M / 'TRANSCRIPT_RELOCATION.json').open('x') as f:
    json.dump(relocations, f, indent=2)

# Exercise the native Linux memory-watchdog exit path with a tiny child.
watchdog_json = M / 'WATCHDOG_PROBE.json'
probe = '''import importlib.util,time,pathlib
p=pathlib.Path(%r)
spec=importlib.util.spec_from_file_location('bench_watchdog_probe',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m._rss_gb() > 0
m._start_mem_watchdog(pathlib.Path(%r),{'idx':'authored_cpu_watchdog_probe'},0.001)
time.sleep(12)
raise SystemExit(11)
''' % (str(ROOT / 'harness/dataset/retro-190_dataset/run_benchmark.py'), str(watchdog_json))
p = subprocess.run([sys.executable, '-I', '-c', probe], capture_output=True, text=True, timeout=20)
assert p.returncode == 99, (p.returncode, p.stdout, p.stderr)
assert json.loads(watchdog_json.read_text())['skipped'] == 'mem_limit'

op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
models = json.load(op.open('http://localhost:8080/v1/models', timeout=15))
assert 'gpt-6-astra' in [x['id'] for x in models['data']]
versions = {k: metadata.version(k) for k in ['rdkit','numpy','requests','pydantic','PyYAML','molbloom']}
report = {'source_files_verified': len(manifest['files']), 'source_mismatches': [],
          'local_completed_preserved': len(relocations), 'protocol_equal_except_transport': True,
          'versions': versions, 'config': str(out_config), 'config_sha256': sha(out_config),
          'linux_native_watchdog_probe_exit': p.returncode,
          'watchdog_limit_GiB': 24, 'cgroup_containment': 'unverified: cgroup filesystem not exposed',
          'gpu_execution': False, 'native_full_canary': 'pending'}
with (M / 'PREFLIGHT.json').open('x') as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2), flush=True)
