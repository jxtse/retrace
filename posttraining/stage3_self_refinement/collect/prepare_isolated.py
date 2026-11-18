"""Per-arm isolated copy of the frozen native harness (same rules as eval-sft/prepare_isolated_v2.py).

Unchanged native code/data, fresh writable state; frozen sources verified against the historical base test40 manifest.
Usage: python prepare_isolated.py <arm>
"""
from pathlib import Path
import datetime, hashlib, json, subprocess, sys


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
SOURCE = Path(RETRACE_SHARE + '/retrace-continual')
OUT = Path(RETRACE_SHARE + '/stage3-collect')
EXCLUDES = ['/runs/', '/.sessions/', '/.cache/', '/tools/.cache/', '/.fix_backups/',
            '/.venv*/', '/.git/', '/.agents', '/.codex', '/.claude/', '/.pytest_cache/',
            '__pycache__/', '*.pyc',
            '/dataset/Enamine_Building_Blocks_Stock_573928cmpd_20260210.sdf',
            '/dataset/USPTO_full', '/dataset/benchmark/SAIS/score_smiles_candidates.csv']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    arm = sys.argv[1]
    assert arm in ('c0', 'c1')
    dest = OUT / 'arms' / arm / 'native-isolated'
    dest.mkdir(parents=True, exist_ok=False)
    manifest_path = SOURCE / 'rollouts/eval_round1_test40/base/launch_manifest.json'
    frozen = json.loads(manifest_path.read_text())['source_sha256']
    drift = [r for r, e in frozen.items() if sha(SOURCE / r) != e]
    if drift:
        raise RuntimeError('Original frozen source drift: %s' % drift)
    n = 0
    for name in ['harness', 'chemoenzymatic-retrosynthesis-verifier']:
        src, dst = SOURCE / name, dest / name
        dst.mkdir()
        subprocess.run(['rsync', '-aL'] + ['--exclude=' + x for x in EXCLUDES] + [str(src) + '/', str(dst) + '/'],
                       check=True, timeout=1800)
        for path in dst.rglob('*'):
            if path.is_symlink():
                raise RuntimeError('Unexpected surviving symlink: ' + str(path))
            if path.is_file():
                n += 1
    bad = [r for r, e in frozen.items() if sha(dest / r) != e]
    if bad:
        raise RuntimeError('Copied frozen source drift: %s' % bad)
    for rel in ['harness/runs', 'harness/.sessions', 'harness/.cache', 'harness/tools/.cache']:
        if (dest / rel).exists():
            raise RuntimeError('Old mutable state copied: ' + rel)
    rec = {'arm': arm, 'verified': True, 'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'source_root': str(SOURCE), 'isolated_root': str(dest), 'frozen_source_files': len(frozen),
           'frozen_manifest': str(manifest_path), 'frozen_manifest_sha256': sha(manifest_path),
           'copied_files': n, 'exclusions': EXCLUDES, 'scientific_source_edits': 0}
    (OUT / 'arms' / arm / 'ISOLATION_VERIFIED.json').write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec))


if __name__ == '__main__':
    main()
