"""OPSD test40 accepted-final snapshot (latest attempt per idx). Derived from
/path/to/placeholder/retrace_share/retrace-test40-teachersft-launch/snapshot_teachersft.py — same schema
shape, our run layout, our alias/adapter.

Runs on the GPU host after the driver's eval phases; writes s3/eval/test40-final-snapshot-v1/
{snapshots,evidence,accepted40_opsd.json}. The Mac side then pulls the whole eval dir
(see SNAPSHOT_READY.json -> pull_command).

usage: python3 s3/scripts/opsd_snapshot.py [--force]
"""
import collections
import datetime
import hashlib
import json
import pathlib
import shutil
import sys


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
S3 = pathlib.Path(RETRACE_SHARE + '/online-arms/s3')
EVAL = S3 / 'eval'
SNAP = EVAL / 'test40-final-snapshot-v1'
ALIAS = 'qwen3.8-27b-opsd'
ADAPTER = S3 / 'export' / 'step08' / 'adapter'
ORDER = ['001', 'infra-rerun1']


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def need(c, m):
    if not c:
        raise SystemExit('FAIL ' + m)


def main():
    force = '--force' in sys.argv
    adapter_sha = (S3 / 'export' / 'step08' / 'adapter.sha256').read_text().split()[0]
    need(sha(ADAPTER / 'adapter_model.safetensors') == adapter_sha, 'adapter sha drift')
    latest, hist = {}, collections.defaultdict(list)
    for h in ('h0', 'h1'):
        for t in ORDER:
            b = EVAL / f'run-{h}-{t}'
            if not b.exists():
                continue
            exit_rec = b / 'exit_receipt.json'
            need(exit_rec.exists(), 'missing exit receipt ' + str(b))
            rc = json.loads(exit_rec.read_text()).get('rc', 0)
            need(rc == 0, 'exit rc!=0 ' + str(b))
            for p in sorted((b / 'runs').glob('*.json')):
                d = json.loads(p.read_text())
                hist[d['idx']].append(f'{h}-{t}')
                latest[d['idx']] = (h, t, b, p, d)
    m0 = json.loads((EVAL / 'run-h0-001' / 'launch_manifest.json').read_text())
    m1 = json.loads((EVAL / 'run-h1-001' / 'launch_manifest.json').read_text())
    ids = sorted(set(m0['indices']) | set(m1['indices']))
    need(len(ids) == 40 and sorted(latest) == ids, f'coverage {len(latest)}/{len(ids)}')
    if SNAP.exists() and force:
        shutil.rmtree(SNAP)
    need(not SNAP.exists(), 'snapshot exists')
    SNAP.mkdir(parents=True)

    def cp(src, dst):
        dst = SNAP / dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        need(sha(src) == sha(dst), 'copy sha ' + str(src))
        return dict(path=str(dst.relative_to(SNAP)), sha256=sha(dst), origin=str(src))

    rows = []
    for i in ids:
        h, t, b, p, d = latest[i]
        need(d.get('finished_at') and not d.get('error') and type(d.get('success')) is bool, f'result {i}')
        m = json.loads((b / 'launch_manifest.json').read_text())
        need(m.get('alias') == ALIAS, f'model alias {i}')
        need(m.get('tag') in ORDER, f'tag {i}')
        tr = pathlib.Path(d['transcript_json'])
        need(tr.is_file(), f'transcript {i}')
        rows.append(dict(idx=i, attempt_id=f'{h}-{t}', success=d['success'],
                         result=cp(p, f'snapshots/{i:03d}/result.json'),
                         transcript=cp(tr, f'snapshots/{i:03d}/transcript.json'),
                         launch=cp(b / 'launch_manifest.json', f'evidence/{h}-{t}-launch.json'),
                         attempts=hist[i]))
    man = dict(schema='opsd.test40.accepted-final.v1', condition='opsd', adapter_sha256=adapter_sha,
               model='Qwen3.8-27B + on-policy OPSD (GKD fixed self-teacher) LoRA, 8 steps x 16 targets',
               policy=('per idx latest infra rerun supersedes earlier; reruns only for harness-failed '
                       'infra-contaminated cases (503/ChemBART/conn/refused/wallclock/MolPort); '
                       'CHEMPRICE_DISABLE_MOLPORT=1; test40@medium frozen protocol'),
               accepted=rows, native_success=sum(r['success'] for r in rows),
               created_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (SNAP / 'accepted40_opsd.json').open('x') as f:
        json.dump(man, f, indent=1)
    print(json.dumps(dict(snapshot=str(SNAP), sha=sha(SNAP / 'accepted40_opsd.json'),
                          success=man['native_success'],
                          by=collections.Counter(r['attempt_id'].split('-', 1)[1] for r in rows))))


if __name__ == '__main__':
    main()
