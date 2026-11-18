"""Baseline accepted-final snapshot (latest attempt per idx). Usage: python3 snapshot_base.py"""
import json,hashlib,pathlib,shutil,datetime,collections

# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
B=pathlib.Path(RETRACE_SHARE + '/eval-base')
SNAP=B/'test40-final-snapshot-v1'
ORDER=['001','infra-rerun1-001','infra-rerun2-001','infra-rerun3-001']
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def need(c,m):
  if not c: raise SystemExit('FAIL '+m)
latest={};hist=collections.defaultdict(list)
for h in ('h0','h1'):
  for t in ORDER:
    b=B/'runs'/f'test40-base-{h}-{t}'
    if not b.exists(): continue
    need((b/'exit_receipt.json').exists() and json.load(open(b/'exit_receipt.json'))['process_returncode']==0,'exit '+str(b))
    for p in sorted((b/'runs').glob('*.json')):
      d=json.load(open(p));hist[d['idx']].append(f'{h}-{t}');latest[d['idx']]=(h,t,b,p,d)
ids=sorted(json.load(open(B/'runs/test40-base-h0-001/launch_manifest.json'))['full_test40'])
need(sorted(latest)==ids,'coverage')
need(not SNAP.exists(),'snapshot exists');SNAP.mkdir()
def cp(src,dst):
  dst=SNAP/dst;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);need(sha(src)==sha(dst),'copy')
  return dict(path=str(dst.relative_to(SNAP)),sha256=sha(dst),origin=str(src))
rows=[]
for i in ids:
  h,t,b,p,d=latest[i]
  need(d.get('finished_at') and not d.get('error') and type(d.get('success')) is bool,'result %d'%i)
  m=json.load(open(b/'launch_manifest.json'));need(m['env']['AGENT_MODEL']=='qwen3.8-27b-student','model %d'%i)
  tr=pathlib.Path(d['transcript_json']);need(tr.is_file(),'transcript %d'%i)
  rows.append(dict(idx=i,attempt_id=f'{h}-{t}',success=d['success'],result=cp(p,f'snapshots/{i:03d}/result.json'),
    transcript=cp(tr,f'snapshots/{i:03d}/transcript.json'),launch=cp(b/'launch_manifest.json',f'evidence/{h}-{t}-launch.json'),attempts=hist[i]))
man=dict(schema='base.test40.accepted-final.v1',condition='base',model='Qwen3.8-27B base (qwen3.8-27b-student, no LoRA)',
  policy='per idx latest infra rerun supersedes earlier; reruns only for harness-failed infra-contaminated cases; MolPort disabled',
  accepted=rows,native_success=sum(r['success'] for r in rows),created_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
with open(SNAP/'accepted40_base.json','x') as f: json.dump(man,f,indent=1)
print(json.dumps(dict(snapshot=str(SNAP),sha=sha(SNAP/'accepted40_base.json'),success=man['native_success'],by=collections.Counter(r['attempt_id'].split('-',1)[1] for r in rows))))
