"""Read-only scan of baseline test40@medium attempts. Prints JSON: per-idx latest attempt + infra signals + rerun set."""
import json,glob,os,re,sys,pathlib

# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
B=pathlib.Path(RETRACE_SHARE + '/eval-base/runs')
TAGS=['001','infra-rerun1-001','infra-rerun2-001','infra-rerun3-001']
def alive(d):
  try:
    r=json.load(open(d/'launch_receipt.json'));st=open(f"/proc/{r['wrapper_pid']}/stat").read().rsplit(')',1)[1].split()
    return int(st[19])==r['start_ticks'] and st[0]!='Z'
  except Exception: return False
batches=[];latest={};hist={}
for t in TAGS:
  for h in ('h0','h1'):
    d=B/f'test40-base-{h}-{t}'
    if not d.exists(): continue
    logs=list(d.glob('logs/*.log'))+[d/'driver.log']
    idle=int(__import__('time').time()-max(os.path.getmtime(x) for x in logs if x.exists()))
    batches.append(dict(dir=str(d),tag=t,half=h,alive=alive(d),exit=(d/'exit_receipt.json').exists(),runs=len(list(d.glob('runs/*.json'))),idle_s=idle))
    for p in sorted(d.glob('runs/*.json')):
      r=json.load(open(p)); i=r['idx']
      lg=d/'logs'/(p.stem+'.log'); txt=lg.read_text(errors='ignore') if lg.exists() else ''
      emp=0
      wl=pathlib.Path(r.get('session_dir') or '/nonexist')/'conversation/wire_log.jsonl'
      if wl.exists():
        for l in open(wl):
          x=json.loads(l);m=((x.get('response') or {}).get('choices') or [{}])[0].get('message') or {}
          if not (m.get('content') or '').strip() and not m.get('tool_calls'): emp+=1
      sig=dict(svc503=txt.count('SERVICE UNAVAILABLE'),chembart_fail=txt.count('engine=chembart failed'),
               conn_refused=len(re.findall(r'Connection refused|NewConnectionError',txt)),wallclock=('Run wallclock budget' in txt),
               http400=txt.count('HTTP 400'),molport=txt.count('Molport API key'),empty=emp)
      row=dict(idx=i,half=h,tag=t,success=r.get('success'),iterations=r.get('iterations'),wall=r.get('wall_seconds'),error=r.get('error'),path=str(p),**sig)
      hist.setdefault(i,[]).append(row);latest[i]=row
ids=sorted(json.load(open(RETRACE_SHARE + '/stage3-endtoend/authorized-training/runtime/test40-s3-h0-001/launch_manifest.json'))['full_test40'])
def contaminated(r): return r['svc503']>0 or r['chembart_fail']>0 or r['conn_refused']>0 or r['wallclock'] or r['molport']>0
rerun=[i for i,r in latest.items() if not r['success'] and contaminated(r)]
out=dict(batches=batches,n_latest=len(latest),complete40=sorted(latest)==ids,success=sum(1 for r in latest.values() if r['success']),
         rerun=sorted(rerun),attempts={i:len(v) for i,v in hist.items()},latest=[latest[i] for i in sorted(latest)],
         success_contaminated=sorted(i for i,r in latest.items() if r['success'] and contaminated(r)))
print(json.dumps(out))
