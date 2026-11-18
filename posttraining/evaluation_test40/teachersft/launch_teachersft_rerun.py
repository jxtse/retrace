"""Launch one teacher-SFT-arm infra-rerun round on the already-running teacher-SFT serving (derived from launch_base_rerun.py). Total concurrency <=3.
Usage: python3 launch_base_rerun.py TAG idx,idx,...   (TAG like infra-rerun1-001)"""
import json,sys,pathlib,subprocess,hashlib,datetime,socket

# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
B=pathlib.Path(RETRACE_SHARE + '/eval-teachersft/runs')
tag=sys.argv[1];ids=sorted(int(x) for x in sys.argv[2].split(','))
assert tag.startswith('infra-rerun') and ids
def sha(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()
def ticks(pid): return int(open(f'/proc/{pid}/stat').read().rsplit(')',1)[1].split()[19])
base={h:json.load(open(B/f'test40-base-{h}-001/launch_manifest.json')) for h in ('h0','h1')}
half={i:h for h in base for i in base[h]['indices']}
groups={h:[i for i in ids if half[i]==h] for h in ('h0','h1')}
n=[len(groups['h0']),len(groups['h1'])]
if n[0] and n[1]: w=[2,1] if n[0]>=n[1] else [1,2]
else: w=[3 if n[0] else 0,3 if n[1] else 0]
w=[min(w[k],n[k]) for k in range(2)]
for h,port in (('h0',8000),('h1',8002)):
  s=socket.socket();s.settimeout(3);s.connect(('127.0.0.1',port));s.close()
s=socket.socket();s.settimeout(3);s.connect(('127.0.0.1',8010));s.close()
for k,h in enumerate(('h0','h1')):
  if not groups[h]: continue
  m=base[h];out=B/f'test40-base-{h}-{tag}';assert not out.exists()
  assert sha(m['config']['path'])==m['config']['sha256']
  env=dict(m['env']);env.update(MPLCONFIGDIR=str(out/'mpl'),NUMBA_CACHE_DIR=str(out/'numba'),XDG_CACHE_HOME=str(out/'cache'))
  argv=list(m['argv']);argv[argv.index('--out')+1]=str(out);argv[argv.index('--output-dir')+1]=str(out)
  argv[argv.index('--only')+1]=','.join(map(str,groups[h]));argv[argv.index('--workers')+1]=str(w[k])
  plan=dict(m);plan.update(out=str(out),argv=argv,env=env,indices=groups[h],rerun_tag=tag,
            rerun_rule='harness-failed latest attempt with infra signal (503/ChemBART fail/conn refused/MolPort/empty-response wallclock); workers changed only for total concurrency<=3')
  out.mkdir(parents=True)
  with open(out/'launch_manifest.json','x') as f: json.dump(plan,f,indent=2)
  with open(out/'driver.log','xb') as log:
    pr=subprocess.Popen(argv,cwd=m['cwd'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  with open(out/'launch_receipt.json','x') as f:
    json.dump(dict(wrapper_pid=pr.pid,start_ticks=ticks(pr.pid),launch_manifest_sha256=sha(out/'launch_manifest.json'),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),f)
  print('RERUN_LAUNCHED',h,tag,groups[h],'workers',w[k],flush=True)
