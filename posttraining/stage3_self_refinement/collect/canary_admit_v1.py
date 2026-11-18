"""Bounded, once-only two-canary dispatch. No scientific code changes."""
from pathlib import Path
import json, os, hashlib, subprocess, urllib.request, time, fcntl, shlex, sys

# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
E=Path(RETRACE_SHARE + '/stage3-supplement')
R=Path(RETRACE_SHARE + '/retrace-continual')
PY=str(R/'venv/bin/python')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def identity(pid):
 s=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
 assert s[0]!='Z'
 return {'pid':pid,'start_ticks':s[19],'sid':os.getsid(pid)}
def main():
 lock=(E/'canary_dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (E/'STOP').exists()
 assert json.loads((E/'RIBENCH_RELEASED.json').read_text())['release_permitted'] is True
 assert not (E/'CANARY_DISPATCHED.json').exists()
 subprocess.run([PY,str(E/'stage_v1/preflight.py')],check=True)
 expected=json.loads((E/'STAGE_CODE_SHA256.json').read_text())
 assert all(sha(E/'stage_v1'/name)==value for name,value in expected.items())
 frozen=json.loads((R/'rollouts/eval_round1_test40/base/launch_manifest.json').read_text())['source_sha256']
 readiness={}
 for arm,port in [('c0',8000),('c1',8002)]:
  assert not (E/'arms'/arm/'rep3_canary').exists(), 'Prior attempt exists; never re-dispatch'
  assert not (E/'canary_owners'/arm/'owner.json').exists()
  assert all(sha(E/'arms'/arm/'native-isolated'/p)==h for p,h in frozen.items())
  owner=json.loads((E/'serving'/str(port)/'owner.json').read_text())
  for role in ['supervisor','child']: assert identity(owner[role]['pid'])==owner[role]
  assert not (E/'serving'/str(port)/'process_exit.json').exists()
  models=json.loads(urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=10).read())['data']
  adapter=next(m for m in models if m['id']=='qwen3.8-27b-s2v2')
  assert adapter['root']==RETRACE_SHARE + '/stage2-train/a1v2_full_sp4_zero3/final-adapter',adapter
  name=f'retrace-s3-canary-{arm}'
  assert subprocess.run(['tmux','has-session','-t',name],capture_output=True).returncode!=0
  readiness[arm]={'port':port,'adapter':adapter,'serving_owner':owner,'frozen_source_files':len(frozen)}
 # Actual frozen scientific tool client, no student or judge generation.
 import importlib.util,csv
 h=E/'arms/c0/native-isolated/harness'; p=h/'tools/_retroplanner_http.py'
 spec=importlib.util.spec_from_file_location('native_client',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 target=json.loads((E/'STAGE_PLAN.json').read_text())['c0']['rep3_canary'][0]
 row=next(r for r in csv.DictReader((h/'dataset/retro-190_dataset/cases.csv').open()) if int(r['idx'])==target)
 result=m.RetroplannerHTTPClient('http://localhost:8010/retroplanner/api/single_step',timeout_s=45,retries=0).predict(row['smiles'],'GraphFP (USPTO-Full)',3)
 assert result and all(isinstance(r['reactants'],str) and isinstance(r['score'],float) for r in result)
 print(json.dumps({'preflight':'PASS','readiness':readiness,'native_tool_candidates':len(result),'dispatch':'--dispatch' in sys.argv}),flush=True)
 if '--dispatch' not in sys.argv:return
 for arm in ['c0','c1']:
  assert not (E/'STOP').exists()
  name=f'retrace-s3-canary-{arm}'
  cmd=['python3',str(E/'stage_v1/run_owned.py'),str(E/'canary_owners'/arm),PY,str(E/'stage_v1/launch_collect.py'),arm,'rep3_canary','1']
  log=E/f'canary_{arm}.log'
  subprocess.run(['tmux','new-session','-d','-s',name,shlex.join(cmd)+' > '+shlex.quote(str(log))+' 2>&1'],check=True)
 with (E/'CANARY_DISPATCHED.json').open('x') as f: json.dump({'at':time.time(),'arms':readiness,'targets':{a:json.loads((E/'STAGE_PLAN.json').read_text())[a]['rep3_canary'] for a in ['c0','c1']},'native_tool_candidates':len(result),'acceptance':'pending actual model/tool/score/save/terminal'},f,indent=2)
 print('CANARY_DISPATCHED',flush=True)
if __name__=='__main__':main()
