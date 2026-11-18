import json,os,signal,time,pathlib,subprocess

# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
S=pathlib.Path(RETRACE_SHARE + '/eval-base/serving')
res={}
for h in ('h0','h1'):
  o=json.load(open(S/h/'OWNER.json'))['identity']
  try: st=int(open(f"/proc/{o['pid']}/stat").read().rsplit(')',1)[1].split()[19])
  except OSError: res[h]='gone';continue
  if st!=o['start_ticks']: res[h]='ticks-mismatch-skip';continue
  os.killpg(o['pgrp'],signal.SIGTERM);res[h]='sigterm'
for _ in range(24):
  time.sleep(5)
  alive=[]
  for h in ('h0','h1'):
    o=json.load(open(S/h/'OWNER.json'))['identity']
    try: os.killpg(o['pgrp'],0);alive.append(h)
    except ProcessLookupError: pass
  if not alive: break
g=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
print(json.dumps(dict(stop=res,still_alive=alive,gpu=g)))
