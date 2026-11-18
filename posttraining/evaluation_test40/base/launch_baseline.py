"""Baseline (Qwen3.8-27B base, no LoRA) test40 @ medium. Same harness/config/serving recipe as S3 test40;
only differences: model alias qwen3.8-27b-student (base) and CHEMPRICE_DISABLE_MOLPORT=1 (MolPort key rejected).
Usage: python3 launch_baseline.py plan|execute"""
import hashlib,json,os,pathlib,subprocess,sys,time,urllib.request,datetime,socket,yaml

# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
W=pathlib.Path(RETRACE_SHARE + '/stage3-endtoend/authorized-training');R=W/'runtime'
B=pathlib.Path(RETRACE_SHARE + '/eval-base')
ALIAS='qwen3.8-27b-student'
PORT={'h0':int(os.environ.get('RETRACE_H0_PORT',8000)),'h1':int(os.environ.get('RETRACE_H1_PORT',8002))};CUDA={'h0':'2,3','h1':'0,1'}
def sha(p):
  h=hashlib.sha256()
  with open(p,'rb') as f:
    for b in iter(lambda:f.read(8<<20),b''):h.update(b)
  return h.hexdigest()
def put(p,d):
  with open(p,'x') as f: json.dump(d,f,indent=2)
def ticks(pid): return int(pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
def port_free(p):
  s=socket.socket()
  try: s.connect(('127.0.0.1',p)); return False
  except OSError: return True
  finally: s.close()
def need(c,m):
  if not c: raise SystemExit('PRECHECK FAIL: '+m)
def main(mode):
  reuse = mode=='relaunch-runners'
  orig={h:json.load(open(R/f'test40-s3-{h}-001/launch_manifest.json')) for h in PORT}
  rec={h:json.load(open(R/f'test40-s3-{h}-infra-recovery-001/launch_manifest.json')) for h in PORT}
  for h in PORT: need(rec[h]['config']==orig[h]['config'] and rec[h]['cwd']==orig[h]['cwd'],'recovery/original drift '+h)
  owners={h:json.load(open(W/f'serving-test40-v1/{h}/OWNER.json')) for h in PORT}
  ids=sorted(orig['h0']['indices']+orig['h1']['indices']);need(len(set(ids))==40 and ids==sorted(orig['h0']['full_test40']),'split')
  for h in PORT:
    need(sha(orig[h]['config']['path'])==orig[h]['config']['sha256'],'config drift '+h)
    need(reuse or port_free(PORT[h]),'port busy %d'%PORT[h])
  g=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout
  need(reuse or all(int(l.split(',')[1])<1000 for l in g.strip().splitlines()),'GPUs not free: '+g)
  need(not B.exists() or mode in ('plan','relaunch-runners'),'baseline root exists')
  plan={}
  for h in PORT:
    m=orig[h];out=B/'runs'/f'test40-base-{h}-001'
    cfg=yaml.safe_load(open(m['config']['path']));cfg['llm']['model']=ALIAS
    env=dict(rec[h]['effective_env']);env.update(AGENT_MODEL=ALIAS,AGENT_CONFIG=str(B/'config'/h/'eval_config.yaml'),
      MPLCONFIGDIR=str(out/'mpl'),NUMBA_CACHE_DIR=str(out/'numba'),XDG_CACHE_HOME=str(out/'cache'),CHEMPRICE_DISABLE_MOLPORT='1')
    argv=list(m['argv']);argv[argv.index('--out')+1]=str(out);argv[argv.index('--output-dir')+1]=str(out)
    need(argv[argv.index('--workers')+1]=='4','workers')
    serve=owners[h]['argv'];need(serve[serve.index('--port')+1]==str(PORT[h]),'serve port')
    plan[h]=dict(schema='base.test40-medium.v1',out=str(out),indices=m['indices'],full_test40=m['full_test40'],argv=argv,env=env,cwd=m['cwd'],
      session_root=m['session_root'],config_yaml=cfg,s3_config_ref=m['config'],serve_argv=serve,serve_cuda=CUDA[h],
      diff_vs_s3_test40=['llm.model / AGENT_MODEL = qwen3.8-27b-student (base, no adapter)','CHEMPRICE_DISABLE_MOLPORT=1'])
  if mode=='plan':
    for h in PORT: print(h,plan[h]['indices'],plan[h]['diff_vs_s3_test40'],plan[h]['config_yaml']['llm'])
    return
  for h in PORT:
    cp=B/'config'/h/'eval_config.yaml'
    if not reuse:
      cp.parent.mkdir(parents=True)
      with open(cp,'x') as f: yaml.safe_dump(plan[h]['config_yaml'],f,sort_keys=False)
    need(yaml.safe_load(open(cp))==plan[h]['config_yaml'],'config file mismatch '+h)
    plan[h]['config']=dict(path=str(cp),sha256=sha(cp))
  S=B/'serving'
  for h in ([] if reuse else PORT):
    d=S/h;d.mkdir(parents=True);log=open(d/'server.log','xb')
    senv=dict(os.environ,CUDA_VISIBLE_DEVICES=CUDA[h])
    for k in ('HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy'): senv.pop(k,None)
    pr=subprocess.Popen(plan[h]['serve_argv'],env=senv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    put(d/'OWNER.json',dict(identity=dict(pid=pr.pid,start_ticks=ticks(pr.pid),pgrp=pr.pid,boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()),
        argv=plan[h]['serve_argv'],cuda=CUDA[h],started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),half=h))
    print('SERVER_STARTED',h,pr.pid,flush=True)
  op=urllib.request.build_opener(urllib.request.ProxyHandler({}));deadline=time.time()+1200
  for h in PORT:
    while True:
      need(time.time()<deadline,'serving readiness deadline '+h)
      try:
        with op.open(f'http://127.0.0.1:{PORT[h]}/v1/models',timeout=3) as r: models=json.load(r)
        break
      except OSError: time.sleep(5)
    need(any(x['id']==ALIAS for x in models['data']),'alias '+h)
    body=json.dumps({'model':ALIAS,'messages':[{'role':'user','content':'Reply with OK.'}],'max_tokens':16,'temperature':0,'stream':False}).encode()
    with op.open(urllib.request.Request(f'http://127.0.0.1:{PORT[h]}/v1/chat/completions',data=body,headers={'Content-Type':'application/json'}),timeout=300) as r: ans=json.load(r)
    need(ans['model']==ALIAS and ans['usage']['completion_tokens']>0,'probe '+h)
    if not reuse: put(S/h/'READY_PROBE.json',dict(models=models,probe=ans))
    print('SERVER_READY',h,flush=True)
  if reuse:
    import socket as _s
    t=_s.socket();t.settimeout(3);t.connect(('127.0.0.1',8010));t.close();print('TOOL_OK',flush=True)
  for h in PORT:
    out=pathlib.Path(plan[h]['out']);out.mkdir(parents=True)
    put(out/'launch_manifest.json',plan[h])
    with open(out/'driver.log','xb') as log:
      pr=subprocess.Popen(plan[h]['argv'],cwd=plan[h]['cwd'],env=plan[h]['env'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    put(out/'launch_receipt.json',dict(wrapper_pid=pr.pid,start_ticks=ticks(pr.pid),launch_manifest_sha256=sha(out/'launch_manifest.json'),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('RUNNER_LAUNCHED',h,pr.pid,len(plan[h]['indices']),flush=True)
if __name__=='__main__': main(sys.argv[1])
