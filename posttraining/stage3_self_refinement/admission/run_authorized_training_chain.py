"""One serialized execution owner for the existing native/train/reload/export CLIs.
No synthesized gates, no retries, no scientific API calls. Each subprocess owns its
existing safety checks and training child deadline. Stop on the first real failure.
"""
import datetime,fcntl,hashlib,json,os,pathlib,subprocess,sys
P=pathlib.Path
cfgpath=P(sys.argv[1]).resolve();cfg=json.loads(cfgpath.read_text());root=P(cfg['execution_root']).resolve()
root.mkdir(parents=True,exist_ok=True)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_new(p,data):
 with p.open('x')as f:json.dump(data,f,indent=2);f.write('\n')
def status(data):
 p=root/'STATE.tmp';p.write_text(json.dumps(data,indent=2)+'\n');os.replace(p,root/'STATE.json')
with (root/'owner.lock').open('a+')as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 stat=(P('/proc')/str(os.getpid())/'stat').read_text().rsplit(')',1)[1].split()
 write_new(root/'OWNER.json',{'pid':os.getpid(),'start_ticks':int(stat[19]),'boot_id':P('/proc/sys/kernel/random/boot_id').read_text().strip(),'config_sha256':hashlib.sha256(cfgpath.read_bytes()).hexdigest(),'driver_sha256':hashlib.sha256(P(__file__).read_bytes()).hexdigest(),'started_at':stamp()})
 for stage in cfg['stages']:
  name=stage['name'];assert name.replace('-','').replace('_','').isalnum()
  argv=stage['argv'][:]
  if '{LAST_CHECKPOINT}'in argv:
   ck=P(cfg['full_run'])/'checkpoints/last-checkpoint';assert ck.is_symlink();actual=ck.resolve(strict=True);assert actual.parent==(P(cfg['full_run'])/'checkpoints').resolve();argv=[str(actual)if x=='{LAST_CHECKPOINT}'else x for x in argv]
  receipt={'stage':name,'argv':argv,'started_at':stamp(),'state':'STARTING'}
  status(receipt);print('START',name,receipt['started_at'],flush=True)
  with (root/(name+'.log')).open('xb')as log:
   child=subprocess.Popen(argv,cwd=cfg['runtime'],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),stdout=log,stderr=subprocess.STDOUT)
   receipt.update(pid=child.pid,state='RUNNING');status(receipt);print('PID',name,child.pid,flush=True)
   code=child.wait()
  receipt.update(returncode=code,finished_at=stamp(),state='PROCESS_EXITED'if code==0 else'BLOCKED')
  write_new(root/(name+'.process.json'),receipt);status(receipt);print('EXIT',name,code,flush=True)
  if code:sys.exit(code)
 status({'state':'TRAINING_CHAIN_PROCESS_COMPLETE_REQUIRES_PARENT_ARTIFACT_VERIFICATION','finished_at':stamp(),'export_dir':cfg['export_dir'],'test40_done':False})
 print('TRAINING_CHAIN_FINISHED_REQUIRES_PARENT_VERIFICATION',flush=True)
