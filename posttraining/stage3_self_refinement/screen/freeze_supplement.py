"""Freeze completed supplemental records, read-only on collection; no generation."""
from pathlib import Path
import json,hashlib,datetime,tarfile

# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
BASE=Path(RETRACE_SHARE + '/stage3-supplement')
OUT=Path(RETRACE_SHARE + '/stage3-supplement-screen/v1')
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'manifest.json').exists(): raise RuntimeError('snapshot already exists')
 auth=json.loads((BASE/'AUTHORIZED_NOT_LAUNCHED.json').read_text())
 split=json.loads(Path(RETRACE_SHARE + '/stage2-agentic/split/SPLIT_STAGE2.json').read_text())
 expected={(int(i),r) for i in auth['target_ids'] for r in ['rep3','rep4']}
 if set(auth['target_ids'])-set(split['train_stage2']) or set(auth['target_ids'])&set(split['dev']+split['test40_eval_only']): raise RuntimeError('split violation')
 sources=sorted(BASE.glob('arms/*/rep*/runs/*.json'))
 manifest=[];seen=set()
 for p in sources:
  rb=p.read_bytes();r=json.loads(rb);idx=r['idx'];rep=p.parents[1].name.split('_')[0];key=f'{idx:03}_{rep}'
  if (idx,rep) not in expected or (idx,rep) in seen:raise RuntimeError('unexpected/duplicate '+key)
  seen.add((idx,rep));d=OUT/'sources'/key;d.mkdir(parents=True)
  tp=Path(r['transcript_json']);tb=tp.read_bytes()
  (d/'result.json').write_bytes(rb);(d/'transcript.json').write_bytes(tb)
  if rb!=p.read_bytes() or tb!=tp.read_bytes():raise RuntimeError('unstable '+key)
  manifest.append(dict(key=key,idx=idx,rep=rep,native_success=r['success'],source=str(p),transcript_source=str(tp),local_dir=RETRACE_OUTPUTS + '/stage3-supplement-screen/v1/sources/'+key,result_sha256=sha(rb),transcript_sha256=sha(tb)))
 cutoff=datetime.datetime.now(datetime.timezone.utc).isoformat()
 for row in manifest:
  if sha(Path(row['source']).read_bytes())!=row['result_sha256'] or sha(Path(row['transcript_source']).read_bytes())!=row['transcript_sha256']:raise RuntimeError('source drift '+row['key'])
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
 (OUT/'SPLIT_STAGE2.json').write_text(json.dumps(split,indent=2))
 (OUT/'AUTHORIZED_SUPPLEMENT.json').write_text(json.dumps(auth,indent=2))
 report=dict(snapshot_utc=cutoff,expected=len(expected),completed=len(seen),pending=[dict(idx=i,rep=r,key=f'{i:03}_{r}') for i,r in sorted(expected-seen)],native_success=sum(x['native_success'] for x in manifest),source_readback_verified=True,training_ready=False)
 (OUT/'SNAPSHOT.json').write_text(json.dumps(report,indent=2))
 archive=OUT.parent/'v1-sources.tar.gz'
 with tarfile.open(archive,'w:gz') as tf:tf.add(OUT,arcname='v1')
 print(json.dumps(report));print('ARCHIVE',str(archive),archive.stat().st_size,sha(archive.read_bytes()))
if __name__=='__main__':main()
