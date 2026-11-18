#!/usr/bin/env python3
"""Validate existing-result screens. Offline; no inference, remote writes, or source mutation."""
from pathlib import Path
import json,hashlib,argparse,collections,csv
ROOT=Path(__file__).resolve().parent
VERDICTS={'ADMIT_CANDIDATE','EXCLUDE','HOLD'}
def strings(value):
 if isinstance(value,str):yield value
 elif isinstance(value,dict):
  for v in value.values():yield from strings(v)
 elif isinstance(value,list):
  for v in value:yield from strings(v)
def need(ok,msg):
 if not ok:raise ValueError(msg)
def validate_target(v,expected_idx,sources):
 need(type(v.get('target_id')) is int and v['target_id']==expected_idx,'target id mismatch')
 rr=v.get('trajectories');need(isinstance(rr,list) and len(rr)==2,f'{expected_idx}: expected two trajectories')
 expected={f'{expected_idx:03}_rep1',f'{expected_idx:03}_rep2'}
 need({x.get('key') for x in rr}==expected,f'{expected_idx}: keys mismatch')
 for x in rr:
  key=x['key'];r,t=sources[key];need(x.get('verdict') in VERDICTS,f'{key}: invalid verdict')
  need(x.get('confidence') in {'high','medium','low'},f'{key}: invalid confidence')
  for field in ['route_assessment_zh','process_assessment_zh','followup_zh']:
   need(isinstance(x.get(field),str),f'{key}: missing string {field}')
  need(isinstance(x.get('substantive_flags'),list) and all(isinstance(q,str) for q in x['substantive_flags']),f'{key}: bad flags')
  rs=x.get('reading_scope');need(isinstance(rs,dict),f'{key}: reading scope missing')
  for f in ['all_final_steps_read','compact_process_read']:need(type(rs.get(f)) is bool,f'{key}: {f} not boolean')
  need(isinstance(rs.get('extra_event_indices'),list) and all(type(i) is int and 0<=i<len(t['events']) for i in rs['extra_event_indices']),f'{key}: invalid extra event index')
  if r['success']:
   need(rs['all_final_steps_read'] and rs['compact_process_read'],f'{key}: incomplete first-pass reading')
  else:need(x['verdict']!='ADMIT_CANDIDATE',f'{key}: native failure cannot admit full trajectory')
  evidence=x.get('evidence');need(isinstance(evidence,list) and len(evidence)>0,f'{key}: evidence required')
  for ei,e in enumerate(evidence):
   loc=f'{key}: evidence {ei}';quote=e.get('quote');need(isinstance(quote,str) and len(quote.strip())>0,loc+' empty quote')
   need(isinstance(e.get('explanation_zh'),str) and bool(e['explanation_zh'].strip()),loc+' explanation missing')
   if e.get('source')=='events':
    ix=e.get('event_index');need(type(ix) is int and 0<=ix<len(t['events']),loc+' bad event index'); container=t['events'][ix]
   elif e.get('source')=='result':
    need(e.get('event_index') is None,loc+' result must not have event_index');container=r
    step=e.get('step_number')
    if step is not None:
     steps=(r.get('best_route') or {}).get('steps') or []
     need(type(step) is int and 1<=step<=len(steps),loc+' bad final step index');container=steps[step-1]
   else:raise ValueError(loc+' invalid source')
   need(any(quote in ss for ss in strings(container)),loc+' quotation not found literally in exact source')
  if x['verdict']=='EXCLUDE':need(bool(x['substantive_flags']),f'{key}: exclusion must state flags')
 admitted=[x['key'] for x in rr if x['verdict']=='ADMIT_CANDIDATE'];hold=any(x['verdict']=='HOLD' for x in rr)
 pref=v.get('preferred_key');need(pref is None or pref in admitted,f'{expected_idx}: preferred key not admitted')
 need(isinstance(v.get('selection_reason_zh'),str),f'{expected_idx}: selection reason missing')
 needed=v.get('needs_additional_rollout');need(needed is None or type(needed) is bool,f'{expected_idx}: needs flag invalid')
 expected_needed=False if admitted else None if hold else True
 need(needed is expected_needed,f'{expected_idx}: needs_additional_rollout conflicts with verdicts')
 return v

def load_sources():
 manifest=json.loads((ROOT/'manifest.json').read_text()); lock=json.loads((ROOT/'SCREEN_LOCK.json').read_text())
 for file,key in [('manifest.json','source_manifest_sha256'),('RUBRIC.md','rubric_sha256'),('structure_facts.json','facts_sha256')]:
  need(hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==lock[key],file+' lock mismatch')
 src={}
 for x in manifest:
  d=Path(x['local_dir']); rb=(d/'result.json').read_bytes();tb=(d/'transcript.json').read_bytes()
  need(hashlib.sha256(rb).hexdigest()==x['result_sha256'],x['key']+' result changed')
  need(hashlib.sha256(tb).hexdigest()==x['transcript_sha256'],x['key']+' transcript changed')
  src[x['key']]=(json.loads(rb),json.loads(tb))
 for idx,h in lock['packet_sha256'].items():need(hashlib.sha256((ROOT/'packets'/f'{int(idx):03}.json').read_bytes()).hexdigest()==h,'packet changed '+idx)
 need(len(src)==260,'source count mismatch');return src

def validate_shard(j,sources):
 ass=json.loads((ROOT/f'assignment-{j:02}.json').read_text());f=ROOT/'reviews'/f'shard-{j:02}.json';v=json.loads(f.read_text())
 need(isinstance(v,list),str(f)+' root must be list')
 expected=set(ass['primary_targets']+ass['overlap_targets']); actual=[x.get('target_id') for x in v]
 need(len(actual)==len(set(actual)) and set(actual)==expected,f'shard {j}: expected {sorted(expected)}, got {actual}')
 for x in v:validate_target(x,x['target_id'],sources)
 return ass,v

def aggregate(sources):
 primary={};overlaps=[];review_hashes={}
 for j in range(6):
  ass,v=validate_shard(j,sources);review_hashes[f'shard-{j:02}']=hashlib.sha256((ROOT/'reviews'/f'shard-{j:02}.json').read_bytes()).hexdigest()
  for x in v:
   if x['target_id'] in ass['primary_targets']:
    need(x['target_id'] not in primary,'duplicate primary');primary[x['target_id']]=x
   else:overlaps.append({'reviewer':j,'review':x})
 need(len(primary)==130,'primary target coverage not complete')
 disagreements=[]
 for o in overlaps:
  x=o['review'];a={r['key']:r['verdict'] for r in x['trajectories']};b={r['key']:r['verdict'] for r in primary[x['target_id']]['trajectories']}
  if a!=b or x['preferred_key']!=primary[x['target_id']]['preferred_key']:disagreements.append({'target_id':x['target_id'],'secondary_reviewer':o['reviewer'],'primary':b,'secondary':a,'primary_preferred':primary[x['target_id']]['preferred_key'],'secondary_preferred':x['preferred_key']})
 # Optional parent adjudications explicitly replace target verdicts, preserving original review files.
 af=ROOT/'ADJUDICATIONS.json';adjudications=json.loads(af.read_text()) if af.exists() else []
 adjids=[]
 for rec in adjudications:
  need(isinstance(rec.get('reason'),str) and rec['reason'].strip(),'adjudication needs reason');v=rec['target'];validate_target(v,v['target_id'],sources);need(v['target_id'] in primary,'unknown adjudication target');adjids.append(v['target_id']);primary[v['target_id']]=v
 need(len(adjids)==len(set(adjids)),'duplicate adjudication')
 unresolved=[x for x in disagreements if x['target_id'] not in adjids]
 need(not unresolved,'overlap disagreements require source adjudication: '+json.dumps(unresolved,ensure_ascii=False))
 records=[];targets=[]
 for idx,x in sorted(primary.items()):
  admitted=[r for r in x['trajectories'] if r['verdict']=='ADMIT_CANDIDATE'];hold=[r for r in x['trajectories'] if r['verdict']=='HOLD'];pref=x['preferred_key'];tie=False
  if admitted and pref is None:pref=sorted(r['key'] for r in admitted)[0];tie=True
  group='HAS_CANDIDATE' if admitted else 'UNRESOLVED' if hold else 'NEEDS_SUPPLEMENT'
  targets.append({'target_id':idx,'group':group,'selected_key':pref,'admitted_candidates':[r['key'] for r in admitted],'hold_candidates':[r['key'] for r in hold],'selection_reason_zh':x['selection_reason_zh'],'tie_representative_not_quality_rank':tie,'training_ready':False})
  for r in x['trajectories']:records.append(r|{'target_id':idx,'native_success':sources[r['key']][0]['success'],'selected':r['key']==pref,'training_ready':False})
 need(len(records)==260 and len({r['key'] for r in records})==260,'trajectory coverage incomplete')
 summary={'targets':len(targets),'trajectories':len(records),'native_success':sum(r['native_success'] for r in records),'verdict_counts':dict(collections.Counter(r['verdict'] for r in records)),'target_groups':dict(collections.Counter(x['group'] for x in targets)),'selected_candidates':sum(x['selected_key'] is not None for x in targets),'overlap_target_count':len(overlaps),'initial_overlap_disagreements':disagreements,'adjudicated_targets':sorted(adjids),'review_hashes':review_hashes,'scope':'Exploratory final-route/compact-process screen, NOT full reasoning-token audit, expert chemistry certification, loss-mask acceptance or training-ready export.'}
 (ROOT/'SCREENED_TRAJECTORIES.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));(ROOT/'TARGET_SELECTION.json').write_text(json.dumps(targets,ensure_ascii=False,indent=2));(ROOT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 with (ROOT/'TARGET_SELECTION.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['target_id','group','selected_key','selection_reason_zh','training_ready']);w.writeheader();w.writerows({k:x[k] for k in w.fieldnames} for x in targets)
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--shard',type=int);p.add_argument('--aggregate',action='store_true');args=p.parse_args();src=load_sources()
 if args.shard is not None:
  ass,v=validate_shard(args.shard,src);print(json.dumps({'status':'PASS','shard':args.shard,'targets':len(v),'trajectories':2*len(v)},ensure_ascii=False))
 elif args.aggregate:print(json.dumps(aggregate(src),ensure_ascii=False,indent=2))
 else:print(json.dumps({'status':'SOURCE_LOCK_PASS','sources':len(src)},ensure_ascii=False))
if __name__=='__main__':main()
