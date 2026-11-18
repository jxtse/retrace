"""Full40 robust native accounting, with explicit historical or strict final joins."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import os
from contract import require, read_json, sha, write_new, verify_ref
from eval_s3 import IDS, ALIAS, shards, verify_tree, verify_serving, checkpoint
from evidence_gates import ref, code_pins


def classify_wire(w,alias):
    invalid=dict(infra=True,identity_or_effort_drift=False,choices=0,no_visible_output=0,finishes={},reason='malformed_wire')
    if not isinstance(w,dict):return invalid
    req,resp=w.get('request'),w.get('response')
    if not isinstance(req,dict) or not isinstance(resp,dict):return invalid
    choices=resp.get('choices')
    if not isinstance(choices,list) or not choices:return dict(invalid,reason='empty_or_invalid_choices')
    empty=0;finishes=Counter()
    for c in choices:
        if not isinstance(c,dict) or not isinstance(c.get('message'),dict):return invalid
        m=c['message'];content=m.get('content');calls=m.get('tool_calls')
        if (content is not None and not isinstance(content,str)) or (calls is not None and (not isinstance(calls,list) or not all(isinstance(x,dict) for x in calls))):return invalid
        if not isinstance(c.get('finish_reason'),str):return invalid
        if not content and not calls:empty+=1;finishes[c['finish_reason']]+=1
    drift=req.get('model')!=alias or resp.get('model')!=alias or req.get('reasoning_effort')!='medium'
    return dict(infra=bool(w.get('error')) or w.get('status_code')!=200 or w.get('outcome') not in ('ok','empty'),
        identity_or_effort_drift=drift,choices=len(choices),no_visible_output=empty,finishes=dict(finishes),reason='transport_event')


def active_consumers(paths, wrapper_sid=None):
    """Read Linux process argv only. A receipt alone cannot certify absence."""
    require(Path('/proc').is_dir(),'strict final join requires live Linux process census')
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:argv=(p/'cmdline').read_bytes().split(b'\0');words=[x.decode(errors='replace') for x in argv if x]
        except (OSError,PermissionError):continue
        try: sid=int((p/'stat').read_text().rsplit(')',1)[1].split()[3])
        except (OSError,ValueError,IndexError):sid=None
        if wrapper_sid is not None and sid==wrapper_sid:
            found.append(dict(pid=int(p.name),argv=words,session_id=sid));continue
        if any(Path(x).name in ('detached_runner.py','run_benchmark.py') for x in words) and any(str(path) in ' '.join(words) for path in paths):
            found.append(dict(pid=int(p.name),argv=words))
    return found


def strict_root(root,terminal):
    root=Path(root).resolve();lp=root/'launch_manifest.json';m=read_json(lp)
    require(m['schema']=='s3.eval-launch.v4','legacy launch forbidden in final mode')
    require(m['indices']==shards(IDS)[m['half']] and m['full_test40']==IDS,'half identity mismatch')
    isolation=read_json(verify_ref(root,m['isolation']));prepared=read_json(verify_ref(root,m['prepared']))
    pins=verify_ref(root,m['source_pins']); require(code_pins(pins)==prepared['source_pins_sha256'],'source pins changed')
    harness=Path(m['cwd']).resolve();require(Path(m['cwd']).is_absolute(),'relative harness path')
    require(Path(m['session_root']).resolve()==harness/'runs','session root drift')
    require(verify_ref(root,m['isolation']).parent/'native-isolated/harness'==harness,'harness isolation mismatch')
    verify_tree(harness.parent,isolation['source_sha256'])
    cfg=verify_ref(root,m['config']);require(sha(cfg)==isolation['config_sha256'],'config mismatch')
    require(verify_ref(root,m['runner'])==Path(m['argv'][1]) and sha(m['argv'][1])==isolation['runner_sha256'],'runner mismatch')
    require(m['env']['AGENT_CONFIG']==str(cfg) and m['env']['AGENT_MODEL']==ALIAS,'effective config/alias mismatch')
    require(m['argv'][m['argv'].index('--workers')+1]=='4','worker count drift')
    require(m['argv'][m['argv'].index('--only')+1]==','.join(map(str,m['indices'])),'argv shard mismatch')
    checkpoint(m['checkpoint'],m['adapter_sha256'],verify_ref(root,m['trained_receipt']))
    verify_serving(verify_ref(root,m['serving_proof']),m['adapter_sha256'],m['half'])
    require(sha(verify_ref(root,m['tool_preflight']))=='c8dfa51c4db7726dc79bb861341cb9d9bf0ae55586ff99258b0b4a84c0ba9ea2','tool preflight swapped')
    t=read_json(terminal)
    require(t['schema']=='s3.eval-terminal.v4' and verify_ref(Path(terminal).parent,t['launch'])==lp,'terminal launch binding')
    hashes={}
    for k,name in [('exit','exit_receipt.json'),('process','process_identity.json'),('launch_receipt','launch_receipt.json')]:
        path=verify_ref(Path(terminal).parent,t[k]);require(path==root/name,'terminal artifact wrong run');hashes[k]=ref(path)
    exit_r=read_json(root/'exit_receipt.json');process=read_json(root/'process_identity.json');lr=read_json(root/'launch_receipt.json')
    require(exit_r['process_returncode']==0,'wrapper failure')
    require(process['wrapper_pid']==lr['wrapper_pid'] and lr['argv']==m['argv'] and lr['launch_manifest_sha256']==sha(lp),'wrapper identity mismatch')
    require(process['argv']==m['argv'][m['argv'].index('--')+1:],'native child argv mismatch')
    require(not active_consumers([root,harness],process['wrapper_sid']),'active consumers remain')
    return m,dict(terminal=ref(terminal),launch=ref(lp),**hashes,active_consumers=[])


def aggregate(roots,alias=ALIAS,mode='historical',terminals=None):
    require(mode in ('historical','final'),'explicit accounting mode required')
    roots=[Path(p).resolve() for p in roots];results={};counts=Counter();cases=[];errors=[];joined=[];manifests={}
    if mode=='final':
        require(len(roots)==2 and terminals and len(terminals)==2,'two final halves and terminal receipts required')
        for root,t in zip(roots,terminals):
            try:m,proof=strict_root(root,t);manifests[root]=m;joined.append(proof)
            except (ValueError,KeyError,TypeError,OSError,AttributeError,UnicodeError) as e:errors.append({'root':str(root),'error':str(e)})
        if len(manifests)==2:
            if {m['half'] for m in manifests.values()}!={'h0','h1'}:errors.append({'error':'duplicate halves'})
            if len({m['adapter_sha256'] for m in manifests.values()})!=1:errors.append({'error':'different S3 checkpoints'})
    for root in roots:
        for p in sorted((root/'runs').glob('*.json')):
            try:
                r=read_json(p);require(isinstance(r,dict),'native result object required')
                i=r.get('idx');require(type(i) is int and i in IDS,'unknown result ID')
            except (ValueError,TypeError,OSError,UnicodeError) as e:
                m=re.match(r'^(\d{3})(?:_|\.)',p.name);i=int(m[1]) if m else None
                r={'idx':i,'_parse_error':str(e)}
            if i not in IDS:errors.append({'path':str(p),'error':'unidentifiable result'});continue
            if i in results:errors.append({'path':str(p),'error':'duplicate result ID'});results[i][0]['_duplicate']=True;continue
            if root in manifests and i not in manifests[root]['indices']:r['_wrong_half']=True
            results[i]=(r,p,root)
    for i in IDS:
        if i not in results:
            cases.append(dict(idx=i,status='missing',native_success=None));counts['missing']+=1;continue
        r,p,root=results[i];infra=[];wire_counts=Counter();drift=False
        success=r.get('success');valid_success=type(success) is bool
        if not valid_success:infra.append('missing_or_invalid_native_success')
        if any(r.get(k) for k in ('error','skipped','_parse_error','_duplicate','_wrong_half')):infra.append('native_error_or_malformed_result')
        session=r.get('session_dir');wire=None
        if isinstance(session,str) and session:
            session=Path(session).resolve()
            if mode=='historical' or (root in manifests and session.is_relative_to(Path(manifests[root]['session_root']).resolve())):
                candidate=(session/'conversation/wire_log.jsonl').resolve()
                if mode=='historical' or candidate.is_relative_to(Path(manifests[root]['session_root']).resolve()):wire=candidate
                else:infra.append('wire_outside_exact_harness_runspace')
            else:infra.append('session_outside_exact_harness_runspace')
        if wire is None or not wire.is_file():infra.append('missing_wire')
        else:
            try:
                for line in wire.read_text().splitlines():
                    wire_counts['wire_rows']+=1
                    try:c=classify_wire(json.loads(line),alias)
                    except (ValueError,TypeError):c=dict(infra=True,identity_or_effort_drift=False,no_visible_output=0,finishes={},reason='malformed_wire_json')
                    wire_counts['no_visible_output']+=c['no_visible_output']
                    for f,n in c['finishes'].items():wire_counts['empty_finish_'+f]+=n
                    if c['infra']:infra.append(c['reason'])
                    drift |= c['identity_or_effort_drift']
            except (OSError,UnicodeError):infra.append('unreadable_wire')
            if not wire_counts['wire_rows']:infra.append('empty_wire')
        if drift:infra.append('adapter_or_effort_drift')
        status='infra_invalid' if infra else 'native_success' if success else 'native_failure'
        counts[status]+=1;counts['raw_native_success']+=int(success is True);counts.update(wire_counts)
        cases.append(dict(idx=i,status=status,native_success=success if valid_success else None,infra_reasons=sorted(set(infra)),
            result_path=str(p),result_sha256=sha(p),wire_path=str(wire) if wire else None,wire_sha256=sha(wire) if wire and wire.is_file() else None,
            iterations=r.get('iterations'),elapsed_s=r.get('wall_seconds',r.get('time',r.get('elapsed_s'))),
            native_diagnostics={k:v for k,v in r.items() if k not in ('route_text','session_dir')},diagnostics=dict(wire_counts)))
    evaluated=mode=='final' and len(joined)==2 and not errors and len(results)==40 and counts['infra_invalid']==0
    return dict(mode=mode,historical_read_only=mode=='historical',denominator=40,coverage=len(results),counts=dict(counts),cases=cases,
        native_success_rate_full40=counts['native_success']/40,raw_native_success_rate_full40=counts['raw_native_success']/40,
        coverage_complete=len(results)==40,complete=evaluated,evaluated=evaluated,terminal_receipts=joined,join_errors=errors,
        chemistry_audit_performed=False,status='NATIVE_EVALUATED' if evaluated else 'HISTORICAL_RECOUNT' if mode=='historical' else 'NOT_EVALUATED')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--alias',default=ALIAS)
    p.add_argument('--mode',choices=['historical','final'],required=True);p.add_argument('--terminal-receipts',nargs='+')
    p.add_argument('--out',required=True);a=p.parse_args();write_new(a.out,aggregate(a.runs,a.alias,a.mode,a.terminal_receipts))
