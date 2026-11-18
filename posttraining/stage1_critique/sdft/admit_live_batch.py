"""Independent live-evidence admission for the NEW fresh batch; old converter stays closed.
Only export after native completion, unchanged runtime/source/weights, split,
PreparedRequest byte evidence, response lineage and full message checks pass.
No training loop or scoring code. No repair/truncation of historical messages.
"""
import argparse, copy, hashlib, json, math
from pathlib import Path
from capture_adapter import load_json, normalize_messages, validate_conversation, require, MANIFEST_SHA256



# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
def digest(raw): return hashlib.sha256(raw).hexdigest()
def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def normalized_assistant(message):
    value={'role':'assistant','content':message.get('content') or None}
    if message.get('tool_calls'):
        value['tool_calls']=copy.deepcopy(message['tool_calls'])
    return normalize_messages([value])[0]

def select_indices(n):
    require(n>0,'No decision states')
    return sorted({round((n-1)*x/3) for x in range(4)})

def group_fixed_budget_attempts(wires):
    """Fixed-output-budget protocol: every attempt of one logical request must carry
    an identical payload; only the final attempt may be 'ok'; earlier attempts must be
    recorded reasoning truncations (finish_reason=length, empty visible output).
    Returns the list of final attempts in request order. Never drops evidence silently:
    any other shape is rejected."""
    groups={}
    for w in wires:
        groups.setdefault(w['request_index'],[]).append(w)
    finals=[]
    for n in sorted(groups):
        g=sorted(groups[n],key=lambda w:w['attempt'])
        require([w['attempt'] for w in g]==list(range(1,len(g)+1)),f'Attempt numbering gap at request {n}')
        for prev in g[:-1]:
            require(prev.get('outcome')=='empty','Non-truncation retry requires separate audit')
            ch=(prev.get('response') or {}).get('choices') or [{}]
            m=ch[0].get('message') or {}
            # Earlier attempt must have produced NO visible output: either the budget
            # was exhausted inside reasoning (length) or the model stopped after
            # reasoning only (stop). Both are recorded model behaviour, not repairs.
            require(ch[0].get('finish_reason') in ('length','stop') and not m.get('content') and not m.get('tool_calls'),'Retry after a response with visible output')
            require(prev['request']==g[-1]['request'],'Retry changed the payload (budget/history drift)')
            require(prev['request'].get('max_tokens')==g[-1]['request'].get('max_tokens')==32768,'Output budget changed across attempts')
        finals.append(g[-1])
    return finals


def verify_wire_record(row, model):
    require(row.get('prepared_request_verified') is True,'Unverified recorder snapshot')
    raw=row.get('raw_request_body_utf8')
    require(isinstance(raw,str),'Missing raw POST body')
    require(digest(raw.encode())==row.get('raw_request_body_sha256'),'POST bytes hash mismatch')
    request=load_json(raw)
    require(request==row.get('request'),'Serialized request mismatch')
    canonical=json.dumps(request,ensure_ascii=False,sort_keys=True).encode()
    require(digest(canonical)==row.get('request_sha256'),'Payload hash mismatch')
    require(request.get('model')==model,'Request model mismatch')
    require(row.get('input_message_count')==len(request['messages']),'History removed before POST')
    require(row.get('phase')=='main','Non-main phase rejected')
    require(row.get('outcome')=='ok' and row.get('status_code')==200,'Recovery/error attempt is not a training state')
    require(type(row.get('attempt')) is int and row['attempt']>=1,'Missing attempt number')
    require(row.get('call_id') and type(row.get('request_index')) is int,'Missing logical request identity')
    require(request.get('max_tokens')==32768 and request.get('reasoning_effort')=='low','Sampling settings drift')
    response=row.get('response',{})
    require(response.get('id') and response.get('model')==model,'Response identity/model mismatch')
    choices=response.get('choices',[])
    require(len(choices)==1 and choices[0].get('finish_reason') in ('stop','tool_calls'),'Clipped/unfinished response')
    require(isinstance(choices[0].get('message'),dict),'No assistant response')
    return request


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True)
    ap.add_argument('--expected-manifest-sha256',required=True)
    ap.add_argument('--root',default=RETRACE_SHARE + '/retrace-continual')
    args=ap.parse_args();root=Path(args.root);run=Path(args.run)
    require(file_sha(run/'launch_manifest.json')==args.expected_manifest_sha256,'Launch plan changed')
    plan=load_json((run/'launch_manifest.json').read_bytes())
    exit=load_json((run/'exit_receipt.json').read_bytes())
    require(exit['process_returncode']==0,'Native driver failed')
    import psutil
    launch=load_json((run/'launch_receipt.json').read_bytes())
    try:
        proc=psutil.Process(launch['wrapper_pid'])
        require(proc.create_time()!=launch['create_time'] or proc.status()==psutil.STATUS_ZOMBIE,'Collection wrapper still active')
    except psutil.NoSuchProcess:pass
    serve=plan['serving_process'];proc=psutil.Process(serve['pid'])
    require(proc.create_time()==serve['create_time'] and proc.cmdline()==serve['argv'],'Student serving identity changed')
    for rel,h in plan['source_sha256'].items():require(file_sha(root/rel)==h,f'Native source drift: {rel}')
    for name,info in plan['model_files'].items():
        p=Path(plan['model'])/name
        require(p.stat().st_size==info['size'] and file_sha(p)==info['sha256'],f'Checkpoint drift: {name}')
    mp=root/'data/reference-v1/provenance/manifest.json'
    require(file_sha(mp)==MANIFEST_SHA256==plan['reference_manifest_sha256'],'Reference corpus changed')
    corpus=load_json(mp.read_bytes())
    rp=root/'data/reference-v1/train/train_references.jsonl'
    require(file_sha(rp)==corpus['artifact_sha256']['train_references.jsonl']==plan['reference_sha256'],'Train reference drift')
    refs={r['idx']:r for r in map(load_json,rp.read_text().splitlines())}
    idxs=plan['indices']; require(set(idxs)<=set(corpus['split']['train']) and not set(idxs)&set(corpus['split']['test']),'Split violation')
    records=[load_json(p.read_bytes()) for p in sorted((run/'runs').glob('*.json'))]
    require(len(records)==len(idxs),'Missing native case output')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(plan['model'],local_files_only=True,padding_side='left')
    output=[];sources=[];tools_ref=None
    for result in records:
        idx=result['idx'];require(idx in idxs,'Unrequested case')
        require(not result.get('error'),'Native exception, not a model failure')
        session=Path(result['session_dir'])
        wire_path=session/'conversation/wire_log.jsonl'
        all_wires=[load_json(l) for l in wire_path.read_text().splitlines()]
        wires=group_fixed_budget_attempts(all_wires)
        recorder=[load_json(l) for l in (session/'conversation/llm_requests.jsonl').read_text().splitlines()]
        require(len(wires)==len(recorder)>0,'Missing attempts/request captures')
        truncation_retries=len(all_wires)-len(wires)
        reference=refs[idx]; require(reference['route_verdict']=='VALID','Reference is not eligible')
        require(reference['target_smiles']==plan['targets'][str(idx)],'Target identity drift')
        prior=[];seen=set();all_rows=[]
        for n,(w,record) in enumerate(zip(wires,recorder),1):
            require(w['request_index']==record['request_index']==n,'Capture order/recorder mismatch')
            req=verify_wire_record(w,plan['served_model'])
            # Compare against the pinned native serializer, not raw optional-null
            # fields in the pre-dispatch recorder. No orphan-drop/recovery here.
            import sys
            sys.path.insert(0,str(root/'harness'))
            from core.llm_client import _build_chat_request
            from services.llm_models import to_chat_completions
            native=to_chat_completions(_build_chat_request(model=req['model'],messages=record['messages'],tools=record.get('tools'),temperature=req['temperature'],max_tokens=req['max_tokens'],reasoning_effort=req['reasoning_effort'])).model_dump(exclude_none=True)
            require(req['messages']==native['messages'],'Native serializer/wire messages differ: reject, never repair')
            require(req.get('tools')==native.get('tools'),'Tool schema serialization drift')
            require(w['ts']>=__import__('datetime').datetime.fromisoformat(plan['created_at']).timestamp(),'Historical capture rejected')
            identity=(w['call_id'],w['response']['id']);require(identity not in seen,'Duplicate capture identity');seen.add(identity)
            messages=normalize_messages(req['messages']);tools=req['tools']
            validate_conversation(messages,tools)
            require(all(not(a['role']==b['role']=='assistant') for a,b in zip(messages,messages[1:])),'Adjacent assistant turns')
            require(reference['target_smiles'] in next(m['content'] for m in messages if m['role']=='user'),'Wrong task')
            require(not any('<privileged_reference>' in str(m.get('content','')) for m in messages),'Student privilege leakage')
            for m in messages:
                if m['role']=='assistant':require(normalized_assistant(m) in prior,'Assistant history not tied to student responses')
            prior.append(normalized_assistant(w['response']['choices'][0]['message']))
            if tools_ref is None:tools_ref=tools
            require(tools==tools_ref,'Tool inventory drift')
            ids=tok.apply_chat_template(messages,tools=tools,enable_thinking=True,reasoning_effort='low',add_generation_prompt=True,tokenize=True,return_dict=False)
            feedback={'native_success':result.get('success'),'iterations':result.get('iterations'),'warning':'Native ROSC/stock/pricing observations are search feedback, not expert chemical truth; fallback prices are not measured quotes.'}
            privileged='\n'.join(s['raw'] for s in reference['steps'])+'\n\nObserved student run feedback (not reward or certification):\n'+json.dumps(feedback,ensure_ascii=False)
            all_rows.append({'prompt':messages,'privileged_context':privileged,'meta':{'idx':idx,'request_index':w['request_index'],'wire_seq':n,'request_id':w['response']['id'],'call_id':w['call_id'],'session_dir':str(session),'actual_wire_verified':True,'formal_training_admissible':True,'checkpoint_sha256':plan['checkpoint_sha256'],'prompt_tokens':len(ids),'reference_uid':reference['uid'],'state_distribution':'fresh base-policy visited states; SDFT generates current-student completions','wire_sha256':file_sha(wire_path)}})
        selected=select_indices(len(all_rows))
        output.extend(all_rows[i] for i in selected)
        sources.append({'idx':idx,'session_dir':str(session),'wire_sha256':file_sha(wire_path),'requests':len(wires),'fixed_budget_truncation_retries':truncation_retries,'selected_request_indices':[all_rows[i]['meta']['request_index'] for i in selected]})
    for i,row in enumerate(output):row['meta']['sample_index']=i
    out=run/'admitted';out.mkdir(exist_ok=False)
    rows_path=out/'train.jsonl';rows_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output))
    (out/'tools.json').write_text(json.dumps(tools_ref,ensure_ascii=False))
    receipt={'status':'admitted','rows':len(output),'sources':sources,'data_sha256':file_sha(rows_path),'tools_sha256':file_sha(out/'tools.json'),'manifest_sha256':args.expected_manifest_sha256,'checkpoint_sha256':plan['checkpoint_sha256'],'formal_training_admissible':True,'dev_split':False,'heldout_used':False,'verifier_sha256':file_sha(__file__)}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print('ADMITTED',json.dumps(receipt))

if __name__=='__main__':main()
