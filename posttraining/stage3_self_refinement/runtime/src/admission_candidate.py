"""Read-only replay of separately pinned admission validators. Never runs assemble/release."""
import importlib
import json
from pathlib import Path
import sys
from contract import require, read_json, sha, digest, verify_ref, validate_tokens
from evidence_gates import ref


def validate_candidate(root,index_path,build_sha,tokenizer_source=None,upstream_map=None):
    root=Path(root).resolve();index_path=Path(index_path).resolve();base=root.parent
    require(index_path==root/'selected/INDEX.jsonl','exact admission selected index required')
    pointer=base/'CURRENT_BUILD.json'
    require(sha(pointer)==build_sha,'admission build pointer mismatch')
    build=read_json(pointer);manifest=verify_ref(base,{'path':build['path'],'sha256':build['sha256']})
    code=read_json(manifest)['implementation_hashes']
    for rel,h in code.items():require(sha(base/rel)==h,'admission validator source changed: '+rel)
    names=('common','receipts','scientific','assemble','render_seal','legacy')
    require(all('src/'+n+'.py' in code for n in names),'unbound admission validator')
    require(not any(n in sys.modules for n in names),'admission module namespace already populated')
    sys.path.insert(0,str(base/'src'))
    common=importlib.import_module('common')
    require(common.ROOT.resolve()==root,'validator root mismatch')
    # Legacy replay sometimes calls freeze on an existing request; compare-only here.
    def readonly_freeze(p,data):
        require(Path(p).is_file() and Path(p).read_bytes()==data,'replay would create/change journal')
        return common.sha(data)
    def forbidden(*a,**kw):raise RuntimeError('candidate acceptance forbids admission writes and scientific calls')
    for name in ('put','put_rows','append','atomic_bytes','durable_mkdir','lock'):setattr(common,name,forbidden)
    common.freeze=readonly_freeze
    # Explicit byte-preserving relocation only for LOCK upstream references. Never edit LOCK.
    mapping=read_json(upstream_map) if upstream_map else {}
    lock=read_json(root/'LOCK.json');require(set(mapping)<=set(lock['upstream_files']),'unknown upstream relocation')
    translated={}
    for original,item in mapping.items():
        p=verify_ref(Path(upstream_map).resolve().parent,item)
        require(sha(p)==lock['upstream_files'][original],'relocation bytes differ from original LOCK')
        translated[original]=p
    common.Path=lambda value: translated.get(str(value),Path(value))
    if tokenizer_source:common.P1=Path(tokenizer_source).resolve()
    common.verify_lock()
    receipts=importlib.import_module('receipts');receipts.post=forbidden
    seal_module=importlib.import_module('render_seal');seal_module.verify_seal()
    scientific=importlib.import_module('scientific');assembly=importlib.import_module('assemble')
    before=sha(index_path);rows=common.rows(index_path)
    require(rows and len({r['uid'] for r in rows})==len(rows),'empty/duplicate selected candidate index')
    full=common.unique(common.rows(root/'full/INDEX.jsonl'),'uid');causal=read_json(root/'CAUSAL_MAP.json')
    census=common.unique(common.rows(root/'CENSUS.jsonl'),'uid');flags=read_json(root/'mech_flags_v1.json')
    routes=common.unique(read_json(root/'route_audit_v1/audit_routes.json'),'uid')
    members=read_json(root/'route_audit_v1/route_members.json');payloads={};canonical=[]
    split=read_json(root/'SPLIT_STAGE2.json')
    for row in rows:
        u=row['uid'];cr=causal[u];idx=full[u];entry=census[u]
        require(entry['status']=='CANDIDATE' and entry['key']==row['source_key']==idx['source_key'],'candidate identity/status')
        rep,target=u.split(':');require(int(target)==row['idx'] and row['source_key']==f'{int(target):03d}_{rep}','target identity mismatch')
        require(row['idx'] in split['train_stage2'] and row['idx'] not in split['dev']+split['test40_eval_only'],'held-out target')
        require(row['source_key'] not in {'174_rep2','069_rep3','160_rep3','011_rep4'},'quarantine')
        src=root/'sources'/row['source_key'];source_manifest=read_json(src/'SOURCE_MANIFEST.json')
        for item in source_manifest:
            require(sha(src/item['name'])==item['sha256'],'source artifact changed')
        require(row['source_hashes']==idx['source_hashes']==cr['source_hashes'],'source mapping changed')
        for name,h in row['source_hashes'].items():require(sha(src/name)==h,'causal source bytes changed')
        requests=common.rows(src/'conversation/llm_requests.jsonl')
        messages=read_json(src/'conversation/latest_messages.json')['messages']
        require(requests and len(requests)==len(idx['request_proofs']),'complete request proof coverage')
        for request,proof in zip(requests,idx['request_proofs']):
            require(request['messages']==messages[:len(request['messages'])] and request['tools']==requests[0]['tools'],'causal prefix/tools changed')
            require(common.digest(request)==proof['request_sha256'],'request proof hash changed')
        route_ids=[rid for rid,us in members.items() if u in us];require(len(route_ids)==1,'route membership ambiguous')
        audit=scientific.audit_receipt(routes[route_ids[0]])
        require(audit['verdict']['route_verdict']=='VALID' and common.digest(audit)==row['audit_receipt_sha256'],'audit mismatch')
        samples=[scientific.selector_receipt(u,n) for n in (1,2)]
        require([common.digest(s) for s in samples]==row['selector_receipt_sha256'],'selector mismatch')
        require(samples[0]['sample']==1 and samples[1]['sample']==2 and samples[0]['evidence']['response_path']!=samples[1]['evidence']['response_path'],'two actual samples required')
        require(sha(root/idx['token_path'])==idx['tokens_sha256']==row['eligible_tokens_sha256'],'eligible tokens changed')
        original=read_json(root/idx['token_path'])
        rebuilt,turns=assembly.mask_row(idx,original,[s['parsed'] for s in samples],flags[u],cr)
        token_path=root/row['token_path'];require(sha(token_path)==row['tokens_sha256'],'selected tokens changed')
        require(read_json(token_path)==rebuilt and turns==row['turns'] and row['render_index_sha256']==common.digest(idx),'selected mask/span mismatch')
        require(row['render_seal_sha256']==sha(seal_module.SEAL),'render seal binding')
        payload=dict(uid=u,source_key=row['source_key'],input_ids=rebuilt['input_ids'],labels=rebuilt['labels'],
                     selected_mask=rebuilt['loss_mask'],eligible_mask=original['loss_mask'])
        validate_tokens(payload,idx['thinking_token_ids'],positive=True);payloads[u]=payload
        canonical.append(dict(uid=u,source_key=row['source_key'],payload_sha256=row['tokens_sha256'],
                              full_tokens=len(payload['input_ids']),selected_tokens=sum(payload['selected_mask'])))
    require(sha(index_path)==before,'candidate index changed during replay')
    common.verify_lock();seal_module.verify_seal()
    require(sha(pointer)==build_sha,'admission code changed during replay')
    return payloads,dict(selected_index_sha256=before,selected_token_hashes={r['token_path']:r['tokens_sha256'] for r in rows},
        render_seal_sha256=sha(seal_module.SEAL),admission_build=ref(pointer),admission_index=ref(index_path),
        admission_validators={n:ref(base/'src'/f'{n}.py') for n in names},canonical_index=canonical)
