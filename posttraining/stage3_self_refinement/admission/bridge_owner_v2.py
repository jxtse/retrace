"""Portable final runtime v3 exporter, after verified admission native release."""
import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent / 'admission/src'))
from common import *
from render_seal import verify_seal,SEAL
from release_evidence import reference,selected_binding,verify_current_release
from scientific import audit_receipt,selector_receipt,receipt_provenance
from assemble import mask_row,full_census_gate


policy_contract = module('owner_runtime_contract', LOCAL / 'runtime/src/contract.py')
EXPORTS = LOCAL / 'integration/exports-owner-v2'


def freeze(path, data):
    # Only this exporter writes here; admission primitives retain their original roots.
    path = Path(path).resolve()
    require(path.is_relative_to(EXPORTS.resolve()), 'write outside owner exports forbidden')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        require(path.read_bytes() == data, f'frozen bytes changed: {path}')
    return sha(data)


def export_runtime(owner_decision):
    verify_lock();verify_seal()
    census=unique(rows(ROOT/'CENSUS.jsonl'));full_census_gate(list(census.values()))
    selected=rows(ROOT/'selected/INDEX.jsonl');require(bool(selected),'no genuinely selected rows to export')
    release=verify_current_release()
    binding=selected_binding()
    original_split=LOCAL/'source_assets/ORIGINAL_FROZEN_SPLIT.json'
    require(sha(original_split.read_bytes()) == policy_contract.SPLIT_SHA, 'original frozen split bytes changed')
    require(read(original_split) == read(ROOT/'SPLIT_STAGE2.json'), 'split semantic drift')
    decision_path=Path(owner_decision).resolve();decision=read(decision_path)
    policy_contract.validate_owner_decision(decision, [r['source_key'] for r in selected], binding['selected_index_sha256'])
    for item in decision['artifacts']:policy_contract.verify_policy_ref(decision_path.parent,item)
    output=EXPORTS/binding['selected_index_sha256'];output.mkdir(parents=True,exist_ok=True)
    def ref(path):
        path=Path(path);return dict(path=os.path.relpath(path,output),sha256=sha(path.read_bytes()))
    def new(name,obj):
        path=output/name;freeze(path,(json.dumps(obj,indent=2,sort_keys=True)+'\n').encode());return ref(path)
    # Preserve exact decision bytes and relative artifact references in a portable bundle.
    policy_dir=output/'owner_policy'
    for item in decision['artifacts']:
        p=policy_contract.verify_policy_ref(decision_path.parent,item)
        freeze(policy_dir/item['path'],p.read_bytes())
    decision_copy=policy_dir/'OWNER_EXECUTION_DECISION.json'
    freeze(decision_copy,decision_path.read_bytes())
    decision_ref=ref(decision_copy)
    exporter_copy=output/'policy_source/bridge_owner.py'
    freeze(exporter_copy,Path(__file__).read_bytes())
    exporter_ref=ref(exporter_copy)
    split_copy=output/'ORIGINAL_FROZEN_SPLIT.json'
    freeze(split_copy,original_split.read_bytes())
    original_split_ref=ref(split_copy)
    full=unique(rows(ROOT/'full/INDEX.jsonl'),'uid');causal=read(ROOT/'CAUSAL_MAP.json');flags=read(ROOT/'mech_flags_v1.json')
    routes=unique(read(ROOT/'route_audit_v1/audit_routes.json'),'uid');members=read(ROOT/'route_audit_v1/route_members.json')
    reverse={u:r for r,us in members.items() for u in us};exported=[]
    for row in selected:
        uid,key=row['uid'],row['source_key'];source=census[key]
        require(source['status']=='CANDIDATE' and source['native_success'] is True and source['screen']['verdict']=='ADMIT_CANDIDATE','candidate/native/screen gate')
        audit=audit_receipt(routes[reverse[uid]]);require(audit['verdict']['route_verdict']=='VALID','actual VALID required')
        samples=[selector_receipt(uid,n) for n in (1,2)]
        index=full[uid];eligible=read(ROOT/index['token_path'])
        actual,turns=mask_row(index,eligible,[s['parsed'] for s in samples],flags[uid],causal[uid])
        require(actual==read(ROOT/row['token_path']) and row['turns']==turns,'selected masks differ from fresh scientific replay')
        require(row['audit_receipt_sha256']==digest(audit) and row['selector_receipt_sha256']==[digest(s) for s in samples],'selected scientific binding drift')
        require(row['render_seal_sha256']==binding['render_seal_sha256'],'selected row missing current render seal')
        payload=new('payloads/'+key+'.json',dict(uid=uid,source_key=key,input_ids=actual['input_ids'],labels=actual['labels'],eligible_mask=eligible['loss_mask'],selected_mask=actual['loss_mask']))
        sd=ROOT/'sources'/key;requests=rows(sd/'conversation/llm_requests.jsonl')
        require(bool(requests) and all(r['tools']==requests[0]['tools'] for r in requests),'tools derivation equality')
        tools=new('tools/'+key+'.json',requests[0]['tools'])
        sources=dict(result=ref(sd/'result.json'),messages=ref(sd/'conversation/latest_messages.json'),requests=ref(sd/'conversation/llm_requests.jsonl'),transcript=ref(sd/'transcript.json'),tools=tools)
        # Complete receipts and their raw response/attempt source files remain in the mirrored admission tree.
        ar=ref(ROOT/'route_audit_v1/receipts'/(reverse[uid].replace(':','_')+'.json'))
        sr=[ref(ROOT/f'selector/final_v4_s{n}'/uid.replace(':','_')/'RECEIPT.json') for n in (1,2)]
        a=new('admissions/'+key+'.json',dict(status='FINAL_SELECTED',stage='TRAINING_DATA_VERIFIED',data_origin='actual_s3',source_key=key,
            payload_sha256=payload['sha256'],sources=sources,audit_sha256=ar['sha256'],route_verdict='VALID',selector_sha256=[r['sha256'] for r in sr],
            selected_mask_sha256=digest(actual['loss_mask']),native_success=True,screen_accepted=True,causal_prefixes_verified=True,
            mechanical_veto_applied=True,frozen_rule_verified=True,render_seal=ref(SEAL),
            tools_derivation=dict(requests_sha256=sources['requests']['sha256'],all_request_tools_equal=True),
            audit_journal_provenance=receipt_provenance(audit),selector_journal_provenance=[receipt_provenance(s) for s in samples]))
        exported.append(dict(uid=uid,source_key=key,target_id=row['idx'],data_origin='actual_s3',payload=payload,sources=sources,
                             receipts=dict(audit=ar,selector1=sr[0],selector2=sr[1],admission=a)))
    frozen={name:ref(ROOT/name) for name in read(ROOT/'PINNED_HASHES.json')}
    frozen.update(render_seal=ref(SEAL),render_index=ref(ROOT/'full/INDEX.jsonl'),source_lock=ref(ROOT/'LOCK.json'),
                  current_build=ref(reference(read(BASE/'CURRENT_BUILD.json'))),selected_index=ref(ROOT/'selected/INDEX.jsonl'),
                  admission_release=ref(reference({k:read(ROOT/'selected/RELEASE_CURRENT.json')[k] for k in ('path','sha256')})),
                  admission_native_report=ref(ROOT/'reader/NATIVE_READER_REPORT.json'))
    # Expose all referenced seal artifacts and all attempt bytes to the reader's hash verifier.
    for n,(rel,h) in enumerate(read(SEAL)['files'].items()):frozen['sealed_artifact_'+str(n)]=ref(BASE/rel)
    for n,p in enumerate(sorted({p for row in selected for d in [ROOT/f'selector/final_v4_s{s}'/row['uid'].replace(':','_')/'raw' for s in (1,2)] + [ROOT/'route_audit_v1/raw'/reverse[row['uid']].replace(':','_')] for p in d.rglob('*') if p.is_file()})):
        frozen['scientific_attempt_'+str(n)]=ref(p)
    frozen.update(owner_decision=decision_ref,owner_exporter=exporter_ref)
    gate=new('PARENT_SCIENTIFIC_GATE.json',dict(stage='SCIENTIFIC_SELECTED',training_ready=True,data_origin='actual_s3',rows_sha256=digest(exported),
          all_causal_prefixes_verified=True,rights_verified=False,execution_authorized=True,execution_policy=policy_contract.OWNER_POLICY,
          legal_clearance='UNVERIFIED',no_public_release=True,owner_decision=decision_ref,exporter_source=exporter_ref,render_seal=ref(SEAL),admission_native_verified=True,production_native_pass_pending=True))
    manifest=dict(schema='retrace.s3.final-selected.v3',mode='FINAL_SELECTED',stage='TRAINING_DATA_VERIFIED',data_origin='actual_s3',training_ready=True,
        admission_native_verified=True,production_native_pass_pending=True,predecessor_sha256='370edde7b44e0c4ef3cedcc35492306e1a95773a67fafd65f6ef4e6c53405e02',
        split=original_split_ref,frozen_sources=frozen,think_token_ids=full[selected[0]['uid']]['thinking_token_ids'],
        counts=dict(rows=len(selected),full_tokens=sum(r['n_tokens'] for r in selected),selected_tokens=sum(r['n_loss_tokens'] for r in selected)),rows=exported,parent_gate=gate)
    manifest_ref=new('FINAL_SELECTED_MANIFEST.json',manifest)
    contract=policy_contract
    contract.Release(output/manifest_ref['path'],manifest_ref['sha256'])
    print(json.dumps(dict(stage='TRAINING_DATA_VERIFIED',manifest=manifest_ref,path=str(output),admission_native_verified=True,production_native_pass_pending=True)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner-decision',required=True)
    export_runtime(parser.parse_args().owner_decision)
