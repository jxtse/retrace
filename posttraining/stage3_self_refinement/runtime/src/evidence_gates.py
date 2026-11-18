"""Hash-bound execution evidence checks; pure stdlib, no model construction."""
import importlib.metadata
import math
from pathlib import Path
from contract import read_json, sha, digest, require, verify_ref, PREDECESSOR_SHA, CONFIG_SHA
ROOT = Path(__file__).resolve().parents[1]


def ref(path):
    p=Path(path).resolve()
    return {'path':str(p),'sha256':sha(p)}


def checked_json(root, item):
    return read_json(verify_ref(root,item))


def json_lines(path):
    return [__import__('json').loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def code_pins(path):
    p=Path(path).resolve(); data=read_json(p)
    require(data['files'], 'empty source pins')
    for r in data['files']: verify_ref(ROOT,r)
    return sha(p)


def canonical_index(release):
    return [dict(uid=u,source_key=r['source_key'],payload_sha256=r['payload']['sha256'],
                 full_tokens=len(release.payload(u)['input_ids']),selected_tokens=sum(release.payload(u)['selected_mask']))
            for u,r in release.rows.items()]


def verify_native(path, release, pins):
    p=Path(path).resolve(); r=read_json(p)
    require(r['schema']=='s3.native-evidence.v4' and r['data_origin']=='actual_native_run' and r['mode']=='production', 'not production native evidence')
    require(r['actual_manifest_sha256']==release.sha256 and r['actual_final_rows']==len(release.rows), 'native manifest/row binding')
    require(r['source_pins_sha256']==code_pins(pins), 'native source pins mismatch')
    require(sha(verify_ref(p.parent,r['source_pins']))==r['source_pins_sha256'], 'native pins reference mismatch')
    index=checked_json(p.parent,r['canonical_data_index'])
    require(index==canonical_index(release) and r['selected_ids']==list(release.rows), 'native canonical data index mismatch')
    rt=r['runtime_identity']
    require(set(rt['packages']) >= {'ms_swift','torch','transformers','datasets'},'native package identity missing')
    require(all(isinstance(rt.get(k),str) and rt[k] for k in ('python','platform','hostname','ms_swift_version')),'native runtime identity missing')
    require(sha(rt['executable'])==rt['executable_sha256'], 'native interpreter changed')
    for name,version in rt['packages'].items():
        require(importlib.metadata.version(name)==version, 'native runtime package drift: '+name)
    for item in r['native_sources'].values(): verify_ref(p.parent,item)
    require(set(r['native_sources']) >= {'get_cached_dataset','sft_postprocessor','native_lazy_reader','template','text_rope_helper','adapter','acceptance_runner','sequence_parallel'}, 'native source coverage')
    details=checked_json(p.parent,r['control_case_artifact'])
    require(isinstance(details,list) and len(details)==r['cases'] and details and
            all(isinstance(c,dict) and c.get('exact') is True for c in details),'native detailed cases invalid')
    require(len({c['case'] for c in details})==len(details),'native duplicate detailed cases')
    detail_names={c['case'] for c in details}
    for u in release.rows:
        require(all('production:'+m+':'+u in detail_names for m in ('selected','zero_control','partial_control')),
                'native detailed row checks missing')
    require(all('production:padded_pair:'+u in detail_names for u in list(release.rows)[1:]),'native detailed pair checks missing')
    cases=json_lines(verify_ref(p.parent,r['case_artifact']))
    for mode in ('selected','zero_control','partial_control'):
        subset=[c for c in cases if c.get('mode')==mode]
        require(len(subset)==len(index) and [c['uid'] for c in subset]==list(release.rows), 'native case row coverage')
        require(all(c['status']=='PASS' and c['index_sha256']==digest(index) for c in subset), 'native stale cases')
    pairs=[c for c in cases if c.get('mode')=='padded_pair']
    require([c['uids'] for c in pairs]==[list(release.rows)[i-1:i+1] for i in range(1,len(index))], 'native pair coverage')
    require(all(c['status']=='PASS' and c['index_sha256']==digest(index) for c in pairs),'native stale pairs')
    for key in ('passed','exact_full_tokens_and_masks','reasoning_loss_zero','causal_shift_exact','collator_exact',
                'padding_exact','zero_and_partial_controls_passed','no_model_constructed','no_optimizer'):
        require(r[key] is True,'native check absent: '+key)
    return r


def verify_optimizer(path, pins, manifest_sha=None, stage=None):
    """Replay raw owner evidence. No Boolean-only receipt can pass this gate."""
    p=Path(path).resolve(); g=read_json(p)
    require(g['schema']=='s3.optimizer-evidence.v4', 'optimizer evidence schema')
    lp=verify_ref(p.parent,g['launch']); launch=read_json(lp); run=lp.parent
    require(launch['pins_sha256']==code_pins(pins)==g['source_pins_sha256'], 'optimizer source/recipe changed')
    require(g['recipe_sha256']==digest(launch['argv']), 'optimizer recipe mismatch')
    require(launch['predecessor_sha256']==PREDECESSOR_SHA and launch['fresh_optimizer'] is True, 'optimizer lineage')
    if manifest_sha: require(launch['manifest_sha256']==manifest_sha, 'optimizer manifest swapped')
    if stage: require(launch['stage']==stage, 'wrong optimizer stage')
    require(checked_json(p.parent,g['exit'])['returncode']==0 and verify_ref(p.parent,g['exit'])==run/'exit.json','nonterminal run')
    ck=Path(g['checkpoint']).resolve()
    require(ck.is_relative_to((run/'checkpoints').resolve()), 'checkpoint outside run')
    state_path=verify_ref(p.parent,g['trainer_state'])
    require(state_path==ck/'trainer_state.json','trainer state/checkpoint mismatch')
    state=read_json(state_path); step=state['global_step']
    require(type(step) is int and step>0, 'invalid terminal step')
    checkpoints=[d for d in (run/'checkpoints').glob('checkpoint-*') if d.is_dir() and d.name[11:].isdigit()]
    require(checkpoints and ck==max(checkpoints,key=lambda d:int(d.name[11:])), 'earlier retained checkpoint forbidden')
    require(int(ck.name[11:])==step,'checkpoint step/name mismatch')
    last=run/'checkpoints/last-checkpoint'
    require(last.is_symlink() and last.resolve()==ck,'last-checkpoint does not bind terminal checkpoint')
    require(g['last_checkpoint_target']==str(ck),'last checkpoint evidence mismatch')
    for field,name in [('adapter','adapter_model.safetensors'),('adapter_config','adapter_config.json')]:
        require(verify_ref(p.parent,g[field])==ck/name,'checkpoint artifact mismatch')
    config=read_json(ck/'adapter_config.json');require(config['r']==8 and config['lora_alpha']==16,'adapter config mismatch')
    schedule=checked_json(p.parent,g['trainer_runtime'])
    require(schedule['launch_sha256']==sha(lp) and schedule['origin']=='actual_trainer_runtime','scheduler provenance')
    require(schedule['max_steps']==state['max_steps']==step and schedule['num_train_epochs']==state['num_train_epochs'],'terminal scheduler mismatch')
    require(type(schedule['steps_in_epoch']) is int and schedule['steps_in_epoch']>0,'missing actual dataloader schedule')
    verify_ref(p.parent,schedule['trainer_source'])
    if launch['stage']=='full':require(state['epoch']>=1.0 and schedule['epoch_based'] is True,'full epoch incomplete')
    else:require(launch['stage']=='smoke' and step==2,'two actual smoke updates required')
    loaded=checked_json(p.parent,g['loaded_adapter'])
    require(loaded['launch_sha256']==sha(lp) and loaded['loaded_path']==launch['argv'][launch['argv'].index('--adapters')+1], 'loaded adapter path mismatch')
    require(loaded['adapter_sha256']==PREDECESSOR_SHA and loaded['adapter_config_sha256']==CONFIG_SHA,'loaded S2 identity')
    require(loaded['origin']=='actual_load_trace' and loaded['is_trainable'] is True,'loaded adapter proof')
    verify_ref(p.parent,loaded['raw_trace'])
    reloaded=checked_json(p.parent,g['save_reload'])
    require(reloaded['launch_sha256']==sha(lp) and reloaded['adapter_sha256']==sha(ck/'adapter_model.safetensors') and
            reloaded['adapter_config_sha256']==sha(ck/'adapter_config.json'),'save/reload swapped checkpoint')
    require(reloaded['origin']=='actual_save_reload' and reloaded['tensor_parity'] is True,'save reload absent')
    verify_ref(p.parent,reloaded['raw_evidence'])
    # Real Trainer log_history contains pre-clip grad_norm and loss at every logging step.
    logs={r['step']:r for r in state['log_history'] if 'loss' in r and 'grad_norm' in r}
    require(set(logs)==set(range(1,step+1)),'missing raw finite update/gradient logs')
    for r in logs.values():
        require(all(type(r[k]) in (int,float) and math.isfinite(r[k]) for k in ('loss','grad_norm')) and r['grad_norm']>0,'nonfinite/zero gradient update')
    exposure=g['optimizer_exposure'];require(set(exposure)=={'0','1','2','3'},'per-rank optimizer exposure missing')
    rank_records=[]
    for rank,item in sorted(exposure.items()):
        records=json_lines(verify_ref(p.parent,item)); require(records,'empty optimizer exposure')
        require([r['step'] for r in records]==list(range(1,step+1)),'optimizer step exposure gap')
        for r in records:
            require(r['rank']==int(rank) and r['launch_sha256']==sha(lp) and r['origin']=='actual_training_step_commit' and r['optimizer_step_skipped'] is False,'uncommitted/reader-only exposure')
            require(r['uids'] and set(r['uids'])<=set(launch['selected_uids']),'unknown consumed UID')
            require(type(r['microbatches']) is int and r['microbatches']>0,'missing consumed microbatches')
        require(set(u for r in records for u in r['uids'])==set(launch['selected_uids']),'optimizer UID coverage')
        if launch['stage']=='full':require(sum(r['microbatches'] for r in records)==schedule['steps_in_epoch'],'epoch microbatch exposure incomplete')
        rank_records.append([(r['step'],r['uids'],r['microbatches']) for r in records])
    require(all(r==rank_records[0] for r in rank_records),'SP rank exposure disagrees')
    return dict(gate=g,launch=launch,run=run,checkpoint=ck,state=state,gate_sha256=sha(p),launch_path=lp)
