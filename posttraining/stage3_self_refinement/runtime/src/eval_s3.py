"""Frozen test40 preparation and gated owner launcher. No chemistry audit."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
from contract import require, read_json, sha, write_new, verify_ref, SPLIT_SHA, PREDECESSOR_SHA
from train_s3 import verify_pins
from evidence_gates import ref, verify_optimizer

# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
ROOT=Path(__file__).resolve().parents[1]
BASE=Path(RETRACE_SHARE + '')
SOURCE=BASE/'retrace-continual'
HIST=SOURCE/'rollouts/eval_round1_test40/base'
SPLIT=BASE/'stage2-agentic/split/SPLIT_STAGE2.json'
ALIAS='qwen3.8-27b-s3'
IDS=[10,13,24,33,34,37,52,53,55,56,58,60,66,80,87,89,90,94,99,102,104,105,108,110,113,115,121,124,125,132,136,147,154,156,158,171,179,181,182,189]
EXCLUDES=['/runs/','/.sessions/','/.cache/','/tools/.cache/','/.fix_backups/','/.venv*/','/.git/',
 '/.agents','/.codex','/.claude/','/.pytest_cache/','__pycache__/','*.pyc',
 '/dataset/Enamine_Building_Blocks_Stock_573928cmpd_20260210.sdf','/dataset/USPTO_full',
 '/dataset/benchmark/SAIS/score_smiles_candidates.csv']
ASSETS=['harness/vendor/legacy_retro_planner/building_block_dataset',
        'harness/dataset/enamine_index.pkl']


def shards(ids):
    require(len(ids)==40 and len(set(ids))==40 and sorted(ids)==IDS,'frozen test40 mismatch')
    result={f'h{i}':sorted(ids)[i::2] for i in (0,1)}
    require(not set(result['h0']) & set(result['h1']) and sorted(result['h0']+result['h1'])==IDS,'bad shards')
    return result


def verify_tree(root,frozen):
    require(frozen, 'empty frozen source map')
    for rel,digest in frozen.items():
        require(sha(Path(root)/rel)==digest,'frozen source mismatch: '+rel)


def config_for(base,half):
    cfg=copy.deepcopy(base)
    cfg['llm'].update(model=ALIAS,base_url=f'http://127.0.0.1:{8000 if half=="h0" else 8002}/v1',reasoning_effort='medium')
    left,right=copy.deepcopy(base),copy.deepcopy(cfg)
    for k in ('model','base_url','reasoning_effort'):
        left['llm'].pop(k);right['llm'].pop(k)
    require(left==right,'unpermitted protocol drift')
    require(cfg['llm']['temperature']==0.7 and cfg['llm']['max_tokens']==32768 and cfg['llm']['fixed_output_budget'] is True,
            'decoding mismatch')
    require(cfg['llm']['context_window']==131072 and cfg['agent']['max_iterations']==40,'context/iteration mismatch')
    return cfg


def prepare(out,envs,pins):
    import yaml
    envs=Path(envs).resolve()
    out=Path(out).resolve(); require(out.is_relative_to(ROOT) and not out.exists(),'new runtime output required')
    verify_pins(pins)
    require(sha(SPLIT)==SPLIT_SHA,'split changed')
    split=shards(read_json(SPLIT)['test40_eval_only'])
    hist=read_json(HIST/'launch_manifest.json')
    require(sorted(hist['indices'])==IDS,'historical split changed')
    frozen=hist['source_sha256'];verify_tree(SOURCE,frozen)
    out.mkdir(parents=True)
    for half,ids in split.items():
        dest=out/half/'native-isolated';dest.mkdir(parents=True)
        for name in ('harness','chemoenzymatic-retrosynthesis-verifier'):
            extra=['/'+p.removeprefix('harness/') for p in ASSETS] if name=='harness' else []
            (dest/name).mkdir()
            subprocess.run(['rsync','-aL']+['--exclude='+x for x in EXCLUDES+extra]+[str(SOURCE/name)+'/',str(dest/name)+'/'],check=True)
        assets={}
        for rel in ASSETS:
            source=SOURCE/rel;target=dest/rel
            require(source.exists() and not target.exists(),'asset missing or unexpectedly copied: '+rel)
            target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(source,target_is_directory=source.is_dir())
            for f in sorted(source.rglob('*')) if source.is_dir() else [source]:
                if f.is_file(): assets[str(f)] = sha(f)
        verify_tree(dest,frozen)
        mutable_asset='harness/dataset/buyability/source_mols.db'
        copied_sha=sha(dest/mutable_asset)
        require(copied_sha==sha(SOURCE/mutable_asset),'copied stock DB differs')
        for rel in ('harness/runs','harness/.sessions','harness/.cache','harness/tools/.cache'):
            require(not (dest/rel).exists(),'old mutable state copied')
        cfg=config_for(yaml.safe_load((HIST/'eval_config.yaml').read_text()),half)
        (out/half/'eval_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        shutil.copyfile(HIST/'detached_runner.py',out/half/'detached_runner.py')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',XDG_CACHE_HOME=str(out/'cache'),
                 AGENT_CONFIG=str(out/half/'eval_config.yaml'))
        command=[str(Path(envs)/'eval/bin/python'),'-B','dataset/retro-190_dataset/run_benchmark.py','--help']
        result=subprocess.run(command,cwd=dest/'harness',env=env,text=True,capture_output=True)
        (out/half/'entrypoint-help.log').write_text(result.stdout+result.stderr)
        require(result.returncode==0,'native eval help failed')
        # Load only configuration; no agent, tools, model or service initialization.
        probe="from core.config import load_config; import json,os; c=load_config(os.environ['AGENT_CONFIG']); assert c['agent']['wallclock_s']==14400 and c['agent']['max_iterations']==40 and c['llm']['reasoning_effort']=='medium'; print(json.dumps(c.get('agent',{})))"
        result=subprocess.run([str(Path(envs)/'eval/bin/python'),'-B','-c',probe],cwd=dest/'harness',env=env,text=True,capture_output=True)
        (out/half/'config-probe.log').write_text(result.stdout+result.stderr)
        require(result.returncode==0,'native config load failed')
        write_new(out/half/'ISOLATION.json',dict(indices=ids,source_sha256=frozen,asset_sha256=assets,
            copied_assets={mutable_asset:copied_sha},
            config_sha256=sha(out/half/'eval_config.yaml'),runner_sha256=sha(out/half/'detached_runner.py'),verified=True,
            native_help_returncode=0,mutable_state_copied=False))
    write_new(out/'PREPARED.json',dict(test40=IDS,shards=split,envs=str(Path(envs).resolve()),
        source_pins_sha256=sha(pins),protocol={'workers_per_half':4,'max_iterations':40,'wallclock_s':14400,
        'reasoning_effort':'medium','temperature':0.7,'max_tokens':32768},scientific_delta=['S3 adapter/alias','new paths/endpoints']))


def checkpoint(path,digest,trained_receipt):
    path=Path(path).resolve()
    require(path.is_dir() and (path/'adapter_model.safetensors').is_file(),'missing checkpoint')
    require(sha(path/'adapter_model.safetensors')==digest and digest!=PREDECESSOR_SHA,'S3 checkpoint hash mismatch')
    c=read_json(path/'adapter_config.json');require(c['r']==8 and c['lora_alpha']==16,'checkpoint LoRA mismatch')
    r=read_json(trained_receipt)
    require(r['schema']=='s3.trained.v4' and r['adapter_config_sha256']==sha(path/'adapter_config.json'),'trained configuration binding')
    evidence=verify_optimizer(verify_ref(Path(trained_receipt).resolve().parent,r['validation']),
                     verify_ref(Path(trained_receipt).resolve().parent,r['source_pins']),stage='full')
    terminal=evidence['checkpoint']
    require(sha(terminal/'adapter_model.safetensors')==digest and
            sha(terminal/'adapter_config.json')==r['adapter_config_sha256'],'export differs from validated terminal checkpoint')
    require(r['manifest_sha256']==evidence['launch']['manifest_sha256'] and
            r['optimizer_steps']==evidence['state']['global_step'] and
            r['terminal_epoch']==evidence['state']['epoch'],'trained receipt differs from terminal run')
    require(verify_ref(Path(trained_receipt).resolve().parent,r['terminal_trainer_state'])==terminal/'trainer_state.json',
            'trained terminal state mismatch')
    require(r['stage']=='full' and r['trained'] is True and r['adapter_sha256']==digest,'full training receipt missing')
    require(r['predecessor_sha256']==PREDECESSOR_SHA and r['fresh_optimizer'] is True,'training lineage mismatch')
    return path


def verify_serving(proof,checkpoint_sha,half):
    p=Path(proof);r=read_json(p)
    require(r['adapter_sha256']==checkpoint_sha and r['alias']==ALIAS and r['port']==(8000 if half=='h0' else 8002),'serving identity mismatch')
    require(r['load_verified'] is True,'serving load proof missing')
    wire=read_json(verify_ref(p.parent,r['wire_probe']))
    require(wire['request']['model']==ALIAS and wire['response']['model']==ALIAS and wire['status_code']==200,'requested/returned adapter mismatch')
    require(wire['response']['choices'],'empty serving probe')
    verify_ref(p.parent,r['load_evidence'])


def launch_paths(prepared, half):
    prepared=Path(prepared).resolve();arm=(prepared/half).resolve()
    return dict(prepared=prepared,arm=arm,runner=(arm/'detached_runner.py').resolve(),
                config=(arm/'eval_config.yaml').resolve(),cwd=(arm/'native-isolated/harness').resolve())


def launch_plan(a):
    import yaml
    ck=checkpoint(a.checkpoint,a.checkpoint_sha256,a.trained_receipt)
    verify_pins(a.pins)
    paths=launch_paths(a.prepared,a.half);prepared=paths['prepared'];rec=read_json(prepared/'PREPARED.json');half=a.half
    require(rec['source_pins_sha256']==sha(a.pins),'preparation source pins changed')
    shard=shards(rec['test40'])[half]
    arm=paths['arm'];isolation=read_json(arm/'ISOLATION.json');harness=paths['cwd']
    verify_tree(arm/'native-isolated',isolation['source_sha256'])
    for path,d in isolation['asset_sha256'].items(): require(sha(path)==d,'asset changed')
    for rel,d in isolation['copied_assets'].items():
        require(sha(arm/'native-isolated'/rel)==d,'copied stock DB changed before launch')
    require(sha(arm/'eval_config.yaml')==isolation['config_sha256'] and sha(arm/'detached_runner.py')==isolation['runner_sha256'],'config/runner changed')
    require(yaml.safe_load((arm/'eval_config.yaml').read_text())==config_for(yaml.safe_load((HIST/'eval_config.yaml').read_text()),half),'config drift')
    require(isolation['indices']==shard,'shard drift')
    out=Path(a.out).resolve(); require(out.is_relative_to(ROOT) and not out.exists(),'new runtime output required')
    envs=Path(rec['envs']).resolve();port=8000 if half=='h0' else 8002
    serve=[str(envs/'serving/bin/python'),str(SOURCE/'venv-vllm/bin/vllm'),'serve',str(BASE/'models/Qwen/Qwen3.8-27B'),
        '--served-model-name','qwen3.8-27b-student','--host','127.0.0.1','--port',str(port),'--dtype','bfloat16',
        '--max-model-len','131072','--gpu-memory-utilization','0.90','--max-num-seqs','8','--enable-auto-tool-choice',
        '--tool-call-parser','qwen3_xml','--reasoning-parser','qwen3','--language-model-only','--generation-config','auto',
        '--tensor-parallel-size','2','--enable-lora','--max-lora-rank','32','--max-loras','1','--lora-modules',ALIAS+'='+str(ck)]
    py=str(envs/'eval/bin/python')
    cmd=[py,str(paths['runner']),'--out',str(out),'--',py,'dataset/retro-190_dataset/run_benchmark.py',
         '--only',','.join(map(str,shard)),'--workers','4','--output-dir',str(out)]
    env=dict(CUDA_VISIBLE_DEVICES='',AGENT_CONFIG=str(paths['config']),AGENT_MODEL=ALIAS,AGENT_THINKING='fast',PYTHONDONTWRITEBYTECODE='1',
        RETROPLANNER_HTTP_URL='http://localhost:8010/retroplanner/api/single_step',PYTHONUNBUFFERED='1',
        XDG_CACHE_HOME=str(out/'cache'),MPLCONFIGDIR=str(out/'mpl'),NUMBA_CACHE_DIR=str(out/'numba'))
    plan=dict(schema='s3.eval-launch.v4',half=half,prepared=ref(prepared/'PREPARED.json'),
              isolation=ref(arm/'ISOLATION.json'),config=ref(arm/'eval_config.yaml'),
              runner=ref(arm/'detached_runner.py'),source_pins=ref(a.pins),trained_receipt=ref(a.trained_receipt),
              checkpoint=str(ck),session_root=str((harness/'runs').resolve()),indices=shard,full_test40=IDS,adapter_sha256=a.checkpoint_sha256,served_model=ALIAS,argv=cmd,
              env=env,cwd=str(harness),serving_argv=serve,serving_cuda='2,3' if half=='h0' else '0,1')
    if not a.execute: print(json.dumps(plan,indent=2));return
    verify_serving(a.serving_proof,a.checkpoint_sha256,half)
    require(sha(a.tool_preflight)=='c8dfa51c4db7726dc79bb861341cb9d9bf0ae55586ff99258b0b4a84c0ba9ea2','use preserved genuine preflight')
    tool=read_json(a.tool_preflight);require(tool['native_query_passed'] is True,'tool relay preflight missing')
    require(tool['url']==env['RETROPLANNER_HTTP_URL'],'tool endpoint mismatch')
    plan.update(serving_proof=ref(a.serving_proof),tool_preflight=ref(a.tool_preflight))
    out.mkdir(parents=True,exist_ok=False);write_new(out/'launch_manifest.json',plan)
    with (out/'driver.log').open('xb') as log:
        proc=subprocess.Popen(cmd,cwd=harness,env=dict(os.environ,**env),stdin=subprocess.DEVNULL,
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_new(out/'launch_receipt.json',{'wrapper_pid':proc.pid,'argv':cmd,'launch_manifest_sha256':sha(out/'launch_manifest.json')})


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','plan'])
    p.add_argument('--out',required=True);p.add_argument('--envs');p.add_argument('--pins',default=str(ROOT/'SOURCE_PINS.fix-v4-final2.json'))
    p.add_argument('--prepared');p.add_argument('--half',choices=['h0','h1']);p.add_argument('--checkpoint')
    p.add_argument('--checkpoint-sha256');p.add_argument('--trained-receipt');p.add_argument('--serving-proof')
    p.add_argument('--tool-preflight');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare(a.out,a.envs,a.pins)
    else:launch_plan(a)
if __name__=='__main__':main()
