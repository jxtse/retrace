"""Explicit NO_CLAIM offline fixtures. No model/optimizer/service is created."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from contract import write_new,sha,digest,PREDECESSOR_SHA,CONFIG_SHA
from evidence_gates import ref,verify_optimizer,verify_native
from aggregate_test40 import aggregate,classify_wire
from eval_s3 import IDS,launch_paths,checkpoint

class FixTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='NO_CLAIM-fix-',dir=Path(__file__).resolve().parents[1]/'evidence/fix'))
    def put(self,path,data):
        p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True);write_new(p,data);return p
    def change(self,p,fn):
        r=json.loads(p.read_text());fn(r);p.write_text(json.dumps(r));return r
    def optimizer(self):
        source=self.root/'source.py';source.write_text('# NO_CLAIM fixture\n')
        pins=self.put('pins.json',{'files':[ref(source)]})
        run=self.root/'run';run.mkdir();ck=run/'checkpoints/checkpoint-3';ck.mkdir(parents=True)
        (run/'checkpoints/checkpoint-1').mkdir();(run/'checkpoints/last-checkpoint').symlink_to(ck)
        # max_steps=3 is explicit runtime metadata, deliberately not ceil(2 UIDs/4).
        launch=self.put('run/launch.json',dict(stage='full',pins_sha256=sha(pins),predecessor_sha256=PREDECESSOR_SHA,
            fresh_optimizer=True,argv=['swift','sft','--adapters','/NO_CLAIM_S2'],manifest_sha256='fixture',selected_uids=['a','b']))
        exitp=self.put('run/exit.json',{'returncode':0})
        state=self.put('run/checkpoints/checkpoint-3/trainer_state.json',dict(global_step=3,max_steps=3,num_train_epochs=1,epoch=1.0,
            log_history=[dict(step=i,loss=.1,grad_norm=.2) for i in range(1,4)]))
        adapter=ck/'adapter_model.safetensors';adapter.write_text('NO_CLAIM text, not model weights')
        config=self.put('run/checkpoints/checkpoint-3/adapter_config.json',{'r':8,'lora_alpha':16})
        trace=self.put('raw.json',{'fixture':'NOT_REAL_EVIDENCE'})
        runtime=self.put('runtime.json',dict(launch_sha256=sha(launch),origin='actual_trainer_runtime',max_steps=3,
            num_train_epochs=1,steps_in_epoch=5,epoch_based=True,trainer_source=ref(source)))
        load=self.put('loaded.json',dict(launch_sha256=sha(launch),loaded_path='/NO_CLAIM_S2',adapter_sha256=PREDECESSOR_SHA,
            adapter_config_sha256=CONFIG_SHA,origin='actual_load_trace',is_trainable=True,raw_trace=ref(trace)))
        reload=self.put('reload.json',dict(launch_sha256=sha(launch),adapter_sha256=sha(adapter),adapter_config_sha256=sha(config),
            origin='actual_save_reload',tensor_parity=True,raw_evidence=ref(trace)))
        exposure={}
        for rank in range(4):
            p=self.root/f'exposure-{rank}.jsonl';rows=[]
            for step,n in [(1,2),(2,2),(3,1)]:rows.append(dict(step=step,rank=rank,launch_sha256=sha(launch),
                origin='actual_training_step_commit',optimizer_step_skipped=False,uids=['a','b'],microbatches=n))
            p.write_text(''.join(json.dumps(r)+'\n' for r in rows));exposure[str(rank)]=ref(p)
        g=self.put('gate.json',dict(schema='s3.optimizer-evidence.v4',launch=ref(launch),source_pins_sha256=sha(pins),
            recipe_sha256=digest(json.loads(launch.read_text())['argv']),exit=ref(exitp),checkpoint=str(ck),trainer_state=ref(state),
            last_checkpoint_target=str(ck),adapter=ref(adapter),adapter_config=ref(config),trainer_runtime=ref(runtime),
            loaded_adapter=ref(load),save_reload=ref(reload),optimizer_exposure=exposure))
        return g,pins,ck
    def test_optimizer_exact_runtime_metadata(self):
        g,pins,ck=self.optimizer();self.assertEqual(verify_optimizer(g,pins,'fixture','full')['checkpoint'],ck)
    def test_reject_earlier_retained_checkpoint(self):
        g,pins,ck=self.optimizer();self.change(g,lambda r:r.update(checkpoint=str(ck.parent/'checkpoint-1')))
        with self.assertRaises(ValueError):verify_optimizer(g,pins)
    def test_reject_last_checkpoint_mismatch(self):
        g,pins,ck=self.optimizer();(ck.parent/'last-checkpoint').unlink();(ck.parent/'last-checkpoint').symlink_to(ck.parent/'checkpoint-1')
        with self.assertRaisesRegex(ValueError,'last-checkpoint'):verify_optimizer(g,pins)
    def test_reject_source_swap(self):
        g,pins,ck=self.optimizer();(self.root/'source.py').write_text('# changed')
        with self.assertRaises(ValueError):verify_optimizer(g,pins)
    def test_reject_missing_raw_evidence(self):
        g,pins,ck=self.optimizer();(self.root/'raw.json').unlink()
        with self.assertRaises(ValueError):verify_optimizer(g,pins)
    def test_reject_reader_only_exposure(self):
        g,pins,ck=self.optimizer();p=self.root/'exposure-0.jsonl';p.write_text(p.read_text().replace('actual_training_step_commit','reader_visit'))
        self.change(g,lambda r:r['optimizer_exposure'].update({'0':ref(p)}))
        with self.assertRaisesRegex(ValueError,'reader-only'):verify_optimizer(g,pins)
    def test_reject_swapped_smoke_manifest_or_stage(self):
        g,pins,ck=self.optimizer()
        with self.assertRaisesRegex(ValueError,'manifest'):verify_optimizer(g,pins,'other')
        with self.assertRaisesRegex(ValueError,'stage'):verify_optimizer(g,pins,stage='smoke')
    def test_reject_nonfinite_and_incomplete_epoch(self):
        g,pins,ck=self.optimizer();state=ck/'trainer_state.json';self.change(state,lambda r:r.update(epoch=.5))
        self.change(g,lambda r:r.update(trainer_state=ref(state)))
        with self.assertRaisesRegex(ValueError,'epoch'):verify_optimizer(g,pins)
        self.change(state,lambda r:r.update(epoch=1.0,log_history=[dict(step=i,loss=.1,grad_norm=0) for i in range(1,4)]))
        self.change(g,lambda r:r.update(trainer_state=ref(state)))
        with self.assertRaisesRegex(ValueError,'gradient'):verify_optimizer(g,pins)
    def test_export_bound_to_validated_terminal_bytes(self):
        from export_adapter import export
        g,pins,ck=self.optimizer();out=self.root/'export'
        export(ck.parent.parent,ck,g,out,pins)
        checkpoint(out,sha(out/'adapter_model.safetensors'),out/'TRAINED.json')
        (out/'adapter_model.safetensors').write_text('NO_CLAIM swapped text, not weights')
        self.change(out/'TRAINED.json',lambda r:r.update(adapter_sha256=sha(out/'adapter_model.safetensors')))
        with self.assertRaisesRegex(ValueError,'validated terminal'):checkpoint(out,sha(out/'adapter_model.safetensors'),out/'TRAINED.json')
    def test_no_model_child_changed_cwd(self):
        prep=self.root/'prepared';arm=prep/'h0';cwd=arm/'native-isolated/harness';cwd.mkdir(parents=True)
        (arm/'eval_config.yaml').write_text('NO_MODEL_CONFIG')
        (arm/'detached_runner.py').write_text('import os,json; from pathlib import Path; print(json.dumps({"cwd":os.getcwd(),"config":Path(os.environ["AGENT_CONFIG"]).read_text()}))')
        paths=launch_paths(os.path.relpath(prep), 'h0')
        result=subprocess.run([sys.executable,'-B',str(paths['runner'])],cwd=paths['cwd'],env=dict(os.environ,AGENT_CONFIG=str(paths['config'])),capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(result.stdout),{'cwd':str(cwd),'config':'NO_MODEL_CONFIG'})
    def test_malformed_wire_shapes(self):
        for x in ([],None,1,{'request':'x','response':{}},{'request':{},'response':{'choices':[]}},
                  {'request':{},'response':{'choices':[{'message':'x'}]}}):self.assertTrue(classify_wire(x,'s3')['infra'])
    def test_aggregate_malformed_results_and_elapsed(self):
        session=self.root/'session';(session/'conversation').mkdir(parents=True)
        (session/'conversation/wire_log.jsonl').write_text('[]\n{"request":"x"}\nnot-json\n')
        self.put('runs/010.json',dict(idx=10,success=True,session_dir=str(session),wall_seconds=5246.2))
        self.put('runs/013.json',dict(idx=13,session_dir=str(session)))
        (self.root/'runs/024.json').write_text('[]')
        (self.root/'runs/033.json').write_bytes(b'\xff')
        r=aggregate([self.root],mode='historical');by={c['idx']:c for c in r['cases']}
        self.assertEqual(by[10]['elapsed_s'],5246.2);self.assertEqual(by[10]['status'],'infra_invalid')
        self.assertIsNone(by[13]['native_success']);self.assertEqual(by[24]['status'],'infra_invalid')
        self.assertEqual(by[33]['status'],'infra_invalid')
        self.assertFalse(r['evaluated']);self.assertEqual(r['denominator'],40)
    def test_final_missing_joins_cannot_evaluate(self):
        r=aggregate([self.root,self.root/'second'],mode='final',terminals=[self.root/'x',self.root/'y'])
        self.assertFalse(r['evaluated']);self.assertEqual(len(r['join_errors']),2)
    def test_native_old_report_rejected(self):
        from types import SimpleNamespace
        p=self.put('old-native.json',dict(schema='old',passed=True))
        with self.assertRaises(ValueError):verify_native(p,SimpleNamespace(),'missing')
    def test_native_cases_and_index_binding(self):
        from types import SimpleNamespace
        payload=dict(input_ids=[1,2],selected_mask=[0,1]);release=SimpleNamespace(sha256='manifest',rows={'u':{'source_key':'001_rep1','payload':{'sha256':'a'*64}}},payload=lambda u:payload)
        from evidence_gates import canonical_index
        index=canonical_index(release);ix=self.put('INDEX.json',index);source=self.root/'source';source.write_text('fixture')
        pins=self.put('pins.json',{'files':[ref(source)]});cases=self.root/'cases.jsonl'
        cases.write_text(''.join(json.dumps(dict(uid='u',mode=m,status='PASS',index_sha256=digest(index)))+'\n' for m in ('selected','zero_control','partial_control')))
        detailed=self.put('detailed.json',[dict(case='production:'+m+':u',exact=True) for m in ('selected','zero_control','partial_control')])
        names=['get_cached_dataset','sft_postprocessor','native_lazy_reader','template','text_rope_helper','adapter','acceptance_runner','sequence_parallel']
        rt=dict(executable=str(Path(sys.executable).resolve()),executable_sha256=sha(Path(sys.executable).resolve()),python='fixture',platform='fixture',hostname='fixture',ms_swift_version='fixture',packages={k:'fixture' for k in ('ms_swift','torch','transformers','datasets')})
        r=dict(schema='s3.native-evidence.v4',data_origin='actual_native_run',mode='production',actual_manifest_sha256='manifest',
            actual_final_rows=1,source_pins_sha256=sha(pins),source_pins=ref(pins),canonical_data_index=ref(ix),selected_ids=['u'],
            runtime_identity=rt,native_sources={n:ref(source) for n in names},case_artifact=ref(cases),control_case_artifact=ref(detailed),cases=3)
        r.update({k:True for k in ('passed','exact_full_tokens_and_masks','reasoning_loss_zero','causal_shift_exact','collator_exact','padding_exact','zero_and_partial_controls_passed','no_model_constructed','no_optimizer')})
        p=self.put('native.json',r)
        with patch('evidence_gates.importlib.metadata.version',return_value='fixture'):
            verify_native(p,release,pins)
            original=detailed.read_bytes();detailed.write_text('[]')
            with self.assertRaises(ValueError):verify_native(p,release,pins)
            detailed.write_bytes(original)
            cases.write_text(cases.read_text().replace('PASS','FAIL'))
            with self.assertRaises(ValueError):verify_native(p,release,pins)
        self.change(p,lambda r:r.update(case_artifact=ref(cases)))
        with patch('evidence_gates.importlib.metadata.version',return_value='fixture'):
            with self.assertRaisesRegex(ValueError,'stale'):verify_native(p,release,pins)

if __name__=='__main__':unittest.main(verbosity=2)
