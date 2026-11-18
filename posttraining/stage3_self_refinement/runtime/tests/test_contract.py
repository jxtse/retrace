import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from contract import Release,validate_tokens,digest,sha,verify_ref,write_new
from eval_s3 import shards,IDS,checkpoint,verify_tree,config_for
from aggregate_test40 import classify_wire,aggregate

class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=Path(tempfile.mkdtemp(prefix='unit-',dir=Path(__file__).resolve().parents[1]/'evidence'))
        self.row=dict(input_ids=[10,11,12,13,14,15,16],eligible_mask=[0,0,0,0,1,1,0],
                      selected_mask=[0,0,0,0,1,0,0],labels=[-100,-100,-100,-100,14,-100,-100])
    def test_zero_partial_reasoning_suffix(self):
        self.assertEqual(validate_tokens(self.row,[11,13],True),1)
        z=copy.deepcopy(self.row);z['selected_mask']=[0]*7;z['labels']=[-100]*7
        self.assertEqual(validate_tokens(z,[11,13]),0)
        with self.assertRaisesRegex(ValueError,'zero-loss'):validate_tokens(z,[11,13],True)
    def test_label_leak(self):
        self.row['labels'][-1]=16
        with self.assertRaises(ValueError):validate_tokens(self.row)
    def test_reasoning_eligibility(self):
        self.row['eligible_mask'][2]=1
        with self.assertRaisesRegex(ValueError,'reasoning'):validate_tokens(self.row,[11,13])
    def test_first_token(self):
        self.row['eligible_mask'][0]=1
        with self.assertRaises(ValueError):validate_tokens(self.row)
    def test_bool_mask_and_float_labels(self):
        self.row['selected_mask'][4]=True
        with self.assertRaises(ValueError):validate_tokens(self.row)
        self.row['selected_mask'][4]=1;self.row['labels'][4]=14.0
        with self.assertRaises(ValueError):validate_tokens(self.row)
    def test_cap_no_shortening(self):
        self.row['input_ids']=[10]*131073
        with self.assertRaises(ValueError):validate_tokens(self.row)
    def test_data_gate_false(self):
        from contract import SCHEMA
        f=self.tmp/'manifest.json';write_new(f,dict(schema=SCHEMA,mode='FINAL_SELECTED',training_ready=False,data_origin='actual_s3'))
        with self.assertRaisesRegex(ValueError,'data gate'):Release(f,sha(f))
    def test_synthetic_never_released(self):
        from contract import SCHEMA
        f=self.tmp/'manifest.json';write_new(f,dict(schema=SCHEMA,mode='FINAL_SELECTED',training_ready=True,data_origin='synthetic'))
        with self.assertRaisesRegex(ValueError,'fixture'):Release(f,sha(f))
    def test_controls_never_production(self):
        from contract import SCHEMA
        f=self.tmp/'manifest.json';write_new(f,dict(schema=SCHEMA,mode='eligible_control'))
        with self.assertRaisesRegex(ValueError,'FINAL_SELECTED'):Release(f,sha(f))
    def test_manifest_hash_failure(self):
        f=self.tmp/'m';f.write_text('{}')
        with self.assertRaisesRegex(ValueError,'digest'):Release(f,'0'*64)
    def test_source_hash_failure(self):
        f=self.tmp/'x';f.write_text('old');ref={'path':'x','sha256':sha(f)};f.write_text('new')
        with self.assertRaisesRegex(ValueError,'hash'):verify_ref(self.tmp,ref)
        with self.assertRaisesRegex(ValueError,'source'):verify_tree(self.tmp,{'x':ref['sha256']})
    def test_no_checkpoint_no_eval(self):
        with self.assertRaisesRegex(ValueError,'missing checkpoint'):checkpoint(self.tmp/'missing','0'*64,self.tmp/'receipt')
    def test_output_no_reuse(self):
        f=self.tmp/'x';write_new(f,{})
        with self.assertRaises(FileExistsError):write_new(f,{})
        from train_s3 import plan
        with self.assertRaisesRegex(ValueError,'new runtime output'):plan('missing','x','full',self.tmp,'missing','missing')
    def test_shards(self):
        s=shards(IDS);self.assertEqual(len(s['h0']),20);self.assertEqual(len(s['h1']),20)
        with self.assertRaises(ValueError):shards(IDS[:-1]+[IDS[0]])
    def test_no_claim_two_selector_fixture(self):
        from contract import read_json
        f=read_json(Path(__file__).parent/'fixtures/selector_no_claim.json')
        self.assertFalse(f['training_ready']);self.assertEqual(f['data_origin'],'synthetic')
        keep=[(x+y)/2>=7 and max(r,s)<=2 and not v for x,y,r,s,v in zip(
            f['overall1'],f['overall2'],f['risky1'],f['risky2'],f['mechanical_veto'])]
        self.assertEqual(keep,f['expected_keep'])

    def test_wire_classifier(self):
        w=dict(request={'model':'s3','reasoning_effort':'medium'},status_code=200,outcome='empty',
               response={'model':'s3','choices':[{'message':{'reasoning_content':'thinking only','content':None,'tool_calls':[]},'finish_reason':'length'}]})
        c=classify_wire(w,'s3');self.assertEqual(c['no_visible_output'],1);self.assertFalse(c['infra'])
        w['error']='timeout';self.assertTrue(classify_wire(w,'s3')['infra'])
        w['response']['model']='s2';self.assertTrue(classify_wire(w,'s3')['identity_or_effort_drift'])
    def test_denominator_missing(self):
        r=aggregate([self.tmp]);self.assertEqual(r['denominator'],40);self.assertEqual(r['counts']['missing'],40)
    def test_config_delta(self):
        c={'llm':dict(model='base',base_url='x',reasoning_effort='low',temperature=.7,max_tokens=32768,
                     fixed_output_budget=True,context_window=131072),'agent':{'max_iterations':40},'tools':['unchanged']}
        n=config_for(c,'h0');self.assertEqual(n['tools'],c['tools']);self.assertEqual(n['llm']['reasoning_effort'],'medium')

if __name__=='__main__':unittest.main(verbosity=2)
