"""CPU-only real-row acceptance. No pipeline/model construction."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import unittest, json, copy
from pathlib import Path
O=Path(__file__).resolve().parent
V1=O.parent/'v1'
class NativeAcceptance(unittest.TestCase):
    def test_interface_exists(self):
        self.assertTrue((O/'pretokenized.py').exists(), 'missing supported pretokenized custom-template adapter')
    def test_all_real_rows(self):
        from pretokenized import run_acceptance
        report=run_acceptance()
        self.assertEqual(report['unique_real_rows'],136)
        self.assertEqual(report['cases'],408)
        self.assertTrue(report['exact_full_tokens_and_masks'])
        self.assertEqual(report['pending_selected_loss_tokens'],0)
    def test_collation_without_even_meta_model_construction(self):
        from unittest.mock import patch
        from pretokenized import record,make_template
        t=make_template(); index=json.loads((V1/'control/INDEX.jsonl').read_text().splitlines()[0])
        with patch.object(t,'_get_model',side_effect=AssertionError('model construction forbidden')):
            t.data_collator([t.encode(record(index,'pending_zero'))])
        self.assertIsNone(t.model)
        self.assertIsNone(t.dummy_model)
    def test_tampering_rejected(self):
        from pretokenized import record, make_template
        index=json.loads((V1/'control/INDEX.jsonl').read_text().splitlines()[0])
        t=make_template()
        r=record(index,'pending_zero')
        for kind in ['ids','labels','uid','mode','length','reasoning','scale']:
            bad=copy.deepcopy(r)
            if kind=='ids': bad['input_ids'][0]+=1
            elif kind=='labels': bad['labels'][-1]=bad['input_ids'][-1]
            elif kind=='uid': bad['uid']='../escape'
            elif kind=='mode': bad['control_mode']='admitted'
            elif kind=='length': bad['lengths']=[1]
            elif kind=='reasoning':
                pos=next(i for i,x in enumerate(bad['input_ids']) if x==t.tokenizer.convert_tokens_to_ids('<think>'))
                bad['labels'][pos]=bad['input_ids'][pos]
            else: bad['loss_scale']=[1.0]*len(bad['input_ids'])
            with self.subTest(kind=kind), self.assertRaises(ValueError): t.encode(bad)
if __name__=='__main__': unittest.main()
