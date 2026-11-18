"""Intentionally RED on unchanged ms-swift text interface; not the v2 suite."""
import os

# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import unittest,json,importlib.util
from pathlib import Path
from swift import get_processor,get_template,load_dataset
O=Path(__file__).resolve().parent; V1=O.parent/'v1'
class LegacyDefects(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('conversion',V1/'pinned/convert_and_check.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        cls.row=min(map(json.loads,(V1/'control/INDEX.jsonl').read_text().splitlines()),key=lambda r:r['n_tokens'])
        cls.ref=json.loads((V1/'control/tok'/f"{cls.row['uid']}.json").read_text())
        msg=json.loads((V1/'control/msgs'/f"{cls.row['uid']}.json").read_text())
        cls.raw={'messages':mod.prep(msg['messages'],[False]*cls.row['n_assistant_turns']),
                 'tools':msg['tools'],'chat_template_kwargs':msg['chat_template_kwargs']}
        cls.t=get_template(get_processor(RETRACE_SHARE + '/models/Qwen/Qwen3.8-27B'),template_type='qwen3_8',
                           loss_scale='ignore_empty_think',enable_thinking=True,add_non_thinking_prefix=True,template_backend='swift')
        cls.t.set_mode('train')
    def test_native_text_loader_token_parity(self):
        path=O/'legacy_zero_control.jsonl';path.write_text(json.dumps(self.raw)+'\n')
        ds,_=load_dataset(str(path),strict=True,shuffle=False,remove_unused_columns=False,load_from_cache_file=False)
        ids=self.t.encode(dict(ds[0]))['input_ids']
        first=next((i for i,(a,b) in enumerate(zip(ids,self.ref['input_ids'])) if a!=b),None)
        self.assertIsNone(first,f'native text loader changed tokens at {first}')
    def test_native_zero_mask_has_no_suffix_leak(self):
        enc=self.t.encode(json.loads(json.dumps(self.raw)))
        batch=self.t.data_collator([enc])
        count=sum(x!=-100 for x in batch['labels'][0].tolist()[1:])
        self.assertEqual(count,0,'native zero-mask suffix labels leaked')
if __name__=='__main__':unittest.main()
