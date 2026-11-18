import json, sys, os

# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
sys.path.insert(0, RETRACE_SHARE + '/online-arms/scripts')
os.environ.setdefault('RETRACE_MANIFEST', RETRACE_SHARE + '/online-arms/data/manifest.json')
import retrace_template as T
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(RETRACE_SHARE + '/models/Qwen/Qwen3.8-27B', trust_remote_code=True)
for line in open(RETRACE_SHARE + '/online-arms/data/rows_gkd_pair.jsonl'):
    r = json.loads(line)
    spans = T.generation_spans(r['input_ids'], tok)
    n = sum(e - s for s, e in spans)
    print(r['uid'], 'spans', len(spans), 'n_loss_scan', n, 'n_loss_row', r['n_loss_tokens'],
          'match', n == r['n_loss_tokens'])
