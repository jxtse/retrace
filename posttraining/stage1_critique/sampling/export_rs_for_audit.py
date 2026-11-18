"""Export accepted rejection-sampling samples into chem_audit input format (row_id = row_id#seed)."""
import json, glob, sys
from pathlib import Path

# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
sys.path.insert(0, RETRACE_SHARE + '/stage1-leakaudit')
from export_chem import q_facts

R = Path(RETRACE_SHARE + '/stage1-selfdistill')
src = {json.loads(l)['row_id']: json.loads(l) for l in open(R / 'inputs/reference_generation_candidates.jsonl')}
out = open(sys.argv[2], 'w'); n = 0
for f in sorted(glob.glob(sys.argv[1] + '/*.json')):
    r = json.load(open(f))
    q = src[r['row_id']]['student_messages'][0]['content']
    for s in r['samples']:
        if s['accepted']:
            out.write(json.dumps({'row_id': f"{r['row_id']}#{s['seed']}", 'question': q,
                                  'reference': src[r['row_id']]['privileged_reference_answer'],
                                  'reasoning': s['reasoning'], 'final': s['content'], 'rdkit': q_facts(q),
                                  'n_tokens': s['n_tokens']}, ensure_ascii=False) + '\n'); n += 1
print('exported', n)
