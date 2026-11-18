"""Export pilot rows for the LLM leak-span audit (CPU, read-only on source data).
Pilot (seed 20261105):
  dep  : 30 rows rejected ONLY for privileged_answer_leak and not regex-repairable (nonconfirm_leak, 635 pool)
  conf : 10 rows already regex-repaired (confirmation sentence deleted) -> the ORIGINAL text, to see if LLM agrees
  s1   : 20 rows from the current S1 training set (1430) -> does the LLM find hidden-answer dependence there?
Each reasoning is split into numbered segments with the same SPLIT rule as repair_leaks.py, so the LLM returns
segment indices (not free text) and every kept character is provably original.
"""
import json, re, random, glob
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
R = Path(RETRACE_SHARE + '/stage1-selfdistill')
P = R / 'projection_batch_v1'
OUT = Path(RETRACE_SHARE + '/stage1-leakaudit')
SPLIT = re.compile(r'(?<=[.!?])[ \t]+|\n')


def segments(text):
    out, pos = [], 0
    for m in SPLIT.finditer(text):
        out.append([pos, m.end()]); pos = m.end()
    out.append([pos, len(text)])
    return [s for s in out if text[s[0]:s[1]].strip()]


def main():
    OUT.mkdir(exist_ok=True)
    src = {json.loads(l)['row_id']: json.loads(l) for l in open(R / 'inputs/reference_generation_candidates.jsonl')}
    rows = {}
    for f in (P / 'full_v5/rows').glob('*.json'):
        r = json.load(open(f)); rows[r['row_id']] = r
    rep = {}
    for f in (P / 'full_v5/repaired').glob('*.json'):
        r = json.load(open(f)); rep[r['row_id']] = r
    kept = [f.replace('.json', '').replace('_', ':', 2) for f in json.load(open(P / 'semantic_review_full/FINAL_SELECTION.json'))['kept_files']]
    leak_only = [rid for rid, r in rows.items() if not r['admitted'] and r.get('reasoning')
                 and {x.split(':')[0] for x in r['reject_reasons']} == {'privileged_answer_leak'}]
    dep_pool = sorted(rid for rid in leak_only if rid not in rep)
    rnd = random.Random(20261105)
    pick = [('dep', x) for x in rnd.sample(dep_pool, 30)] + [('conf', x) for x in rnd.sample(sorted(rep), 10)] + \
           [('s1', x) for x in rnd.sample(sorted(kept), 20)]
    # S1 rows: use the exact TRAINED text (decoded from train.jsonl tokens; covers regex-repaired rows too)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(RETRACE_SHARE + '/models/Qwen/Qwen3.8-27B', local_files_only=True)
    trained = {}
    for l in open(P / 'train_data/train.jsonl'):
        t = json.loads(l)
        cont = tok.decode(t['input_ids'][t['prompt_length']:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        reasoning, final = cont[:-len('<|im_end|>\n')].split('\n</think>\n\n')
        trained[t['row_id']] = (reasoning, final)
    with open(OUT / 'pilot_inputs.jsonl', 'w') as fh:
        for kind, rid in pick:
            if kind == 's1':
                text, final = trained[rid]
            else:
                text, final = rows[rid]['reasoning'], rows[rid]['content']
            segs = segments(text)
            fh.write(json.dumps({'kind': kind, 'row_id': rid, 'question': src[rid]['student_messages'][0]['content'],
                                 'reference': src[rid]['privileged_reference_answer'], 'reasoning': text,
                                 'segments': segs, 'final': final, 'repaired_in_s1': rid in rep}, ensure_ascii=False) + '\n')
    print('dep_pool', len(dep_pool), 'written', len(pick))


if __name__ == '__main__':
    main()
