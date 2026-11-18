"""Export calibration + full-audit inputs for the DeepSeek chemistry/consistency re-audit of S1 (CPU).
For every S1 training row: question, reference, TRAINED reasoning/final (decoded from train.jsonl tokens),
the prior sub-agent verdict/flags/evidence, and RDKit facts for every SMILES in the question
(formula, heavy atoms, ring sizes, aromatic ring count, canonical SMILES) so the auditor can check counts.
Calibration subset (seed 20261105): 20 prior FAIL(chem_error) [from dropped rows], 30 prior MINOR+chem_error,
30 prior PASS (no flags) -> measures agreement before the full run.
"""
import json, re, random, glob
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
R = Path(RETRACE_SHARE + '/stage1-selfdistill')
P = R / 'projection_batch_v1'
OUT = Path(RETRACE_SHARE + '/stage1-leakaudit/chem')


def facts(smi):
    out = []
    for part in smi.split('.'):
        m = Chem.MolFromSmiles(part)
        if m is None:
            out.append({'smiles': part, 'parse': False}); continue
        ri = m.GetRingInfo()
        out.append({'smiles': part, 'formula': rdMolDescriptors.CalcMolFormula(m), 'heavy_atoms': m.GetNumHeavyAtoms(),
                    'ring_sizes': sorted(len(r) for r in ri.AtomRings()),
                    'aromatic_rings': sum(all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r) for r in ri.AtomRings())})
    return out


def q_facts(q):
    prod = re.search(r'^Product:\s*(\S+)', q, re.M).group(1)
    cands = re.findall(r'^\s*(\d+)\.\s*(\S+)', q, re.M)
    return {'product': facts(prod), 'candidates': {i: facts(s) for i, s in cands}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(RETRACE_SHARE + '/models/Qwen/Qwen3.8-27B', local_files_only=True)
    src = {json.loads(l)['row_id']: json.loads(l) for l in open(R / 'inputs/reference_generation_candidates.jsonl')}
    m = {}
    for f in glob.glob(str(P / 'semantic_review_full/packet_*.json')):
        for it in json.load(open(f)): m[it['review_id']] = it['row_id']
    prior = {}
    for f in glob.glob(str(P / 'semantic_review_full/results_*.json')):
        for r in json.load(open(f)):
            if r['review_id'] in m: prior[m[r['review_id']]] = r
    for rid, v in json.load(open(P / 'semantic_review_full/prior_reviews.json')).items():
        prior.setdefault(rid, v)
    # texts: trained rows from train.jsonl; dropped FAIL rows from rows/ or repaired/
    texts = {}
    for l in open(P / 'train_data/train.jsonl'):
        t = json.loads(l)
        cont = tok.decode(t['input_ids'][t['prompt_length']:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        a, b = cont[:-len('<|im_end|>\n')].split('\n</think>\n\n')
        texts[t['row_id']] = (a, b, True)
    for d in ('repaired', 'rows'):
        for f in (P / 'full_v5' / d).glob('*.json'):
            r = json.load(open(f))
            if r['row_id'] not in texts and r.get('reasoning'):
                texts[r['row_id']] = (r['reasoning'], r['content'], False)

    def rec(rid):
        q = src[rid]['student_messages'][0]['content']
        a, b, in_s1 = texts[rid]
        p = prior.get(rid, {})
        return {'row_id': rid, 'in_s1': in_s1, 'question': q, 'reference': src[rid]['privileged_reference_answer'],
                'reasoning': a, 'final': b, 'rdkit': q_facts(q), 'prior_verdict': p.get('verdict'),
                'prior_flags': p.get('flags', []), 'prior_evidence': p.get('evidence', '')}
    s1 = [rid for rid, v in texts.items() if v[2]]
    fail_chem = sorted(rid for rid, p in prior.items() if p.get('verdict') == 'FAIL' and 'chem_error' in p.get('flags', [])
                       and rid in texts)
    minor_chem = sorted(rid for rid in s1 if prior.get(rid, {}).get('verdict') == 'MINOR'
                        and 'chem_error' in prior[rid].get('flags', []))
    passed = sorted(rid for rid in s1 if prior.get(rid, {}).get('verdict') == 'PASS' and not prior[rid].get('flags'))
    rnd = random.Random(20261105)
    cal = rnd.sample(fail_chem, 20) + rnd.sample(minor_chem, 30) + rnd.sample(passed, 30)
    with open(OUT / 'calib_inputs.jsonl', 'w') as fh:
        for rid in cal: fh.write(json.dumps(rec(rid), ensure_ascii=False) + '\n')
    with open(OUT / 's1_all_inputs.jsonl', 'w') as fh:
        for rid in sorted(s1): fh.write(json.dumps(rec(rid), ensure_ascii=False) + '\n')
    print('s1', len(s1), 'fail_chem pool', len(fail_chem), 'minor_chem', len(minor_chem), 'pass', len(passed), 'calib', len(cal))


if __name__ == '__main__':
    main()
