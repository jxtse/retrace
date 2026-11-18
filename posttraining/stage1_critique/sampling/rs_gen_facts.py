"""S1 v2 rejection sampling WITH RDKit facts: the base model samples on its own, given RDKit
facts; each target is sampled several times, mostly for targets that the clean (post-audit)
training data did not cover.
Sampling prompt = format-only system message (same as rs_gen.py) + an RDKit facts block computed from the question's
SMILES only (no reference information) + student question. Training prompt stays the student question only.
Every sample: unchanged S1 gates + finalize_rows reference-quote filters (no length cap) + filters for mentioning the
instructions or the facts block / RDKit. Seeds differ from rs_gen.py (offset 1000).
Usage: python rs_gen_facts.py <url> <shard> <n_shards> <K> <ids.json> <out_dir>"""
import asyncio, json, os, re, sys, time
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
PB = Path(RETRACE_SHARE + '/stage1-selfdistill/projection_batch_v1')
sys.path.insert(0, str(PB.parent)); sys.path.insert(0, str(PB))
import aiohttp
import gen
_argv = sys.argv; sys.argv = sys.argv[:1]   # finalize_rows parses sys.argv at import time
import finalize_rows
sys.argv = _argv
from thinking_v2.native import render_prompt
from thinking_v2.common import KWARGS
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
RDLogger.DisableLog('rdApp.*')
assert KWARGS['reasoning_effort'] == 'medium'

MAX_TOKENS = 40960
SYS_FORMAT = ("The task concerns direct single-step precursor proposals. Reagents, catalysts and solvents may be omitted; "
              "standard ancillary reagents may be implied. Do not introduce missing atom-contributing building blocks or "
              "arbitrary multistep sequences. Avoid unrelated structural elaboration.\n"
              "In the final answer return one index|keep/reject|reason line per candidate, in original index order, with exact "
              "lowercase keep or reject, followed by keep: and comma-separated kept indices (or keep: none). No preamble or "
              "code fence. Give concise candidate-specific explanations that can be read with the original question alone.")
FACTS_HEAD = ("Structural facts computed exactly from the SMILES in the question (formula, heavy atoms, rings, functional "
              "groups, atom difference of each candidate set versus the product). They are correct; use them to avoid "
              "miscounting or misreading structures. Reason as if you derived these facts yourself from the SMILES: never "
              "mention these notes, RDKit, or that facts were provided.")
GROUPS = [('carboxylic acid', '[CX3](=O)[OX2H1,OX1-]'), ('ester', '[#6][CX3](=O)[OX2][#6]'), ('amide', '[NX3][CX3]=[OX1]'),
          ('primary amine', '[NX3;H2;!$(NC=[O,S,N])][#6]'), ('secondary amine', '[NX3;H1;!$(NC=[O,S,N]);!$(N[a]);!$(Nc)]([#6])[#6]'),
          ('aldehyde', '[CX3H1](=O)[#6]'), ('ketone', '[#6][CX3](=O)[#6]'), ('nitrile', 'C#N'), ('alcohol', '[OX2H][CX4]'),
          ('phenol', '[OX2H]c'), ('ether', '[OD2]([#6;!$(C=O)])[#6;!$(C=O)]'), ('nitro', '[N+](=O)[O-]'),
          ('sulfonamide', 'S(=O)(=O)N'), ('sulfonyl', '[SX4](=O)(=O)'), ('boron', '[B]'), ('urea', '[NX3]C(=O)[NX3]'),
          ('carbamate', '[NX3]C(=O)O[#6]'), ('alkene', '[CX3]=[CX3]'), ('alkyne', 'C#C'), ('acyl halide', 'C(=O)[Cl,Br,I,F]'),
          ('tin', '[Sn]'), ('magnesium', '[Mg]'), ('zinc', '[Zn]')]
PATS = [(n, Chem.MolFromSmarts(s)) for n, s in GROUPS]


def atoms(m):
    c = {}
    mh = Chem.AddHs(m)
    for a in mh.GetAtoms():
        c[a.GetSymbol()] = c.get(a.GetSymbol(), 0) + 1
    return c


def ring_desc(m):
    out = []
    for r in m.GetRingInfo().AtomRings():
        ar = all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r)
        het = {}
        for i in r:
            s = m.GetAtomWithIdx(i).GetSymbol()
            if s != 'C': het[s] = het.get(s, 0) + 1
        adj = any(m.GetAtomWithIdx(r[k]).GetSymbol() != 'C' and m.GetAtomWithIdx(r[(k + 1) % len(r)]).GetSymbol() != 'C'
                  for k in range(len(r)))
        h = ','.join(f'{k}{v}' for k, v in sorted(het.items()))
        out.append(f"{len(r)}-membered {'aromatic' if ar else 'non-aromatic'}" + (f" ({h}{', heteroatoms adjacent' if adj and sum(het.values()) > 1 else ''})" if h else ''))
    return sorted(out)


def groups(m):
    g = []
    for n, p in PATS:
        k = len(m.GetSubstructMatches(p))
        if k: g.append(f'{n} x{k}')
    hal = {s: sum(a.GetSymbol() == s for a in m.GetAtoms()) for s in ('F', 'Cl', 'Br', 'I')}
    g += [f'{s} x{k}' for s, k in hal.items() if k]
    return g


def mol_line(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, f'{smi}: not parseable'
    st = Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)
    return m, (f"{rdMolDescriptors.CalcMolFormula(m)}, heavy atoms {m.GetNumHeavyAtoms()}; rings: "
               f"{'; '.join(ring_desc(m)) or 'none'}; groups: {', '.join(groups(m)) or 'none'}; "
               f"stereocentres {len(st)} ({sum(1 for _, c in st if c != '?')} specified)")


def facts_block(q):
    prod = re.search(r'^Product:\s*(\S+)', q, re.M).group(1)
    pm, pl = mol_line(prod)
    lines = [FACTS_HEAD, f'Product: {pl}']
    pa = atoms(pm) if pm else {}
    for idx, smi in re.findall(r'^\s*(\d+)\.\s*(\S+)', q, re.M):
        comps = smi.split('.')
        tot = {}; descs = []
        for c in comps:
            m, l = mol_line(c)
            descs.append(f'[{c}] {l}')
            if m is not None:
                for k, v in atoms(m).items(): tot[k] = tot.get(k, 0) + v
        delta = {k: tot.get(k, 0) - pa.get(k, 0) for k in set(tot) | set(pa)}
        ds = ', '.join(f'{k}{v:+d}' for k, v in sorted(delta.items()) if v) or 'none'
        lines.append(f'Candidate {idx} ({len(comps)} component{"s" if len(comps) > 1 else ""}): ' + ' | '.join(descs) +
                     f' | atoms of all components minus product: {ds}')
    return '\n'.join(lines)


FACT_REF = re.compile(r"\b(?:rdkit|facts? (?:block|provided|given|list|section|above|say|says|state|states|show|shows)|"
                      r"(?:provided|given|computed|listed|precomputed) (?:facts?|data|notes?|formulas?)|the (?:facts?|notes?|table|data) "
                      r"(?:say|says|show|shows|state|states|indicate|indicates|list|lists)|structural facts|according to the (?:facts?|notes?|data|table)|"
                      r"atoms of all components minus product)\b", re.I)
INSTR_REF = re.compile(r"\b(?:system (?:prompt|message)|the instructions?|instructed|as (?:requested|required|specified)|"
                       r"the (?:required|requested|specified) format|format (?:requires|says|asks))\b", re.I)


async def do_row(row, svc, tok, K, sem, out):
    f = out / 'rows' / (row['row_id'].replace(':', '_') + '.json')
    if f.exists():
        return 'skip'
    q = row['student_messages'][0]['content']
    fb = facts_block(q)
    msgs = [{'role': 'system', 'content': SYS_FORMAT + '\n\n' + fb}] + row['student_messages']
    s_ids = render_prompt(tok, msgs)['token_ids']
    base_seed = int.from_bytes(row['row_id'].encode()[-6:], 'big') % (2 ** 31) + 1000

    async def one(seed):
        async with sem:
            try:
                p = await svc.propose(s_ids, [], MAX_TOKENS, seed)
            except Exception as e:
                return {'seed': seed, 'infra_error': str(e)[:300]}
            ids = p['ids']
            reasons, flags, extra = gen.gates(tok, row, ids)
            if not reasons:
                rr = {'reasoning': extra['reasoning'], 'content': extra['content'], 'train_tokens': extra['train_tokens']}
                fin = finalize_rows.MAX_TOKENS; finalize_rows.MAX_TOKENS = 10 ** 9
                reasons = finalize_rows.extra_reasons(rr, q, None)
                finalize_rows.MAX_TOKENS = fin
                t = extra['reasoning'] + '\n' + extra['content']
                if INSTR_REF.search(t): reasons.append('instruction_reference')
                if FACT_REF.search(t): reasons.append('facts_reference')
            return {'seed': seed, 'finish': p['finish'], 'n_tokens': len(ids), 'latency': round(p['latency'], 1),
                    'reasons': reasons, 'flags': flags, 'accepted': not reasons,
                    'reasoning': extra.get('reasoning'), 'content': extra.get('content'),
                    'train_tokens': extra.get('train_tokens'), 'ids': ids if not reasons else None}
    samples = await asyncio.gather(*[one(base_seed + k) for k in range(K)])
    if any('infra_error' in s for s in samples):
        return 'infra'
    rec = {'row_id': row['row_id'], 'source_key': row['source_key'], 'composition': row['composition'],
           'sampling_prompt_tokens': len(s_ids), 'facts_block': fb, 'kwargs': KWARGS, 'sampling': gen.SAMPLING,
           'max_tokens': MAX_TOKENS, 'samples': samples, 'n_accepted': sum(s['accepted'] for s in samples)}
    tmp = f.with_suffix('.tmp'); tmp.write_text(json.dumps(rec, ensure_ascii=False)); tmp.rename(f)
    return 'ok'


async def main(url, shard, nshard, K, ids_file, out):
    out = Path(out); (out / 'rows').mkdir(parents=True, exist_ok=True)
    tok = gen.load_tokenizer(PB.parent)
    src = {json.loads(l)['row_id']: json.loads(l) for l in open(PB.parent / 'inputs/reference_generation_candidates.jsonl')}
    ids = json.load(open(ids_file))[shard::nshard]
    sem = asyncio.Semaphore(int(os.environ.get('RS_CONC', '32')))
    t0 = time.time(); c = {'ok': 0, 'skip': 0, 'infra': 0}
    async with aiohttp.ClientSession() as session:
        svc = gen.Service(url, session)
        tasks = [asyncio.create_task(do_row(src[i], svc, tok, K, sem, out)) for i in ids]
        for i, t in enumerate(asyncio.as_completed(tasks), 1):
            c[await t] += 1
            if i % 10 == 0 or i == len(tasks):
                print(json.dumps({'t': round(time.time() - t0), 'done': i, 'of': len(tasks), **c}), flush=True)


if __name__ == '__main__':
    a = sys.argv
    if a[1] == 'show':
        src = {json.loads(l)['row_id']: json.loads(l) for l in open(PB.parent / 'inputs/reference_generation_candidates.jsonl')}
        print(facts_block(src[a[2]]['student_messages'][0]['content']))
    else:
        asyncio.run(main(a[1], int(a[2]), int(a[3]), int(a[4]), a[5], a[6]))
