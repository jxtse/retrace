"""Finalize training rows after full generation.

Pipeline: admitted originals + leak-repaired rows -> extra filters learned from semantic review round 1 -> train_rows/.
Extra filters (each motivated by sub-agent findings in semantic_review_r1):
  F1 instruction_quote: thinking/final copies >=6 consecutive words of the teacher-only instruction (e.g. "Reagents,
     catalysts and solvents may be omitted"), which the student never sees.
  F2 hidden_source_phrase: "mentioned in the answer", "from the answer", "the answer says", and
     "the problem/task says/states" followed by a quote that is not in the question.
  F3 length cap: train_tokens > MAX_TOKENS. In round 1, non-leak FAIL rate was 4/79 (5%) at <=8000 tokens and 5/8 above.
  F4 reference_fragment_copy: a backtick local-graph fragment that exists only in the hidden reference (not in the
     question) appears verbatim in the thinking or final; reviewers flagged these as underived copies.
"""
import json, re, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

instr = re.search(r"TEACHER_INSTRUCTION_V4 = '''(.*?)'''", (HERE / 'gen.py').read_text(), re.S).group(1)
_w = instr.split()
NGRAMS = {' '.join(_w[i:i + 6]).lower() for i in range(len(_w) - 5)}
SRC_PHRASE = re.compile(r"\b(?:mentioned|given|stated|quoted|listed) in the (?:answer|explanation|solution)\b|"
                        r"\b(?:from|in|per) the answer\b|\bthe answer (?:says|states|mentions|notes|gives)\b", re.I)
QUOTE = re.compile(r"\b(?:the|this) (?:problem|task|prompt|question|instructions?) (?:says|states|notes|mentions|specifies|requires)"
                   r"[^\"“\n]{0,30}[\"“]([^\"”\n]{8,})[\"”]", re.I)


def extra_reasons(rec, question, ref=None):
    t = rec['reasoning'] + '\n' + rec['content']
    flat = ' '.join(t.split()).lower()
    out = []
    if any(g in flat for g in NGRAMS):
        out.append('instruction_quote')
    if SRC_PHRASE.search(t):
        out.append('hidden_source_phrase')
    qn = ' '.join(question.split()).lower()
    for m in QUOTE.finditer(t):
        if ' '.join(m.group(1).split()).lower() not in qn:
            out.append('quote_not_in_question'); break
    if rec['train_tokens'] > MAX_TOKENS:
        out.append(f'length_gt_{MAX_TOKENS}')
    if ref is not None:
        frags = [f for f in re.findall(r'`([^`]{12,})`', ref) if f not in question]
        if any(f in t for f in frags):
            out.append('reference_fragment_copy')
    return out




def main():
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'full_v5'
    src = {json.loads(l)['row_id']: json.loads(l) for l in open(ROOT / 'inputs/reference_generation_candidates.jsonl')}
    out = run / 'train_rows'
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    stats = {'admitted_original': 0, 'repaired': 0, 'kept': 0, 'kept_repaired': 0, 'dropped': {}}
    ledger = []
    cands = [(f, False) for f in sorted((run / 'rows').glob('*.json'))] + \
            [(f, True) for f in sorted((run / 'repaired').glob('*.json'))]
    for f, rep in cands:
        r = json.load(open(f))
        if not r['admitted']:
            continue
        stats['repaired' if rep else 'admitted_original'] += 1
        s = src[r['row_id']]
        reasons = extra_reasons(r, s['student_messages'][0]['content'], s['privileged_reference_answer'])
        ledger.append({'row_id': r['row_id'], 'repaired': rep, 'train_tokens': r['train_tokens'], 'drop': reasons})
        if reasons:
            for x in reasons:
                stats['dropped'][x] = stats['dropped'].get(x, 0) + 1
            continue
        assert not (out / f.name).exists()
        shutil.copy(f, out / f.name)
        stats['kept'] += 1; stats['kept_repaired'] += rep
    with open(run / 'finalize_ledger.jsonl', 'w') as fh:
        for x in ledger:
            fh.write(json.dumps(x) + '\n')
    json.dump(stats, open(run / 'finalize_stats.json', 'w'), indent=1)
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
