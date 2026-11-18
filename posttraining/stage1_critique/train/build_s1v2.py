"""Build the S1 v2 training file for ms-swift.
Selection (s1v2_selection.json): DeepSeek-audited pass/minor only; max 2 per question, pass first, then S1 first.
  - src s1     : current S1 rows (verbatim trained text; already token-verified vs S1 in train_s1_swift.jsonl)
  - src rs/rsfacts : base rejection-sampling samples (student-prompt training rows; accepted by S1 gates + filters)
Every output row: messages = student question + assistant '<think>\\nR\\n</think>\\n\\nF', chat_template_kwargs medium.
Checks: swift template encode succeeds; input_ids == HF apply_chat_template(student, medium, + assistant) tokens;
loss covers the whole assistant turn; no LEAK/instruction/facts words in text; no duplicate texts.
Usage: venv-swift/bin/python build_s1v2.py"""
import json, re, sys, hashlib, collections
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
B = Path(__file__).resolve().parent
PB = Path(RETRACE_SHARE + '/stage1-selfdistill/projection_batch_v1')
RS = Path(RETRACE_SHARE + '/stage1-sampling')
MODEL = RETRACE_SHARE + '/models/Qwen/Qwen3.8-27B'
KW = {'enable_thinking': True, 'reasoning_effort': 'medium', 'preserve_thinking': True}
BAD = re.compile(r"\bhidden\b|\breference\b|answer key|\bteacher\b|\bstudent\b|\binstructions?\b|\bas given\b|"
                 r"expected answer|\brdkit\b|structural facts|system (?:prompt|message)", re.I)


def main():
    from swift import get_processor, get_template
    proc = get_processor(MODEL)
    tok = proc.tokenizer if hasattr(proc, 'tokenizer') else proc
    tmpl = get_template(proc, template_type='qwen3_8', loss_scale='ignore_empty_think', add_non_thinking_prefix=True,
                        enable_thinking=True, template_backend='swift')
    tmpl.set_mode('train')
    src = {json.loads(l)['row_id']: json.loads(l) for l in open(PB.parent / 'inputs/reference_generation_candidates.jsonl')}
    s1 = {}
    for l in open(B / 'train_s1_swift.jsonl'):
        x = json.loads(l)
        s1[x['messages'][0]['content']] = x
    rs = {}
    for d in ('rows', 'facts_pilot/rows'):
        for f in (RS / d).glob('*.json'):
            r = json.load(open(f))
            for s in r['samples']:
                if s['accepted']:
                    rs[f"{r['row_id']}#{s['seed']}"] = s
    sel = json.load(open(B / 's1v2_selection.json'))
    out, seen, rep = [], set(), collections.Counter()
    for s in sel:
        q = src[s['qid']]['student_messages'][0]['content']
        if s['src'] == 's1':
            x = s1[q]
            content = x['messages'][1]['content']
        else:
            smp = rs[s['sid']]
            content = '<think>\n' + smp['reasoning'].strip() + '\n</think>\n\n' + smp['content'].strip()
        body = content.split('</think>', 1)
        if BAD.search(content):
            rep['drop_bad_word'] += 1; continue
        h = hashlib.sha256(content.encode()).hexdigest()
        if h in seen:
            rep['drop_dup'] += 1; continue
        seen.add(h)
        sample = {'messages': [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': content}],
                  'chat_template_kwargs': KW, '_src': s['src'], '_verdict': s['verdict'], '_sid': s['sid']}
        enc = tmpl.encode({k: v for k, v in sample.items() if not k.startswith('_')})
        ids = list(enc['input_ids'])
        hf = tok.apply_chat_template([{'role': 'user', 'content': q},
                                      {'role': 'assistant', 'reasoning_content': body[0].replace('<think>', '').strip(),
                                       'content': body[1].strip()}], tokenize=False, **KW)
        hf_ids = tok(hf, add_special_tokens=False)['input_ids']
        if ids != hf_ids:
            rep['drop_token_mismatch'] += 1; continue
        ls = enc.get('loss_scale')
        n_loss = sum(1 for i, l in enumerate(enc['labels']) if l != -100 and (ls is None or ls[i] > 0))
        rep['loss_tokens'] += n_loss; rep['tokens'] += len(ids); rep['rows_' + s['src']] += 1
        out.append(sample)
    with open(B / 'train_s1v2_swift.jsonl', 'w') as fh:
        for x in out:
            fh.write(json.dumps({k: v for k, v in x.items() if not k.startswith('_')}, ensure_ascii=False) + '\n')
    with open(B / 'train_s1v2_manifest.jsonl', 'w') as fh:
        for x in out:
            fh.write(json.dumps({'sid': x['_sid'], 'src': x['_src'], 'verdict': x['_verdict']}) + '\n')
    rep['rows'] = len(out); rep['questions'] = len({x['messages'][0]['content'] for x in out})
    rep['sha256'] = hashlib.sha256((B / 'train_s1v2_swift.jsonl').read_bytes()).hexdigest()
    lens = sorted(len(tok(x['messages'][1]['content'], add_special_tokens=False)['input_ids']) for x in out)
    rep['assistant_tokens_p50_p90_max'] = [lens[len(lens) // 2], lens[int(.9 * len(lens))], lens[-1]]
    (B / 'train_s1v2_report.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
