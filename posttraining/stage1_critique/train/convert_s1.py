"""Convert S1 training rows (projection_batch_v1/train_data/train.jsonl, 1430 single-turn thinking rows) into
ms-swift 4.6 messages JSONL and verify on EVERY row that swift's qwen3_8 template (train mode, same settings as the
A1-swift run) reproduces exactly:
  - input_ids == S1 input_ids (S1 rendered via HF chat template, enable_thinking/medium/preserve_thinking)
  - (labels != -100 and loss_scale > 0) == S1 completion_mask, except swift also supervises the 2 tokens
    '<think>' '\n' right after '<|im_start|>assistant\n' (S1 put them in the generation prompt, as vLLM does at
    inference). Any other diff fails the row.
Source of messages: inputs/reference_generation_candidates.jsonl (student_messages) + full_v5/rows/*.json
(reasoning, content) -- the same join build_dataset.py used.
Usage: venv-swift/bin/python convert_s1.py [N|all]"""
import json, sys, glob, hashlib
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
R = Path(RETRACE_SHARE + '/stage1-selfdistill')
P = R / 'projection_batch_v1'
MODEL = RETRACE_SHARE + '/models/Qwen/Qwen3.8-27B'
OUT = Path(__file__).resolve().parent
KWARGS = {'enable_thinking': True, 'reasoning_effort': 'medium', 'preserve_thinking': True}


def main(arg):
    import swift
    from swift import get_processor, get_template
    proc = get_processor(MODEL)
    tok = proc.tokenizer if hasattr(proc, 'tokenizer') else proc
    tmpl = get_template(proc, template_type='qwen3_8', loss_scale='ignore_empty_think', add_non_thinking_prefix=True,
                        enable_thinking=True, template_backend='swift')
    tmpl.set_mode('train')
    src = {json.loads(l)['row_id']: json.loads(l) for l in open(R / 'inputs/reference_generation_candidates.jsonl')}
    gen = {}
    for f in glob.glob(str(P / 'full_v5/rows/*.json')):
        r = json.load(open(f))
        gen[r['row_id']] = r
    train = [json.loads(l) for l in open(P / 'train_data/train.jsonl')]
    if arg != 'all':
        train = sorted(train, key=lambda r: len(r['input_ids']))
        n = int(arg)
        train = [train[0], train[len(train) // 2], train[-1]][:n] if n <= 3 else train[:n]
    rep = {'swift_version': swift.__version__, 'n': 0, 'ok': 0, 'fail': [], 'loss_tokens_swift': 0, 'loss_tokens_s1': 0,
           'source': str(P / 'train_data/train.jsonl'),
           'source_sha256': hashlib.sha256((P / 'train_data/train.jsonl').read_bytes()).hexdigest()}
    out_f = open(OUT / ('train_s1_swift.jsonl' if arg == 'all' else 'train_s1_swift_sample.jsonl'), 'w')
    for t in train:
        rid = t['row_id']
        # use the repaired/finalized text if build_dataset consumed it; verify by token equality below
        # recover the exact trained text from S1's own tokens (rows may have been leak-repaired after generation)
        pl = t['prompt_length']
        cont = tok.decode(t['input_ids'][pl:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        assert cont.endswith('<|im_end|>\n') and cont.count('\n</think>\n\n') == 1, rid
        reasoning, content = cont[:-len('<|im_end|>\n')].split('\n</think>\n\n')
        msgs = json.loads(json.dumps(src[rid]['student_messages'])) + [
            {'role': 'assistant', 'content': '<think>\n' + reasoning.strip() + '\n</think>\n\n' + content.strip()}]
        # swift's qwen3_5/3_8 template ignores `reasoning_content`; thinking must be inline in content
        # (it re-wraps <think>...</think> canonically, matching the HF template)
        sample = {'messages': msgs, 'chat_template_kwargs': KWARGS}
        enc = tmpl.encode(json.loads(json.dumps(sample)))
        ids = list(enc['input_ids'])
        ls = enc.get('loss_scale')
        mask = [int(l != -100 and (ls is None or ls[i] > 0)) for i, l in enumerate(enc['labels'])]
        ref_ids, ref_mask = t['input_ids'], t['completion_mask']
        res = {'row_id': rid, 'n_ref': len(ref_ids), 'n_swift': len(ids), 'ids_equal': ids == ref_ids,
               'mask_equal': mask == ref_mask, 'loss_tokens': [sum(mask), sum(ref_mask)]}
        diff = [i for i in range(min(len(ids), len(ref_ids))) if mask[i] != ref_mask[i]]
        think_open = tok.convert_tokens_to_ids('<think>')
        cls = {'think_open_prefix': 0, 'other': 0}
        for i in diff:
            if mask[i] and not ref_mask[i] and i < pl and (ids[i] == think_open or (i > 0 and ids[i - 1] == think_open)):
                cls['think_open_prefix'] += 1
            else:
                cls['other'] += 1
        res['mask_diff'] = cls
        res['mask_ok'] = res['ids_equal'] and cls['other'] == 0
        rep['n'] += 1
        if res['mask_ok']:
            rep['ok'] += 1
            rep['loss_tokens_swift'] += sum(mask); rep['loss_tokens_s1'] += sum(ref_mask)
            out_f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        else:
            for k2, a_, b_ in (('ids', ids, ref_ids), ('mask', mask, ref_mask)):
                d = next((i for i, (x, y) in enumerate(zip(a_, b_)) if x != y), None)
                if d is None and len(a_) != len(b_):
                    d = min(len(a_), len(b_))
                if d is not None:
                    res[f'{k2}_first_diff'] = d
                    res[f'{k2}_ctx_swift'] = tok.decode(ids[max(0, d - 10):d + 10])
                    res[f'{k2}_ctx_ref'] = tok.decode(ref_ids[max(0, d - 10):d + 10])
            rep['fail'].append(res)
        if rep['n'] <= 5 or not res['mask_ok']:
            print(json.dumps(res, ensure_ascii=False)[:800])
    out_f.close()
    (OUT / f'convert_s1_report_{arg}.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print('SUMMARY', rep['n'], 'ok', rep['ok'], 'fail', len(rep['fail']), 'loss_tokens', rep['loss_tokens_swift'], rep['loss_tokens_s1'])


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '3')
