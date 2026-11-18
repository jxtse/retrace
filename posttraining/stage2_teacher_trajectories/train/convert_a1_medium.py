"""A1 data re-rendered at reasoning_effort=medium (train/infer consistency: high-effort teacher
trajectories are trained at medium, not low). CPU only.

Input : swift-env/train_swift.jsonl (167 rows, already verified token-exact vs A1 at low).
Output: train_a1_medium.jsonl  (identical messages / per-turn loss flags; only chat_template_kwargs effort -> medium)

Checks on EVERY row:
  1. swift(medium) input_ids == HF apply_chat_template(medium) tokens (prefix; swift drops nothing at the end because
     rows already end on an assistant turn).
  2. swift(medium) == swift(low) with exactly one contiguous block removed, the block decodes to the low-effort
     instruction + '\\n\\n', and none of the removed tokens carry loss.
  3. loss mask(medium) == loss mask(low) with that block removed  => same supervised tokens, same per-turn selection.
Usage: venv-swift/bin/python convert_a1_medium.py [N|all]
"""
import json, sys, hashlib
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
SRC = Path(RETRACE_SHARE + '/swift-env/train_swift.jsonl')
MODEL = RETRACE_SHARE + '/models/Qwen/Qwen3.8-27B'
OUT = Path(__file__).resolve().parent
LOW_INSTR = ('Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the '
             'conclusion without unnecessary elaboration.')


def enc(tmpl, sample):
    e = tmpl.encode(json.loads(json.dumps(sample)))
    ls = e.get('loss_scale')
    mask = [int(l != -100 and (ls is None or ls[i] > 0)) for i, l in enumerate(e['labels'])]
    return list(e['input_ids']), mask


def hf_messages(sample):
    """Undo swift-only data shaping for the HF reference render: split the merged user message back, drop 'loss'."""
    SEP = '<|im_end|>\n<|im_start|>user\n'
    out = []
    for m in sample['messages']:
        m = {k: v for k, v in m.items() if k != 'loss'}
        if m['role'] == 'user' and SEP in (m.get('content') or ''):
            for part in m['content'].split(SEP):
                out.append({'role': 'user', 'content': part})
        else:
            out.append(m)
    return out


def main(arg):
    import swift
    from swift import get_processor, get_template
    proc = get_processor(MODEL)
    tok = proc.tokenizer if hasattr(proc, 'tokenizer') else proc
    tmpl = get_template(proc, template_type='qwen3_8', loss_scale='ignore_empty_think', add_non_thinking_prefix=True,
                        enable_thinking=True, template_backend='swift')
    tmpl.set_mode('train')
    rows = [json.loads(l) for l in SRC.read_text().splitlines()]
    if arg != 'all':
        rows = rows[:int(arg)]
    rep = {'swift_version': swift.__version__, 'source': str(SRC),
           'source_sha256': hashlib.sha256(SRC.read_bytes()).hexdigest(), 'n': 0, 'ok': 0, 'fail': [],
           'loss_tokens_low': 0, 'loss_tokens_medium': 0, 'tokens_low': 0, 'tokens_medium': 0}
    out = open(OUT / ('train_a1_medium.jsonl' if arg == 'all' else 'train_a1_medium_sample.jsonl'), 'w')
    for k, s in enumerate(rows):
        assert s['chat_template_kwargs'] == {'enable_thinking': True, 'reasoning_effort': 'low'}, s['chat_template_kwargs']
        med = json.loads(json.dumps(s))
        med['chat_template_kwargs'] = {'enable_thinking': True, 'reasoning_effort': 'medium'}
        ids_l, m_l = enc(tmpl, s)
        ids_m, m_m = enc(tmpl, med)
        res = {'row': k, 'n_low': len(ids_l), 'n_medium': len(ids_m)}
        # check 2/3: one contiguous removed block
        i = next((j for j, (a, b) in enumerate(zip(ids_l, ids_m)) if a != b), len(ids_m))
        d = len(ids_l) - len(ids_m)
        block = ids_l[i:i + d]
        res['removed_text'] = tok.decode(block)
        res['block_ok'] = (d > 0 and ids_l[:i] + ids_l[i + d:] == ids_m
                           and res['removed_text'].strip() == LOW_INSTR and sum(m_l[i:i + d]) == 0)
        res['mask_ok'] = m_l[:i] + m_l[i + d:] == m_m
        # check 1: HF reference render at medium
        hf_text = tok.apply_chat_template(hf_messages(med), tools=med.get('tools'), tokenize=False,
                                          **med['chat_template_kwargs'])
        hf_ids = tok(hf_text, add_special_tokens=False)['input_ids']
        res['hf_prefix_ok'] = hf_ids[:len(ids_m)] == ids_m
        res['hf_has_low_instr'] = LOW_INSTR in hf_text
        good = res['block_ok'] and res['mask_ok'] and res['hf_prefix_ok'] and not res['hf_has_low_instr']
        rep['n'] += 1
        if good:
            rep['ok'] += 1
            rep['loss_tokens_low'] += sum(m_l); rep['loss_tokens_medium'] += sum(m_m)
            rep['tokens_low'] += len(ids_l); rep['tokens_medium'] += len(ids_m)
            out.write(json.dumps(med, ensure_ascii=False) + '\n')
        else:
            rep['fail'].append(res)
        if rep['n'] <= 3 or not good:
            print(json.dumps(res, ensure_ascii=False)[:600])
    out.close()
    (OUT / f'convert_a1_medium_report_{arg}.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print('SUMMARY', rep['n'], 'ok', rep['ok'], 'fail', len(rep['fail']), 'loss_tokens low/medium',
          rep['loss_tokens_low'], rep['loss_tokens_medium'], 'tokens low/medium', rep['tokens_low'], rep['tokens_medium'])


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '3')
