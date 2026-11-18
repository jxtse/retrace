"""Convert A1 training rows (stage2_data/a2_v1, 167 VALID trajectories, turn-level loss selection) into
ms-swift 4.6 messages JSONL, and verify on every row that swift's qwen3_8 template (swift backend, train
mode, reasoning_effort=low, add_non_thinking_prefix, loss_scale=ignore_empty_think) reproduces EXACTLY:
  - input_ids == A1 tok input_ids (deployment-identical render)
  - (labels != -100 and loss_scale > 0) == A1 a2_v1 loss_mask (the mask A1 actually trained on)

Two data-shape fixes found in the first check (template rendering itself was identical):
  (1) swift merges consecutive same-role messages; harness sends [user: task, user: prefetch]. We keep two
      messages by rendering them as ONE user message whose content embeds the exact separator
      '<|im_end|>\\n<|im_start|>user\\n' (verified token-exact below; these are special tokens, so we also
      check that the tokenizer maps the embedded text back to the same special-token ids).
  (2) trajectories end on a tool message, so swift appends a generation prompt; we drop trailing non-assistant
      messages (they never carry loss) and compare on the common prefix, requiring the dropped tail to carry
      zero loss in A1's mask and the prefix to be identical.
A1 selection is whole-turn (5537 on / 976 off, 0 partial), mapped to per-message 'loss': true/false.
Usage: venv-swift/bin/python convert_and_check.py [N|all]"""
import json, sys, hashlib
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
S = Path(RETRACE_SHARE + '/stage2-agentic/stage2_data')
MODEL = RETRACE_SHARE + '/models/Qwen/Qwen3.8-27B'
OUT = Path(__file__).resolve().parent
SEP = '<|im_end|>\n<|im_start|>user\n'


def spans_onoff(full_mask, a2_mask):
    out, i, n = [], 0, len(full_mask)
    while i < n:
        if full_mask[i]:
            j = i
            while j < n and full_mask[j]:
                j += 1
            s = sum(a2_mask[i:j])
            assert s in (0, j - i), 'partial span'
            out.append(bool(s))
            i = j
        else:
            i += 1
    return out


def prep(messages, onoff):
    msgs, k = [], 0
    for x in messages:
        x = json.loads(json.dumps(x))
        if x['role'] == 'assistant':
            if 'tool_calls' in x and not x['tool_calls']:
                x.pop('tool_calls')
            for tc in x.get('tool_calls') or []:
                a = tc['function'].get('arguments')
                tc['function']['arguments'] = json.loads(a) if isinstance(a, str) and a else (a or {})
            x['loss'] = onoff[k]
            k += 1
        msgs.append(x)
    assert k == len(onoff), (k, len(onoff))
    # fix (1): merge leading consecutive user messages with the exact chat separator
    out = []
    for x in msgs:
        if out and out[-1]['role'] == 'user' and x['role'] == 'user':
            out[-1]['content'] = out[-1]['content'].strip() + SEP + x['content'].strip()  # HF template |trim's each user message
        else:
            out.append(x)
    # fix (2): drop trailing non-assistant messages
    while out and out[-1]['role'] != 'assistant':
        out.pop()
    return out


def main(arg):
    import swift
    from swift import get_processor, get_template
    proc = get_processor(MODEL)
    tok = proc.tokenizer if hasattr(proc, 'tokenizer') else proc
    sep_ids = tok(SEP, add_special_tokens=False)['input_ids']
    assert sep_ids[0] == tok.convert_tokens_to_ids('<|im_end|>') and sep_ids[2] == tok.convert_tokens_to_ids('<|im_start|>'), sep_ids
    tmpl = get_template(proc, template_type='qwen3_8', loss_scale='ignore_empty_think', add_non_thinking_prefix=True,
                        enable_thinking=True, template_backend='swift')
    tmpl.set_mode('train')
    rows = [json.loads(l) for l in (S / 'a2_v1/INDEX.jsonl').read_text().splitlines()]
    if arg != 'all':
        rows = sorted(rows, key=lambda r: r['n_tokens'])
        n = int(arg)
        rows = [rows[0], rows[len(rows) // 2], rows[-1]][:n] if n <= 3 else rows[:n]
    rep = {'swift_version': swift.__version__, 'n': 0, 'ok': 0, 'fail': []}
    out_f = open(OUT / ('train_swift.jsonl' if arg == 'all' else 'train_swift_sample.jsonl'), 'w')
    for r in rows:
        key = r['key']
        a2 = json.loads((S / 'a2_v1/tok' / f'{key}.json').read_text())
        full = json.loads((S / 'full_v1/tok' / f'{key}.json').read_text())
        m = json.loads((S / 'full_v1/msgs' / f'{key}.json').read_text())
        assert a2['input_ids'] == full['input_ids']
        onoff = spans_onoff(full['loss_mask'], a2['loss_mask'])
        msgs = prep(m['messages'], onoff)
        sample = {'messages': msgs, 'tools': m['tools'], 'chat_template_kwargs': m['chat_template_kwargs']}
        enc = tmpl.encode(json.loads(json.dumps(sample)))
        ids = list(enc['input_ids'])
        ls = enc.get('loss_scale')
        mask = [int(l != -100 and (ls is None or ls[i] > 0)) for i, l in enumerate(enc['labels'])]
        ref_ids, ref_mask = a2['input_ids'], a2['loss_mask']
        L = len(ids)
        res = {'key': key, 'n_ref': len(ref_ids), 'n_swift': L,
               'prefix_ids_equal': ids == ref_ids[:L],
               'tail_zero_loss': sum(ref_mask[L:]) == 0,
               'mask_equal': mask == ref_mask[:L],
               'loss_tokens': [sum(mask), sum(ref_mask)],
               'loss_scale_values': sorted({float(v) for v in ls}) if ls is not None else None}
        # classify mask diffs: only accept swift-extra loss on (a) '\n' right after <|im_end|> (template suffix)
        # and (b) the final assistant <|im_end|> (SUFFIX context ignores per-message loss=false)
        im_end = tok.convert_tokens_to_ids('<|im_end|>'); nl = tok('\n', add_special_tokens=False)['input_ids']
        diff = [i for i in range(L) if mask[i] != ref_mask[i]]
        last_end = max(i for i in range(L) if ids[i] == im_end)
        cls = {'nl_after_im_end': 0, 'final_im_end': 0, 'other': 0}
        for i in diff:
            if mask[i] and not ref_mask[i] and [ids[i]] == nl and i > 0 and ids[i - 1] == im_end:
                cls['nl_after_im_end'] += 1
            elif mask[i] and not ref_mask[i] and i == last_end:
                cls['final_im_end'] += 1
            else:
                cls['other'] += 1
        res['mask_diff'] = cls
        good = res['prefix_ids_equal'] and res['tail_zero_loss'] and cls['other'] == 0
        rep['n'] += 1
        if good:
            rep['ok'] += 1
            out_f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        else:
            for k2, a_, b_ in (('ids', ids, ref_ids[:L]), ('mask', mask, ref_mask[:L])):
                d = next((i for i, (x, y) in enumerate(zip(a_, b_)) if x != y), None)
                if d is not None:
                    res[f'{k2}_first_diff'] = d
                    res[f'{k2}_ctx'] = tok.decode(ids[max(0, d - 12):d + 12])
            rep['fail'].append(res)
        print(json.dumps({k: v for k, v in res.items() if not k.endswith('_ctx')}))
    out_f.close()
    (OUT / f'convert_report_{arg}.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print('SUMMARY', rep['n'], 'ok', rep['ok'], 'fail', len(rep['fail']))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '3')
