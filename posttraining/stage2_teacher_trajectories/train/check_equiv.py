"""CPU-only equivalence check: ms-swift 4.6 qwen3_8 template vs ReTRACE A1 render (build_full_v1.py).
For N A1 rows: re-encode the stored messages with swift's template in train mode with
reasoning_effort=low, add_non_thinking_prefix=True, loss_scale=ignore_empty_think, and compare
  (1) input_ids  vs A1 tok/<uid>.json input_ids (full trajectory, deployment-identical render)
  (2) labels!=-100 / loss_scale>0 positions vs A1 full_v1 loss_mask (all assistant turns)
  (3) same vs the a2_v1 mask actually used for A1 training (segment-selected subset)
No model weights are loaded (tokenizer/processor only). Writes equivalence_report.json.
Usage: venv-swift/bin/python check_equiv.py [uid ...]"""
import json, sys, hashlib, traceback
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
S = Path(RETRACE_SHARE + '/stage2-agentic')
FULL = S / 'stage2_data/full_v1'
A2 = S / 'stage2_data/a2_v1'
MODEL = RETRACE_SHARE + '/models/Qwen/Qwen3.8-27B'
OUT = Path(__file__).resolve().parent


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main(uids):
    import swift
    from swift import get_processor, get_template
    proc = get_processor(MODEL, load_model=False) if 'load_model' in get_processor.__code__.co_varnames else get_processor(MODEL)
    tok = proc.tokenizer if hasattr(proc, 'tokenizer') else proc
    tmpl = get_template(proc, template_type='qwen3_8', loss_scale='ignore_empty_think',
                        add_non_thinking_prefix=True, enable_thinking=True,
                        max_length=None, truncation_strategy='raise')
    tmpl.set_mode('train')
    a2_index = {json.loads(l)['uid']: json.loads(l) for l in (A2 / 'INDEX.jsonl').read_text().splitlines()} \
        if (A2 / 'INDEX.jsonl').exists() else {}
    rep = {'swift_version': swift.__version__, 'template_type': 'qwen3_8', 'loss_scale': 'ignore_empty_think',
           'add_non_thinking_prefix': True, 'rows': []}
    for uid in uids:
        r = {'uid': uid}
        try:
            ref = json.loads((FULL / 'tok' / f'{uid}.json').read_text())
            m = json.loads((FULL / 'msgs' / f'{uid}.json').read_text())
            kw = m['chat_template_kwargs']
            assert kw == {'enable_thinking': True, 'reasoning_effort': 'low'}, kw
            msgs = []
            for x in m['messages']:
                x = dict(x)
                if x['role'] == 'assistant':
                    if 'tool_calls' in x and not x['tool_calls']:
                        x.pop('tool_calls')
                    for tc in x.get('tool_calls') or []:
                        a = tc['function'].get('arguments')
                        tc['function']['arguments'] = json.loads(a) if isinstance(a, str) and a else (a or {})
                msgs.append(x)
            row = {'messages': msgs, 'tools': m['tools'], 'chat_template_kwargs': kw}
            enc = tmpl.encode(row)
            ids = list(enc['input_ids'])
            labels = list(enc.get('labels') or [])
            ls = enc.get('loss_scale')
            sw_mask = [int(l != -100 and (ls is None or ls[i] > 0)) for i, l in enumerate(labels)]
            # labels are next-token-aligned the same way as input_ids in swift (shift happens in the loss)
            r.update(n_ref=len(ref['input_ids']), n_swift=len(ids), ids_equal=ids == ref['input_ids'],
                     ids_first_diff=first_diff(ids, ref['input_ids']))
            if r['ids_first_diff'] is not None:
                i = r['ids_first_diff']
                r['ctx_ref'] = tok.decode(ref['input_ids'][max(0, i - 30):i + 30])
                r['ctx_swift'] = tok.decode(ids[max(0, i - 30):i + 30])
            if r['ids_equal']:
                full_m = ref['loss_mask']
                r['mask_vs_full_v1_equal'] = sw_mask == full_m
                r['mask_vs_full_v1_first_diff'] = first_diff(sw_mask, full_m)
                r['n_loss_swift'] = sum(sw_mask)
                r['n_loss_full_v1'] = sum(full_m)
                if r['mask_vs_full_v1_first_diff'] is not None:
                    i = r['mask_vs_full_v1_first_diff']
                    r['mask_ctx'] = tok.decode(ids[max(0, i - 20):i + 20])
                    r['mask_at'] = {'swift': sw_mask[i], 'full_v1': full_m[i], 'token': tok.decode([ids[i]])}
                    # classify all diffs
                    only_sw = [i for i in range(len(ids)) if sw_mask[i] and not full_m[i]]
                    only_ref = [i for i in range(len(ids)) if full_m[i] and not sw_mask[i]]
                    from collections import Counter
                    r['only_swift_tokens'] = Counter(tok.decode([ids[i]]) for i in only_sw).most_common(8)
                    r['only_full_v1_tokens'] = Counter(tok.decode([ids[i]]) for i in only_ref).most_common(8)
                    r['n_only_swift'] = len(only_sw)
                    r['n_only_full_v1'] = len(only_ref)
                if ls is not None:
                    vals = sorted({float(v) for v in ls})
                    r['loss_scale_values'] = vals[:10]
        except Exception as e:
            r['error'] = f'{type(e).__name__}: {e}'
            r['tb'] = traceback.format_exc()[-1500:]
        rep['rows'].append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ('tb', 'ctx_ref', 'ctx_swift', 'mask_ctx')}, ensure_ascii=False))
    (OUT / 'equivalence_report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:] or ['rep2_128', 'rep1_174', 'rep2_177'])
