"""S2b parity: our frozen (vLLM-equivalent) ids vs (a) the stock ms-swift template the
S2 trainer actually used and (b) the vLLM serving env's own render.

CPU only. Usage (venv-mcore):
  PYTHONPATH=$W/pylibs venv-mcore/bin/python scripts/s2b_parity.py [--stock] [--serving]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
sys.path.insert(0, str(Path(__file__).resolve().parent))
W = Path(RETRACE_SHARE + '/online-arms')
MODEL = RETRACE_SHARE + '/models/Qwen/Qwen3.8-27B'
SERVING = RETRACE_SHARE + '/stage3-endtoend/runtime/envs-attempt1/serving/bin/python'

rows = [json.loads(l) for l in (W / 's2b/data/rows_topk.jsonl').read_text().splitlines() if l.strip()]
index = json.loads((W / 's2b/frozen/index.json').read_text())


def diff(a, b):
    n = min(len(a), len(b))
    d = next((i for i in range(n) if a[i] != b[i]), None)
    return {'len_a': len(a), 'len_b': len(b), 'delta': len(a) - len(b), 'first_divergence': d}


def stock_render(row):
    """The template the S2 trainer used (swift backend, template_type qwen3_8, defaults)."""
    from swift import get_processor, get_template
    processor = get_processor(MODEL)
    template = get_template(processor, template_type='qwen3_8', loss_scale='default',
                            add_non_thinking_prefix=True, template_backend='swift')
    template.set_mode('train')
    out = template.encode(dict(messages=row['messages'], tools=row['tools'],
                               chat_template_kwargs=row['chat_template_kwargs'],
                               add_eos=False), return_length=True)
    return list(out['input_ids']), out.get('labels')


def serving_render(row, view):
    """Render with the vLLM serving env's tokenizer + jinja chat template (vLLM's own path)."""
    msg = row['teacher_prompt'] if view == 'teacher' else None
    payload = json.dumps({'messages': row['messages'], 'tools': row['tools'],
                          'teacher_prompt': msg, 'reasoning_effort': 'medium'})
    code = '''
import json, sys
from transformers import AutoTokenizer
MODEL = %r
p = json.loads(sys.stdin.read())
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
msgs = p["messages"]
if p.get("teacher_prompt"):
    import copy
    msgs = copy.deepcopy(msgs)
    i = max(k for k, m in enumerate(msgs) if m["role"] == "user")
    msgs[i]["content"] = p["teacher_prompt"]
def norm(messages):
    import copy
    out = []
    for m in messages:
        m2 = dict(m)
        if m2.get("tool_calls"):
            tcs = []
            for tc in m2["tool_calls"]:
                tc2 = copy.deepcopy(tc)
                fn = tc2.get("function") or {}
                a = fn.get("arguments")
                if isinstance(a, str):
                    try:
                        fn["arguments"] = json.loads(a) if a.strip() else {}
                    except Exception:
                        pass
                tcs.append(tc2)
            m2["tool_calls"] = tcs
        out.append(m2)
    return out
txt = tok.apply_chat_template(norm(msgs), tools=p["tools"], add_generation_prompt=False,
                              tokenize=False, reasoning_effort=p["reasoning_effort"])
ids = tok(txt, add_special_tokens=False)["input_ids"]
print(json.dumps(ids))
''' % MODEL
    r = subprocess.run([SERVING, '-c', code], input=payload, capture_output=True, text=True,
                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'TRANSFORMERS_OFFLINE': '1'})
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-800:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', action='store_true')
    ap.add_argument('--serving', action='store_true')
    a = ap.parse_args()
    res = []
    for row in rows:
        e = index[row['uid']]
        r = {'uid': row['uid'], 'tag': row['tag'], 'n_student': e['n_student'],
             'n_teacher': e['n_teacher'], 'n_loss': e['n_loss']}
        if a.serving:
            for view in ('student', 'teacher'):
                try:
                    ids = serving_render(row, view)
                except Exception as ex:
                    r[f'serving_{view}_error'] = str(ex)[-400:]
                    continue
                r[f'serving_{view}'] = diff(ids, e[f'{view}_ids'])
                r[f'serving_{view}_identical'] = ids == e[f'{view}_ids']
        if a.stock:
            ids, labels = stock_render(row)
            r['stock_swift'] = diff(ids, e['student_ids'])
            r['stock_n_loss'] = sum(1 for x in (labels or []) if x != -100)
            r['stock_n_tokens'] = len(ids)
            n_u = sum(1 for x in row['messages'] if x.get('role') == 'user')
            n_merge = sum(1 for m1, m2 in zip(row['messages'], row['messages'][1:])
                          if m1['role'] == m2['role'] == 'user')
            r['row_n_user'] = n_u
            r['row_n_consecutive_user_pairs'] = n_merge
            r['row_n_assistant'] = sum(1 for x in row['messages'] if x.get('role') == 'assistant')
        res.append(r)
        print(json.dumps(r, ensure_ascii=False))
    (W / 's2b/logs/s2b_parity.json').write_text(json.dumps(res, indent=1))
    print('written', W / 's2b/logs/s2b_parity.json')


if __name__ == '__main__':
    main()
