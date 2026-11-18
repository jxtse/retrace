"""S2b: server-side /tokenize parity against the frozen ids (vLLM 0.29, running server).

The offline check (s2b_parity.py) already renders with the serving env's tokenizer.
This one asks the running server, so the request processor path vLLM actually uses is
exercised.  chat_template_kwargs must be forwarded: the chat template injects the
`reasoning_effort` block into the system prompt, and dropping it shifts the prefix.

Usage: serving/bin/python scripts/s2b_tokenize_parity.py <port>
"""
import json
import sys
import urllib.request


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
W = RETRACE_SHARE + '/online-arms'
port = sys.argv[1] if len(sys.argv) > 1 else '8032'
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
rows = [json.loads(l) for l in open(f'{W}/s2b/data/rows_topk.jsonl') if l.strip()]
idx = json.load(open(f'{W}/s2b/frozen/index.json'))


def tokenize(body):
    req = urllib.request.Request(f'http://127.0.0.1:{port}/tokenize', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with op.open(req, timeout=300) as r:
        d = json.load(r)
    return d.get('tokens') or d.get('token_ids')


out = []
for row in rows:
    res = {'uid': row['uid'], 'tag': row['tag'], 'n_frozen': idx[row['uid']]['n_student']}
    for label, extra in (('with_ctk', {'chat_template_kwargs': {'reasoning_effort': 'medium'}}),
                         ('no_ctk', {})):
        body = {'model': 'qwen3.8-27b-s2bopsd', 'messages': row['messages'],
                'tools': row['tools'], 'add_generation_prompt': False, **extra}
        try:
            got = tokenize(body)
        except Exception as e:
            res[label] = {'error': str(e)[-300:]}
            continue
        exp = idx[row['uid']]['student_ids']
        res[label] = {'n_vllm': len(got), 'n_frozen': len(exp), 'identical': got == exp,
                      'first_divergence': next((i for i in range(min(len(got), len(exp)))
                                                if got[i] != exp[i]), None)}
    out.append(res)
    print(json.dumps(res, ensure_ascii=False))
json.dump(out, open(f'{W}/s2b/logs/s2b_tokenize_parity.json', 'w'), indent=1)
