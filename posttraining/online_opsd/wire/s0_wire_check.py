"""S0 wire-log renderer/analyzer: replicate vLLM chat rendering, check append-only + parity.

Reads S3-collect result JSONs. CPU only (no CUDA, tokenizer only).

Checks per trajectory:
  A. message-level append-only: request[i+1].messages extends request[i].messages by
     the previous assistant message + newly added tool/user messages, unchanged prefix.
  B. token-level prefix: tokens(request[i].messages) is a prefix of tokens(request[i+1].messages).
  C. prompt length parity: local render tokens == vLLM usage.prompt_tokens.
  D. response length parity: len(tokens(response text)) vs vLLM usage.completion_tokens.
"""
import json
import sys
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
MODEL = RETRACE_SHARE + '/models/Qwen/Qwen3.8-27B'
REASONING_EFFORT = 'medium'  # harness wire payloads carry reasoning_effort=medium


def load_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)


def vllm_normalize(messages):
    """Mimic vLLM: OpenAI tool_calls carry JSON-string arguments; the chat template wants a mapping."""
    out = []
    n_parsed = 0
    for m in messages:
        m2 = dict(m)
        if m2.get('tool_calls'):
            tcs = []
            for tc in m2['tool_calls']:
                tc2 = json.loads(json.dumps(tc))
                fn = tc2.get('function') or {}
                args = fn.get('arguments')
                if isinstance(args, str):
                    try:
                        fn['arguments'] = json.loads(args) if args.strip() else {}
                        n_parsed += 1
                    except json.JSONDecodeError:
                        pass
                tcs.append(tc2)
            m2['tool_calls'] = tcs
        out.append(m2)
    return out, n_parsed


def render_ids(tok, messages, tools, reasoning_effort=REASONING_EFFORT, add_generation_prompt=True):
    norm, _ = vllm_normalize(messages)
    text = tok.apply_chat_template(norm, tools=tools, add_generation_prompt=add_generation_prompt,
                                   tokenize=False, reasoning_effort=reasoning_effort)
    return tok(text, add_special_tokens=False)['input_ids']


def response_text(msg):
    """Text the model emitted for one assistant turn (reasoning + content + tool calls as template renders)."""
    parts = []
    if msg.get('reasoning_content'):
        parts.append(msg['reasoning_content'])
    if msg.get('content'):
        parts.append(msg['content'])
    return '\n'.join(parts) if len(parts) > 1 else (parts[0] if parts else '')


def canonical_records(wl_path):
    recs = [json.loads(l) for l in Path(wl_path).read_text().splitlines() if l.strip()]
    ok = [r for r in recs if r.get('outcome') == 'ok' and r.get('status_code') == 200]
    return ok, len(recs)


def check_trajectory(res_path, tok, verbose=True):
    res = json.loads(Path(res_path).read_text())
    wl = Path(res['session_dir']) / 'conversation' / 'wire_log.jsonl'
    recs, n_raw = canonical_records(wl)
    reqs = [r['request'] for r in recs]
    tools = reqs[0]['tools']
    info = {'result': str(res_path), 'n_wire': n_raw, 'n_ok': len(recs), 'checks': []}
    prev_tokens = None
    prev_msgs = None
    deltas = []
    for i, (r, req) in enumerate(zip(recs, reqs)):
        msgs = req['messages']
        # A: message-level append-only
        if prev_msgs is not None:
            n_prev = len(prev_msgs)
            same_prefix = json.dumps(msgs[:n_prev], sort_keys=False) == json.dumps(prev_msgs, sort_keys=False)
            added = [m['role'] for m in msgs[n_prev:]]
            if verbose:
                print(f'  T{i} prefix_equal={same_prefix} added_roles={added}')
            info['checks'].append({'turn': i, 'type': 'msg_prefix', 'ok': same_prefix, 'added_roles': added})
        # C: prompt token parity
        ids = render_ids(tok, msgs, tools)
        delta = r['response']['usage']['prompt_tokens'] - len(ids)
        deltas.append(delta)
        # B: token-level prefix
        if prev_tokens is not None:
            pref = ids[:len(prev_tokens)] == prev_tokens
            if verbose and not pref:
                j = next((k for k in range(min(len(ids), len(prev_tokens)) if False else len(prev_tokens))
                          if ids[k] != prev_tokens[k]), None)
                print(f'  T{i} token_prefix={pref} first_divergence={j}')
            info['checks'].append({'turn': i, 'type': 'tok_prefix', 'ok': pref})
        info['checks'].append({'turn': i, 'type': 'prompt_len', 'ok': delta == 0, 'delta': delta, 'n_local': len(ids)})
        prev_tokens, prev_msgs = ids, msgs
        # D: response token parity
        resp_msg = None
        rmsg = r['response']['choices'][0].get('message')
        if rmsg is not None:
            norm_msg = vllm_normalize([{'role': 'assistant', **rmsg}])[0][0]
            rt = response_text(rmsg)
            n_tok = len(tok(rt, add_special_tokens=False)['input_ids'])
            ct = r['response']['usage'].get('completion_tokens')
            info['checks'].append({'turn': i, 'type': 'resp_len', 'ok': n_tok == ct,
                                   'n_local': n_tok, 'vllm': ct})
    n_msg_ok = sum(1 for c in info['checks'] if c['type'] == 'msg_prefix' and c['ok'])
    n_msg = sum(1 for c in info['checks'] if c['type'] == 'msg_prefix')
    n_tok_ok = sum(1 for c in info['checks'] if c['type'] == 'tok_prefix' and c['ok'])
    n_tok = sum(1 for c in info['checks'] if c['type'] == 'tok_prefix')
    n_len_ok = sum(1 for c in info['checks'] if c['type'] == 'prompt_len' and c['ok'])
    n_len = sum(1 for c in info['checks'] if c['type'] == 'prompt_len')
    n_resp_ok = sum(1 for c in info['checks'] if c['type'] == 'resp_len' and c['ok'])
    n_resp = sum(1 for c in info['checks'] if c['type'] == 'resp_len')
    info['summary'] = {'msg_prefix': f'{n_msg_ok}/{n_msg}', 'tok_prefix': f'{n_tok_ok}/{n_tok}',
                       'prompt_len': f'{n_len_ok}/{n_len}', 'resp_len': f'{n_resp_ok}/{n_resp}',
                       'delta_min': min(deltas) if deltas else None, 'delta_max': max(deltas) if deltas else None}
    return info


if __name__ == '__main__':
    tok = load_tok()
    allres = []
    for p in sys.argv[1:]:
        print('===', p)
        info = check_trajectory(p, tok, verbose=True)
        print('SUMMARY', info['n_ok'], info['summary'])
        allres.append(info)
    Path('s0_wire_check.json').write_text(json.dumps(allres, indent=1))
