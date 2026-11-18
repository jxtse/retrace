"""ReTRACE GRPO/OPSD replay core (our own code; no framework edits).

Wire log (native harness, vLLM OpenAI streaming) -> one training row per trajectory:
  * exact token ids of the rendered conversation (vLLM-equivalent rendering),
  * completion loss mask over every model-generated token (reasoning/content/tool-call),
  * an OPSD teacher view that appends the privileged block to the *last user message*.

CPU only: tokenizer + chat template. No torch forward.
"""
import copy
import hashlib
import json
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
MODEL_DIR = RETRACE_SHARE + '/models/Qwen/Qwen3.8-27B'
REASONING_EFFORT = 'medium'
PRIV_PREFIX = '\n\n<privileged_audit_context>\n'
PRIV_SUFFIX = '\n</privileged_audit_context>\n'

_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        _TOK.im_end_id = _TOK.convert_tokens_to_ids('<|im_end|>')
    return _TOK


def load_records(result_path):
    """Return (canonical_ok_records, n_raw_records) for a native-harness result JSON."""
    res = json.loads(Path(result_path).read_text())
    wl = Path(res['session_dir']) / 'conversation' / 'wire_log.jsonl'
    raw = [json.loads(l) for l in wl.read_text().splitlines() if l.strip()]
    ok = [r for r in raw if r.get('outcome') == 'ok' and r.get('status_code') == 200]
    return res, ok, len(raw)


def vllm_normalize(messages):
    """Reproduce vLLM's tool-call normalization: JSON-string arguments -> mapping.

    Without this the Qwen chat template raises (``tool_call.arguments|items``).
    Returns (messages, n_parsed).
    """
    out, n_parsed = [], 0
    for m in messages:
        m2 = dict(m)
        if m2.get('tool_calls'):
            tcs = []
            for tc in m2['tool_calls']:
                tc2 = copy.deepcopy(tc)
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


def render_text(messages, tools, add_generation_prompt=True):
    tok = tokenizer()
    norm, _ = vllm_normalize(messages)
    return tok.apply_chat_template(
        norm, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=False,
        reasoning_effort=REASONING_EFFORT)


def render_ids(messages, tools, add_generation_prompt=True):
    return tokenizer()(render_text(messages, tools, add_generation_prompt),
                       add_special_tokens=False)['input_ids']


def build_conversation(recs, include_final_reasoning=True):
    """Canonical chain -> (messages, meta).

    The harness keeps prior assistant turns WITHOUT reasoning_content (verified: 0/184062
    request-side assistant messages carry it), so the served context per turn contains
    content + tool_calls only. The final response's reasoning is available from the wire
    log and is kept as the tail of the last assistant message (it was generated).
    """
    msgs = copy.deepcopy(recs[-1]['request']['messages'])
    final = recs[-1]['response']['choices'][0]['message']
    last = {'role': 'assistant', 'content': final.get('content') or ''}
    if include_final_reasoning and final.get('reasoning_content'):
        last['reasoning_content'] = final['reasoning_content']
    if final.get('tool_calls'):
        last['tool_calls'] = final['tool_calls']
    msgs.append(last)
    meta = {
        'finish_reason': recs[-1]['response']['choices'][0].get('finish_reason'),
        'final_content_len': len(final.get('content') or ''),
        'final_reasoning_len': len(final.get('reasoning_content') or ''),
        'final_tool_calls': len(final.get('tool_calls') or []),
        'n_messages': len(msgs),
    }
    return msgs, meta


def assistant_spans(ids, messages, tools, include_eos=False):
    """[(msg_idx, start, end_exclusive)] of model-generated tokens in the rendered conversation.

    start = end of the template's generation prompt for that turn; end = position of the
    turn's ``<|im_end|>`` terminator (excluded unless include_eos).
    """
    tok = tokenizer()
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    spans, warnings = [], []
    for k, m in enumerate(messages):
        if m['role'] != 'assistant':
            continue
        pre = render_ids(messages[:k], tools, add_generation_prompt=True)
        start = len(pre)
        j = start
        while j < len(ids) and ids[j] != im_end:
            j += 1
        if j >= len(ids):
            warnings.append(f'msg {k}: no <|im_end|> terminator found after {start}')
            end = len(ids)
        else:
            end = j + 1 if include_eos else j
        spans.append((k, start, end))
    return spans, warnings


def build_row(messages, tools, include_reasoning=True, include_eos=False):
    ids = render_ids(messages, tools, add_generation_prompt=False)
    spans, warnings = assistant_spans(ids, messages, tools, include_eos=include_eos)
    labels = [-100] * len(ids)
    for _, s, e in spans:
        labels[s:e] = ids[s:e]
    n_loss = sum(1 for x in labels if x != -100)
    return {
        'input_ids': ids, 'labels': labels, 'spans': spans, 'n_tokens': len(ids),
        'n_loss_tokens': n_loss, 'warnings': warnings,
    }


def parity_check(messages, tools, recs, include_eos=False):
    """Independent parity path for the tail.

    Row = assembled conversation render. Independent path = served final prompt render
    (raw wire-log messages) + the final response body rendered by the template itself.
    Any difference is the tokenizer merge at the seam; we report its size.
    """
    row_ids = render_ids(messages, tools, add_generation_prompt=False)
    prompt_ids = render_ids(recs[-1]['request']['messages'], tools, add_generation_prompt=True)
    final = recs[-1]['response']['choices'][0]['message']
    last = {'role': 'assistant', 'content': final.get('content') or ''}
    if final.get('reasoning_content'):
        last['reasoning_content'] = final['reasoning_content']
    if final.get('tool_calls'):
        last['tool_calls'] = final['tool_calls']
    dummy = [{'role': 'user', 'content': ''}]
    pre = render_ids(dummy, None, add_generation_prompt=True)
    full = render_ids(dummy + [last], None, add_generation_prompt=False)
    tail = full[len(pre):]
    tok = tokenizer()
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    while tail and tail[-1] == im_end and not include_eos:
        tail = tail[:-1]
    combined = prompt_ids + tail
    div = next((i for i in range(min(len(row_ids), len(combined))) if row_ids[i] != combined[i]), None)
    return {
        'row_len': len(row_ids), 'prompt_len': len(prompt_ids), 'tail_len': len(tail),
        'combined_len': len(combined), 'prefix_equal': row_ids[:len(prompt_ids)] == prompt_ids,
        'first_divergence': div, 'len_delta': len(row_ids) - len(combined),
        'identical': row_ids == combined,
    }



def teacher_view(messages, block):
    """OPSD teacher view: append the privileged block to the LAST user message content."""
    msgs = copy.deepcopy(messages)
    idx = max(i for i, m in enumerate(msgs) if m['role'] == 'user')
    base = msgs[idx].get('content') or ''
    msgs[idx]['content'] = base + block
    return msgs, idx


def teacher_check(messages, teacher_msgs, tools, block):
    """Verify student/teacher encodings differ only in the privileged block and that the
    shared response tokens are identical."""
    ids = render_ids(messages, tools, add_generation_prompt=False)
    t_ids = render_ids(teacher_msgs, tools, add_generation_prompt=False)
    user_idx = max(i for i, m in enumerate(teacher_msgs) if m['role'] == 'user')
    pre = render_ids(messages[:user_idx], tools, add_generation_prompt=False)
    s_up = render_ids(messages[:user_idx + 1], tools, add_generation_prompt=False)
    t_up = render_ids(teacher_msgs[:user_idx + 1], tools, add_generation_prompt=False)
    block_ids = tokenizer()(block, add_special_tokens=False)['input_ids']
    head_ok = ids[:len(pre)] == t_ids[:len(pre)]
    inserted_ok = t_ids[len(pre):len(pre) + len(block_ids)] == block_ids
    tail_ok = ids[len(s_up):] == t_ids[len(t_up):]
    return {
        'last_user_idx': user_idx, 'head_identical': head_ok, 'block_inserted_verbatim': inserted_ok,
        'shared_response_identical': tail_ok, 'n_block_tokens': len(block_ids),
        'n_teacher_tokens': len(t_ids), 'n_student_tokens': len(ids),
        'delta': len(t_ids) - len(ids),
    }


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()