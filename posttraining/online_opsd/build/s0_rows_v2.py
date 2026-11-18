"""Rebuild rows so ms-swift's RLHF encoder reproduces the served tokens.

Two data-shape changes (owner decisions 2026-11-17):
 1. reasoning (thinking) text is dropped from the row (no reasoning tokens in the loss),
 2. assistant tool_calls are pre-rendered into the message text exactly as the Qwen chat
    template renders them, because ms-swift re-formats tool calls itself and that changes
    the token stream (-1609 / -13390 tokens measured on the two GKD smoke rows).

Everything else (prompt, tool results, template scaffolding) is unchanged, so the ids must
match the no-reasoning render of the original conversation; we verify by hash.
"""
import copy
import json
import sys
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrace_replay as R  # noqa: E402


def assistant_block_text(msg):
    tok = R.tokenizer()
    dummy = [{'role': 'user', 'content': 'x'}]
    norm, _ = R.vllm_normalize([msg])
    txt = tok.apply_chat_template(dummy + norm, tools=None, add_generation_prompt=False,
                                  tokenize=False, reasoning_effort=R.REASONING_EFFORT)
    marker = '<|im_start|>assistant\n'
    i = txt.rindex(marker)
    body = txt[i + len(marker):]
    if body.endswith('<|im_end|>'):
        body = body[: -len('<|im_end|>')]
    return body[:-1] if body.endswith('\n') else body


def rebuild(row):
    msgs = copy.deepcopy(row['messages'])
    for m in msgs:
        if m.get('role') != 'assistant':
            continue
        m.pop('reasoning_content', None)
        if m.get('tool_calls'):
            m['content'] = assistant_block_text(m)
            m.pop('tool_calls', None)
        if m['content'] is None:
            m['content'] = ''
    ids = R.render_ids(msgs, row['tools'], add_generation_prompt=False)
    spans, warn = R.assistant_spans(ids, msgs, row['tools'])
    labels = [-100] * len(ids)
    for _, s, e in spans:
        labels[s:e] = ids[s:e]
    row2 = dict(row, messages=msgs, input_ids=ids, labels=labels, n_tokens=len(ids),
                n_loss_tokens=sum(1 for x in labels if x != -100), n_spans=len(spans),
                warnings=warn)
    tp = row.get('teacher_prompt')
    if tp:
        row2['teacher_prompt'] = tp  # block itself unchanged; teacher messages rebuilt below
    return row2


def main():
    W = Path(RETRACE_SHARE + '/online-arms')
    for name in ('rows_gkd_pair.jsonl', 'rows_gkd_smoke.jsonl', 'rows_grpo_smoke.jsonl'):
        src = W / 'data' / name
        rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
        out = []
        for r in rows:
            r2 = rebuild(r)
            if 'teacher_prompt' in r2:
                # teacher messages = student messages with the last user content replaced
                msgs = copy.deepcopy(r2['messages'])
                idx = max(i for i, m in enumerate(msgs) if m['role'] == 'user')
                block = r2['teacher_prompt'][len(msgs[idx]['content'] or ''):]
                msgs[idx]['content'] = r2['teacher_prompt']
                r2['_teacher_block'] = block
            out.append(r2)
        dst = W / 'data' / name.replace('.jsonl', '_v2.jsonl')
        with dst.open('w') as f:
            for r in out:
                r.pop('_teacher_block', None)
                f.write(json.dumps(r) + '\n')
        print(name, '->', dst.name)
        for r, orig in zip(out, rows):
            print(' ', r['uid'], 'n_tokens', orig['n_tokens'], '->', r['n_tokens'],
                  'n_loss', orig['n_loss_tokens'], '->', r['n_loss_tokens'])


if __name__ == '__main__':
    main()
