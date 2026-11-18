"""S2 rows: worst-case OPSD GKD smoke rows (reasoning KEPT in the loss).

Differences vs S0's rows_*_v2.jsonl:
  * the final assistant turn keeps its reasoning_content (worst case for loss-token count),
  * assistant tool_calls are still pre-rendered into the message text (the S1 fix that shrank
    the ms-swift/vLLM encoder delta from -1609/-13390 to -51 tokens on a 22k row),
  * one row per extreme: globally shortest and globally longest train trajectory.

Output: data/rows_gkd_s2_reason.jsonl, logs/s2_rows.json
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

W = Path(RETRACE_SHARE + '/online-arms')


def toolcall_body(msg):
    """Template rendering of an assistant message body without reasoning (content + tool calls)."""
    m = copy.deepcopy(msg)
    m.pop('reasoning_content', None)
    dummy = [{'role': 'user', 'content': 'x'}]
    norm, _ = R.vllm_normalize([m])
    txt = R.tokenizer().apply_chat_template(dummy + norm, tools=None, add_generation_prompt=False,
                                            tokenize=False, reasoning_effort=R.REASONING_EFFORT)
    marker = '<|im_start|>assistant\n'
    i = txt.rindex(marker)
    body = txt[i + len(marker):]
    if body.endswith('<|im_end|>'):
        body = body[:-len('<|im_end|>')]
    return body[:-1] if body.endswith('\n') else body


def pre_render_toolcalls(msgs):
    out = copy.deepcopy(msgs)
    n = 0
    for m in out:
        if m.get('role') != 'assistant' or not m.get('tool_calls'):
            continue
        m['content'] = toolcall_body(m)
        m.pop('tool_calls', None)
        n += 1
        if m.get('content') is None:
            m['content'] = ''
    return out, n


def render_row(msgs, tools):
    ids = R.render_ids(msgs, tools, add_generation_prompt=False)
    spans, warn = R.assistant_spans(ids, msgs, tools)
    labels = [-100] * len(ids)
    for _, s, e in spans:
        labels[s:e] = ids[s:e]
    return {'input_ids': ids, 'labels': labels, 'n_tokens': len(ids),
            'n_loss_tokens': sum(1 for x in labels if x != -100),
            'n_spans': len(spans), 'warnings': warn}


def main():
    lengths = {int(k): v for k, v in json.loads((W / 'logs' / 's0_lengths.json').read_text()).items()}
    priv = {}
    for line in (W / 'data' / 'privileged_routes.jsonl').read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            priv[int(r['idx'])] = r

    allt = [(e['n_tokens'], idx, e) for idx, v in lengths.items() for e in v]
    allt.sort(key=lambda x: x[0])
    # extremes must have a privileged route (OPSD needs teacher_prompt)
    short = next(t for t in allt if t[1] in priv)
    long_ = next(t for t in allt[::-1] if t[1] in priv)
    picks = [('shortest', short), ('longest', long_)]

    rows, ev = [], []
    for tag, (ntok, idx, ent) in picks:
        res, recs, n_raw = R.load_records(ent['path'])
        tools = recs[0]['request']['tools']
        msgs, meta = R.build_conversation(recs, include_final_reasoning=True)
        msgs_pr, n_prerender = pre_render_toolcalls(msgs)
        row = render_row(msgs_pr, tools)

        msgs_nr = copy.deepcopy(msgs_pr)
        for m in msgs_nr:
            m.pop('reasoning_content', None)
        row_nr = render_row(msgs_nr, tools)

        block = priv[idx]['block']
        t_msgs, user_idx = R.teacher_view(msgs_pr, block)
        t_row = render_row(t_msgs, tools)
        teacher_prompt = t_msgs[user_idx]['content']

        parity = R.parity_check(msgs_pr, tools, recs)
        tcheck = R.teacher_check(msgs_pr, t_msgs, tools, block)

        uid = f"{ent['arm']}/{ent['rep']}/{idx}"
        out = {
            'uid': uid, 'idx': idx, 'arm': ent['arm'], 'rep': ent['rep'], 'tag': tag,
            'source': ent['path'], 'messages': msgs_pr, 'tools': tools,
            'teacher_prompt': teacher_prompt,
            'chat_template_kwargs': {'reasoning_effort': R.REASONING_EFFORT, '_retrace_uid': uid},
            'input_ids': row['input_ids'], 'labels': row['labels'],
            'n_tokens': row['n_tokens'], 'n_loss_tokens': row['n_loss_tokens'],
            'n_loss_tokens_no_reasoning': row_nr['n_loss_tokens'],
            'n_tokens_no_reasoning': row_nr['n_tokens'],
            'n_spans': row['n_spans'], 'n_prerendered_toolcall_msgs': n_prerender,
            'warnings': row['warnings'],
            'teacher_n_tokens': t_row['n_tokens'], 'teacher_n_loss_tokens': t_row['n_loss_tokens'],
        }
        rows.append(out)
        ev.append({'tag': tag, 'uid': uid, 'n_tokens_census': ntok,
                   'n_tokens': row['n_tokens'], 'n_loss_tokens': row['n_loss_tokens'],
                   'n_loss_tokens_no_reasoning': row_nr['n_loss_tokens'],
                   'n_reasoning_tokens': row['n_tokens'] - row_nr['n_tokens'],
                   'n_prerendered_toolcall_msgs': n_prerender,
                   'teacher_n_tokens': t_row['n_tokens'],
                   'teacher_delta': t_row['n_tokens'] - row['n_tokens'],
                   'teacher_loss_ratio_vs_student': t_row['n_loss_tokens'] / row['n_loss_tokens'],
                   'parity': parity, 'teacher_check': tcheck, 'warnings': row['warnings'],
                   'meta': meta, 'n_wire': n_raw})

    with (W / 'data' / 'rows_gkd_s2_reason.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    (W / 'logs' / 's2_rows.json').write_text(json.dumps(ev, indent=1))
    for e in ev:
        print(json.dumps({k: e[k] for k in ('tag', 'uid', 'n_tokens', 'n_loss_tokens',
                                            'n_loss_tokens_no_reasoning', 'teacher_n_tokens',
                                            'teacher_delta')}))
        print('  parity', e['parity'])
        print('  teacher_check', e['teacher_check'])
        print('  n_reasoning_tokens', e['n_reasoning_tokens'])
    print('ROWS', len(rows))


if __name__ == '__main__':
    main()
