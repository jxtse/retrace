"""S2b: freeze the exact student/teacher token streams for 4 real train rows.

Why: ms-swift's own RLHF encoder re-renders the conversation (it drops the empty
```

` wrappers, merges two consecutive user messages, and re-formats tool calls), so
the trainer's token stream drifts from what vLLM actually served.  Instead of
patching the framework we freeze the ids here with the vLLM-equivalent renderer
(scripts/retrace_replay.py) and let a registered pretokenized template
(scripts/s2b_template.py) hand those ids to the trainer verbatim.

Rules (owner decisions):
  * reasoning tokens are NOT trained -> reasoning_content is dropped from every
    assistant message and, additionally, any ``...<｜end▁of▁thinking｜>`` interior is masked.
  * loss covers ALL assistant turns (not only the last round).
  * teacher view = same tokens with the privileged block appended to the LAST
    user message; the assistant bodies are spliced from the student stream so
    student and teacher share exactly the same response tokens (required by
    swift's extract_active label-count assert).

Outputs (under W/s2b/):
  frozen/index.json     uid -> {student_ids, student_labels, teacher_ids, teacher_labels, ...}
  data/rows_topk.jsonl  one row per length bucket: messages + teacher_prompt +
                        chat_template_kwargs{_retrace_uid}
  logs/s2b_build.json   per-row evidence
"""
import copy
import hashlib
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
TARGETS = [(22000, 'p22k'), (56000, 'p56k'), (75000, 'p75k'), (99800, 'p100k')]


def sha(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


def norm_tools(tools):
    """vLLM canonical tool form: {"type": "function", "function": {name, description, parameters}}.

    The wire log stores {"function": {description, name, parameters}, "type": ...}; vLLM
    re-serialises tools in its own key order before rendering, so without this the tool
    preamble tokenises differently (same length, different ids).
    """
    out = []
    for t in tools or []:
        fn = t.get('function') or {}
        out.append({'type': t.get('type', 'function'),
                    'function': {k: fn[k] for k in ('name', 'description', 'parameters') if k in fn}})
    return out


def think_spans(ids):
    """[(start, end_exclusive)] of every `<think>`..`<｜end▁of▁thinking｜>` interior (delimiters included)."""
    tok = R.tokenizer()
    op = tok.convert_tokens_to_ids('<think>')
    cl = tok.convert_tokens_to_ids('<｜end▁of▁thinking｜>')
    out, start = [], None
    for i, x in enumerate(ids):
        if x == op:
            start = i
        elif x == cl and start is not None:
            out.append((start, i + 1))
            start = None
    return out


def masks_from_spans(ids, spans):
    labels = [-100] * len(ids)
    for _, s, e in spans:
        labels[s:e] = ids[s:e]
    n_before = sum(1 for x in labels if x != -100)
    for s, e in think_spans(ids):
        for i in range(max(s, 0), min(e, len(labels))):
            labels[i] = -100
    return labels, n_before


def splice_bodies(t_ids, t_spans, s_ids, s_spans):
    """Replace every assistant body in the teacher stream with the student's exact ids."""
    assert len(t_spans) == len(s_spans), (len(t_spans), len(s_spans))
    out, prev_t, prev_s = [], 0, 0
    for (_, ts, te), (_, ss, se) in zip(t_spans, s_spans):
        out += t_ids[prev_t:ts]
        out += s_ids[ss:se]
        prev_t, prev_s = te, se
    out += t_ids[prev_t:]
    return out


def build_one(idx, ent, priv, tag):
    recs, n_raw = R.load_records(ent['path'])[1:]
    tools = recs[0]['request']['tools']
    msgs, meta = R.build_conversation(recs, include_final_reasoning=False)
    # reasoning must be gone everywhere
    n_reasoning_msgs = sum(1 for m in msgs if m.get('reasoning_content'))
    tools = norm_tools(tools)
    s_ids = R.render_ids(msgs, tools, add_generation_prompt=False)
    s_spans, warn = R.assistant_spans(s_ids, msgs, tools)
    s_labels, n_before = masks_from_spans(s_ids, s_spans)

    block = priv[idx]['block']
    t_msgs, user_idx = R.teacher_view(msgs, block)
    teacher_prompt = t_msgs[user_idx]['content']
    t_ids = R.render_ids(t_msgs, tools, add_generation_prompt=False)
    t_spans, twarn = R.assistant_spans(t_ids, t_msgs, tools)
    t_ids = splice_bodies(t_ids, t_spans, s_ids, s_spans)
    # re-locate spans on the spliced teacher stream (scaffolding unchanged, bodies replaced)
    t_spans2, _ = R.assistant_spans(t_ids, t_msgs, tools)
    t_labels, t_n_before = masks_from_spans(t_ids, t_spans2)

    n_s_loss = sum(1 for x in s_labels if x != -100)
    n_t_loss = sum(1 for x in t_labels if x != -100)
    assert n_s_loss == n_t_loss, (n_s_loss, n_t_loss)
    # every assistant body token shared?
    bodies_equal = all(s_ids[ss:se] == t_ids[ts:te] for (_, ts, te), (_, ss, se) in zip(t_spans2, s_spans))

    uid = f"{ent['arm']}/{ent['rep']}/{idx}"
    cu = max(i for i, m in enumerate(msgs) if m['role'] == 'user')
    entry = {
        'uid': uid, 'tag': tag, 'idx': idx, 'arm': ent['arm'], 'rep': ent['rep'],
        'source': ent['path'],
        'student_ids': s_ids, 'student_labels': s_labels,
        'teacher_ids': t_ids, 'teacher_labels': t_labels,
        'n_student': len(s_ids), 'n_teacher': len(t_ids),
        'n_loss': n_s_loss, 'n_loss_with_think_wrappers': n_before,
        'n_spans': len(s_spans), 'student_ids_sha256': sha(s_ids),
        'teacher_ids_sha256': sha(t_ids),
        'last_user_len_student': len(msgs[cu].get('content') or ''),
        'last_user_len_teacher': len(teacher_prompt or ''),
        'block_sha256': hashlib.sha256(block.encode('utf-8')).hexdigest(),
    }
    row = {
        'uid': uid, 'idx': idx, 'arm': ent['arm'], 'rep': ent['rep'], 'tag': tag,
        'source': ent['path'], 'messages': msgs, 'tools': tools,
        'teacher_prompt': teacher_prompt,
        'chat_template_kwargs': {'reasoning_effort': R.REASONING_EFFORT, '_retrace_uid': uid},
        'n_tokens': len(s_ids), 'n_loss_tokens': n_s_loss,
    }
    ev = {
        'tag': tag, 'uid': uid, 'census_n_tokens': ent.get('n_tokens', ent.get('census')),
        'n_student': len(s_ids), 'n_teacher': len(t_ids),
        'teacher_delta': len(t_ids) - len(s_ids),
        'n_loss': n_s_loss, 'n_loss_with_think_wrappers': n_before,
        'n_spans': len(s_spans), 'n_assistant_msgs': sum(1 for m in msgs if m['role'] == 'assistant'),
        'n_reasoning_msgs_left': n_reasoning_msgs, 'n_block_tokens': len(t_ids) - len(s_ids),
        'bodies_equal': bodies_equal,
        'warnings': warn + twarn, 'meta': meta, 'n_wire': n_raw,
    }
    print(json.dumps({k: ev[k] for k in ('tag', 'uid', 'n_student', 'n_teacher', 'n_loss',
                                         'n_loss_with_think_wrappers', 'n_spans',
                                         'bodies_equal', 'n_reasoning_msgs_left')}))
    if ev['warnings']:
        print('  warnings', ev['warnings'][:3])
    return entry, row, ev


def main():
    buckets = json.loads((W / 'logs' / 's0_lengths.json').read_text())
    priv = {int(json.loads(l)['idx']): json.loads(l)
            for l in (W / 'data' / 'privileged_routes.jsonl').read_text().splitlines() if l.strip()}
    picks_path = W / 's2b' / 'logs' / 'picks.json'
    if picks_path.exists():  # chosen by s2b_census.py on the reasoning-free length
        picks = [(p['tag'], (p['n_nr'], p['idx'], p)) for p in json.loads(picks_path.read_text())]
    else:
        allt = [(e['n_tokens'], idx, e) for idx, v in buckets.items() for e in v]
        picks, used = [], set()
        for target, tag in TARGETS:
            cand = [t for t in allt if int(t[1]) in priv and (int(t[1]), t[2]['rep']) not in used]
            cand.sort(key=lambda t: abs(t[0] - target))
            picks.append((tag, cand[0]))
            used.add((int(cand[0][1]), cand[0][2]['rep']))

    (W / 's2b' / 'frozen').mkdir(parents=True, exist_ok=True)
    (W / 's2b' / 'data').mkdir(parents=True, exist_ok=True)
    (W / 's2b' / 'logs').mkdir(parents=True, exist_ok=True)
    index, rows, evs = {}, [], []
    for tag, (ntok, idx, ent) in picks:
        entry, row, ev = build_one(int(idx), ent, priv, tag)
        index[entry['uid']] = entry
        rows.append(row)
        evs.append(ev)
    (W / 's2b' / 'frozen' / 'index.json').write_text(json.dumps(index))
    with (W / 's2b' / 'data' / 'rows_topk.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    # single-row datasets, each row duplicated 4x (one microbatch per copy)
    for r in rows:
        p = W / 's2b' / 'data' / f'ds_{r["tag"]}.jsonl'
        with p.open('w') as f:
            f.write((json.dumps(r) + '\n') * 4)
    (W / 's2b' / 'logs' / 's2b_build.json').write_text(json.dumps(evs, indent=1))
    print('OK rows', len(rows), 'index_bytes', (W / 's2b' / 'frozen' / 'index.json').stat().st_size)


if __name__ == '__main__':
    main()
