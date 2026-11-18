"""OPSD row builder: native-harness trajectory -> frozen student/teacher token streams.

Same rules as S2b (scripts/s2b_build.py), generalised to N targets per step:
  * reasoning tokens are NOT trained: reasoning_content is dropped from the assistant
    messages and any `<think>`..`<|end▁of▁thinking｜>` interior is masked in the labels.
  * the loss covers every assistant turn.
  * teacher view = the same tokens with the privileged block appended to the LAST user
    message; assistant bodies are spliced from the student stream so student and teacher
    share exactly the same response tokens (swift asserts equal label counts).
  * tokens come from our vLLM-equivalent renderer (retrace_replay.py), not from the
    framework's re-render, and are handed to the trainer by the registered
    pretokenized template retrace_opsd_ptok_v1.
"""
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
sys.path.insert(0, RETRACE_SHARE + '/online-arms/scripts')
import retrace_replay as R


def sha(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


def norm_tools(tools):
    """vLLM canonical tool form (same key order vLLM uses when it re-serialises tools)."""
    out = []
    for t in tools or []:
        fn = t.get('function') or {}
        out.append({'type': t.get('type', 'function'),
                    'function': {k: fn[k] for k in ('name', 'description', 'parameters') if k in fn}})
    return out


def think_spans(ids):
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
    assert len(t_spans) == len(s_spans), (len(t_spans), len(s_spans))
    out, prev_t, prev_s = [], 0, 0
    for (_, ts, te), (_, ss, se) in zip(t_spans, s_spans):
        out += t_ids[prev_t:ts]
        out += s_ids[ss:se]
        prev_t, prev_s = te, se
    out += t_ids[prev_t:]
    return out


def build_one(uid, result_path, block, meta=None):
    """One trajectory -> (index entry, dataset row, evidence)."""
    res, recs, n_raw = R.load_records(result_path)
    if not recs:
        raise RuntimeError(f'{uid}: no canonical wire records in {result_path}')
    tools = recs[0]['request']['tools']
    msgs, cmeta = R.build_conversation(recs, include_final_reasoning=False)
    n_reasoning_msgs = sum(1 for m in msgs if m.get('reasoning_content'))
    tools = norm_tools(tools)
    s_ids = R.render_ids(msgs, tools, add_generation_prompt=False)
    s_spans, warn = R.assistant_spans(s_ids, msgs, tools)
    s_labels, n_before = masks_from_spans(s_ids, s_spans)

    t_msgs, user_idx = R.teacher_view(msgs, block)
    teacher_prompt = t_msgs[user_idx]['content']
    t_ids = R.render_ids(t_msgs, tools, add_generation_prompt=False)
    t_spans, twarn = R.assistant_spans(t_ids, t_msgs, tools)
    t_ids = splice_bodies(t_ids, t_spans, s_ids, s_spans)
    t_spans2, _ = R.assistant_spans(t_ids, t_msgs, tools)
    t_labels, _ = masks_from_spans(t_ids, t_spans2)

    n_s_loss = sum(1 for x in s_labels if x != -100)
    n_t_loss = sum(1 for x in t_labels if x != -100)
    assert n_s_loss == n_t_loss, (uid, n_s_loss, n_t_loss)
    bodies_equal = all(s_ids[ss:se] == t_ids[ts:te]
                       for (_, ts, te), (_, ss, se) in zip(t_spans2, s_spans))
    cu = max(i for i, m in enumerate(msgs) if m['role'] == 'user')
    entry = {
        'uid': uid, 'idx': int(res['idx']), 'source': str(result_path),
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
        'uid': uid, 'idx': int(res['idx']), 'source': str(result_path),
        'messages': msgs, 'tools': tools, 'teacher_prompt': teacher_prompt,
        'chat_template_kwargs': {'reasoning_effort': R.REASONING_EFFORT, '_retrace_uid': uid},
        'n_tokens': len(s_ids), 'n_loss_tokens': n_s_loss,
    }
    ev = {
        'uid': uid, 'idx': int(res['idx']), 'result': str(result_path),
        'harness_success': bool(res.get('success')), 'iterations': res.get('iterations'),
        'wall_seconds': res.get('wall_seconds'), 'census_n_tokens': None,
        'n_student': len(s_ids), 'n_teacher': len(t_ids),
        'teacher_delta': len(t_ids) - len(s_ids),
        'n_loss': n_s_loss, 'n_loss_with_think_wrappers': n_before,
        'n_spans': len(s_spans), 'n_reasoning_msgs_left': n_reasoning_msgs,
        'bodies_equal': bodies_equal, 'warnings': (warn + twarn)[:5],
        'n_wire': n_raw, 'finish_reason': cmeta.get('finish_reason'),
        'final_content_len': cmeta.get('final_content_len'),
        'final_reasoning_len': cmeta.get('final_reasoning_len'),
        'final_tool_calls': cmeta.get('final_tool_calls'),
        'meta': meta or {},
    }
    return entry, row, ev


def write_step(outdir, built, gbs=16):
    """Write index.json + ds.jsonl (+ padding note) for one training step."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    entries = {e['uid']: e for e, _, _ in built}
    rows = [r for _, r, _ in built]
    padded = []
    while len(rows) < gbs:
        src = min(rows, key=lambda r: r['n_tokens'])
        dup = dict(src)
        dup['padding_copy_of'] = src['uid']
        rows.append(dup)
        padded.append(src['uid'])
    assert len(rows) == gbs
    (outdir / 'index.json').write_text(json.dumps(entries))
    with (outdir / 'ds.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    ev = [e for _, _, e in built]
    (outdir / 'build.json').write_text(json.dumps({
        'n_built': len(built), 'n_rows': len(rows), 'gbs': gbs, 'padded_with': padded,
        'max_n_tokens': max(e['n_student'] for e in ev) if ev else None,
        'max_n_teacher': max(e['n_teacher'] for e in ev) if ev else None,
        'total_loss_tokens': sum(e['n_loss'] for e in ev),
        'per_row': ev,
    }, indent=1, ensure_ascii=False))
    return dict(n_built=len(built), n_rows=len(rows), gbs=gbs, padded=padded,
                max_n_tokens=max((e['n_student'] for e in ev), default=None),
                max_n_teacher=max((e['n_teacher'] for e in ev), default=None),
                total_loss_tokens=sum(e['n_loss'] for e in ev), evidence=ev)
