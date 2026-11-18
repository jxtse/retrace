"""S0 pipeline: append-only audit, token-length census, converter rows, parity + teacher proof.

Evidence files (remote work dir):
  logs/s0_append_only.json   message-level append-only / reasoning-retention / prompt-length delta
  logs/s0_lengths.json       full-context token census for all 260 S3 trajectories (130 targets)
  logs/s0_report.json        converter parity, loss-token counts, teacher-view proof
  data/rows_gkd_smoke.jsonl  shortest + longest train trajectory rows (with teacher_prompt)
  data/rows_grpo_smoke.jsonl 2 targets x 2 reps rows (fixture rewards, no teacher_prompt)
"""
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrace_replay as R  # noqa: E402

WORK = Path(RETRACE_SHARE + '/online-arms')
COLLECT = Path(RETRACE_SHARE + '/stage3-collect/arms')
SPLIT = Path(RETRACE_SHARE + '/stage2-agentic/split/SPLIT_STAGE2.json')
PRIV = WORK / 'data' / 'privileged_routes.jsonl'
ARMS = ('c0', 'c1')
REPS = ('rep1', 'rep2')


def index_trajectories():
    idx_map = defaultdict(list)
    for arm in ARMS:
        for rep in REPS:
            for p in sorted((COLLECT / arm / rep / 'runs').glob('*.json')):
                idx = int(p.stem.split('_')[0])
                idx_map[idx].append({'arm': arm, 'rep': rep, 'path': str(p)})
    return idx_map


def analyze_append_only(paths):
    """Message-level append-only check + reasoning retention + per-turn prompt-length delta."""
    tok = R.tokenizer()
    out = []
    for p in paths:
        res, recs, n_raw = R.load_records(p)
        tools = recs[0]['request']['tools']
        msg_ok = msg_total = 0
        delta_values = []
        resp_delta = []
        reasoning_in_requests = 0
        assistant_msgs = 0
        prev_msgs = None
        for r in recs:
            msgs = r['request']['messages']
            if prev_msgs is not None:
                msg_total += 1
                same = json.dumps(msgs[:len(prev_msgs)]) == json.dumps(prev_msgs)
                if same:
                    msg_ok += 1
                added = [m['role'] for m in msgs[len(prev_msgs):]]
                if added and added[0] != 'assistant':
                    out.append({'warning': 'non-assistant first appended', 'path': p})
            for m in msgs:
                if m['role'] == 'assistant':
                    assistant_msgs += 1
                    if m.get('reasoning_content'):
                        reasoning_in_requests += 1
            ids = R.render_ids(msgs, tools)
            delta_values.append(r['response']['usage']['prompt_tokens'] - len(ids))
            msg = r['response']['choices'][0].get('message') or {}
            body = ''
            if msg.get('reasoning_content'):
                body += msg['reasoning_content']
            body += msg.get('content') or ''
            resp_delta.append(r['response']['usage'].get('completion_tokens') - len(
                tok(body, add_special_tokens=False)['input_ids']))
            prev_msgs = msgs
        out.append({
            'path': p, 'label': res['label'], 'idx': res['idx'], 'n_wire': n_raw, 'n_ok': len(recs),
            'msg_prefix_ok': msg_ok, 'msg_prefix_total': msg_total,
            'prompt_len_delta_unique': sorted(set(delta_values)),
            'assistant_msgs_in_requests': assistant_msgs,
            'assistant_msgs_with_reasoning': reasoning_in_requests,
            'resp_len_delta_median': statistics.median(resp_delta) if resp_delta else None,
            'resp_len_delta_max': max(abs(x) for x in resp_delta) if resp_delta else None,
        })
    return out


def census(idx_map, train):
    """Full-context token count for every collected train trajectory (one render each)."""
    lengths = {}
    for idx in sorted(idx_map):
        if idx not in train:
            continue
        for ent in idx_map[idx]:
            res, recs, _ = R.load_records(ent['path'])
            msgs, meta = R.build_conversation(recs)
            ids = R.render_ids(msgs, recs[0]['request']['tools'], add_generation_prompt=False)
            lengths.setdefault(idx, []).append({'arm': ent['arm'], 'rep': ent['rep'],
                                                'path': ent['path'], 'n_tokens': len(ids),
                                                'finish_reason': meta['finish_reason'],
                                                'n_messages': meta['n_messages']})
    return lengths


def build_row(idx, ent, tools_route_block, include_reasoning=True, teacher=False):
    res, recs, _ = R.load_records(ent['path'])
    tools = recs[0]['request']['tools']
    msgs, meta = R.build_conversation(recs, include_final_reasoning=include_reasoning)
    row = R.build_row(msgs, tools, include_reasoning=include_reasoning)
    uid = f"{ent['arm']}/{ent['rep']}/{idx}"
    out = {
        'uid': uid, 'idx': idx, 'arm': ent['arm'], 'rep': ent['rep'], 'source': ent['path'],
        'messages': msgs, 'tools': tools,
        'chat_template_kwargs': {'reasoning_effort': R.REASONING_EFFORT, '_retrace_uid': uid},
        'input_ids': row['input_ids'], 'labels': row['labels'],
        'n_tokens': row['n_tokens'], 'n_loss_tokens': row['n_loss_tokens'],
        'n_spans': len(row['spans']), 'warnings': row['warnings'],
    }
    extra = {'parity': R.parity_check(msgs, tools, recs), 'meta': meta}
    if teacher:
        block = tools_route_block['block']
        t_msgs, user_idx = R.teacher_view(msgs, block)
        out['teacher_prompt'] = t_msgs[user_idx]['content']
        extra['teacher'] = R.teacher_check(msgs, t_msgs, tools, block)
        extra['last_user_idx'] = user_idx
        extra['n_messages'] = len(msgs)
        # teacher row labels (same span rule)
        t_ids = R.render_ids(t_msgs, tools, add_generation_prompt=False)
        t_spans, t_warn = R.assistant_spans(t_ids, t_msgs, tools)
        extra['teacher_n_loss_tokens'] = sum(e - s for _, s, e in t_spans)
        extra['teacher_warnings'] = t_warn
    # reasoning-token decomposition (only the final turn carries reasoning in the row)
    if include_reasoning:
        msgs_nr, _ = R.build_conversation(recs, include_final_reasoning=False)
        row_nr = R.build_row(msgs_nr, tools, include_reasoning=False)
        extra['n_tokens_without_reasoning'] = row_nr['n_tokens']
        extra['n_loss_tokens_without_reasoning'] = row_nr['n_loss_tokens']
        extra['n_reasoning_tokens'] = row['n_tokens'] - row_nr['n_tokens']
        extra['n_loss_tokens_all'] = row['n_loss_tokens']
    return out, extra


def main():
    split = json.loads(SPLIT.read_text())
    train = set(split['train_stage2'])
    idx_map = index_trajectories()
    priv = {int(r['idx']): r for r in (json.loads(l) for l in PRIV.read_text().splitlines() if l.strip())}

    # ---- 1. append-only / reasoning audit on a spread sample
    sample = []
    for idx in sorted(idx_map):
        if idx in train:
            sample.append(idx_map[idx][0]['path'])
    sample = sample[::max(1, len(sample) // 8)][:8]
    ao_path = WORK / 'logs' / 's0_append_only.json'
    if os.environ.get('RETRACE_REUSE') == '1' and ao_path.exists():
        ao = json.loads(ao_path.read_text())
    else:
        ao = analyze_append_only(sample)
        ao_path.write_text(json.dumps(ao, indent=1))
    agg = {
        'n_trajectories': len([a for a in ao if 'msg_prefix_ok' in a]),
        'msg_prefix_ok': sum(a['msg_prefix_ok'] for a in ao if 'msg_prefix_ok' in a),
        'msg_prefix_total': sum(a['msg_prefix_total'] for a in ao if 'msg_prefix_ok' in a),
        'prompt_len_delta_values': sorted({d for a in ao if 'prompt_len_delta_unique' in a
                                          for d in a['prompt_len_delta_unique']}),
        'assistant_msgs_in_requests': sum(a.get('assistant_msgs_in_requests', 0) for a in ao),
        'assistant_msgs_with_reasoning': sum(a.get('assistant_msgs_with_reasoning', 0) for a in ao),
        'resp_len_delta_median_range': [min(a['resp_len_delta_median'] for a in ao if a.get('resp_len_delta_median') is not None),
                                        max(a['resp_len_delta_median'] for a in ao if a.get('resp_len_delta_median') is not None)],
    }
    print('APPEND_ONLY', json.dumps(agg))

    # ---- 2. census
    len_path = WORK / 'logs' / 's0_lengths.json'
    if os.environ.get('RETRACE_REUSE') == '1' and len_path.exists():
        lengths = {int(k): v for k, v in json.loads(len_path.read_text()).items()}
    else:
        lengths = census(idx_map, train)
        len_path.write_text(json.dumps(lengths, indent=1))
    best = {idx: min(v, key=lambda e: e['n_tokens']) for idx, v in lengths.items()}
    all_len = [e['n_tokens'] for v in lengths.values() for e in v]
    print('CENSUS', json.dumps({'n_targets': len(lengths), 'n_traj': len(all_len),
                                'min': min(all_len), 'median': int(statistics.median(all_len)),
                                'p90': sorted(all_len)[int(0.9 * len(all_len))], 'max': max(all_len),
                                'over_100k': sum(1 for x in all_len if x > 100000),
                                'over_131072': sum(1 for x in all_len if x > 131072)}))
    shortest_idx = min(best, key=lambda i: best[i]['n_tokens'])
    longest_idx = max(best, key=lambda i: best[i]['n_tokens'])
    ordered = sorted(best, key=lambda i: best[i]['n_tokens'])
    median_idx = ordered[len(ordered) // 2]

    # ---- 3. rows
    gkd_targets = [i for i in (shortest_idx, median_idx, longest_idx)]
    grpo_targets = []
    for idx in sorted(lengths):
        if idx in priv and len(lengths[idx]) >= 2:
            grpo_targets.append(idx)
        if len(grpo_targets) == 2:
            break
    rows_gkd, evidence = [], []
    for idx in gkd_targets:
        ent = best[idx]
        row, extra = build_row(idx, ent, priv[idx], teacher=idx in priv)
        if 'teacher_prompt' not in row:
            continue
        rows_gkd.append(row)
        evidence.append({'kind': 'gkd', 'idx': idx, 'uid': row['uid'], **extra})
    rows_grpo = []
    for idx in grpo_targets:
        for ent in sorted(lengths[idx], key=lambda e: (e['arm'], e['rep']))[:2]:
            row, extra = build_row(idx, ent, priv.get(idx, {}), teacher=False)
            rows_grpo.append(row)
            evidence.append({'kind': 'grpo', 'idx': idx, 'uid': row['uid'], **extra})

    with (WORK / 'data' / 'rows_gkd_smoke.jsonl').open('w') as f:
        for r in rows_gkd:
            f.write(json.dumps(r) + '\n')
    with (WORK / 'data' / 'rows_grpo_smoke.jsonl').open('w') as f:
        for r in rows_grpo:
            f.write(json.dumps(r) + '\n')

    report = {
        'append_only': agg,
        'census': {'n_targets': len(lengths), 'n_traj': len(all_len), 'min': min(all_len),
                   'median': int(statistics.median(all_len)),
                   'p90': sorted(all_len)[int(0.9 * len(all_len))], 'max': max(all_len),
                   'over_100k': sum(1 for x in all_len if x > 100000),
                   'over_131072': sum(1 for x in all_len if x > 131072)},
        'selected': {'shortest': shortest_idx, 'median': median_idx, 'longest': longest_idx,
                     'gkd_targets_with_privileged_route': [i for i in gkd_targets if i in priv],
                     'grpo_targets': grpo_targets,
                     'lengths': {str(i): best[i]['n_tokens'] for i in gkd_targets}},
        'rows': evidence,
        'fixture_rewards': {r['uid']: (0.5 if i % 2 else 1.0) for i, r in enumerate(rows_grpo)},
    }
    (WORK / 'logs' / 's0_report.json').write_text(json.dumps(report, indent=1))
    for e in evidence:
        print(e['kind'], e['uid'], 'n_tokens', e.get('parity', {}).get('row_len'),
              'loss', None, 'parity_identical', e.get('parity', {}).get('identical'),
              'div', e.get('parity', {}).get('first_divergence'),
              'teacher', e.get('teacher', {}))
    print('ROWS', len(rows_gkd), len(rows_grpo))


if __name__ == '__main__':
    main()
