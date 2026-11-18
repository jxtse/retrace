"""S1 v2 rejection sampling: base Qwen3.8-27B answers the STUDENT prompt only (no hidden
reference anywhere in the prompt), reasoning_effort=medium (same chat-template KWARGS as S1 data/training),
same sampling as S1 generation (T=0.7, top_p=0.8, top_k=20). K samples per question; every sample goes through the
unchanged S1 automatic gates (gen.gates: complete thinking/EOS boundaries, no leak words, final parses, keep/reject
== reference, RDKit SMILES, native-template retokenization). Nothing is selected here; all samples are recorded.
Usage: python rs_gen.py <service_url> <shard_index> <n_shards> [K] [limit]
Output: rows/<row_id>.json (resumable; a row is skipped if its file exists)."""
import asyncio, json, os, random, re, sys, time
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
PB = Path(RETRACE_SHARE + '/stage1-selfdistill/projection_batch_v1')
sys.path.insert(0, str(PB.parent)); sys.path.insert(0, str(PB))
import aiohttp
import gen
from thinking_v2.native import render_prompt
from thinking_v2.common import KWARGS

OUT = Path(__file__).resolve().parent
MAX_TOKENS = 40960
# Sampling-only system message = the task/format part of S1's TEACHER_INSTRUCTION_V4, with every hidden-answer /
# teacher/student sentence removed. The student prompt alone gives no output format (pilot v0: 40/40 markdown answers,
# final_parse). The training row is still the student prompt only, exactly like S1; instruction-quote filters below.
SYS_FORMAT = ("The task concerns direct single-step precursor proposals. Reagents, catalysts and solvents may be omitted; "
              "standard ancillary reagents may be implied. Do not introduce missing atom-contributing building blocks or "
              "arbitrary multistep sequences. Avoid unrelated structural elaboration.\n"
              "In the final answer return one index|keep/reject|reason line per candidate, in original index order, with exact "
              "lowercase keep or reject, followed by keep: and comma-separated kept indices (or keep: none). No preamble or "
              "code fence. Give concise candidate-specific explanations that can be read with the original question alone.")
import finalize_rows
INSTR_REF = re.compile(r"\b(?:system (?:prompt|message)|the instructions?|instructed|as (?:requested|required|specified)|"
                       r"the (?:required|requested|specified) format|format (?:requires|says|asks))\b", re.I)
assert KWARGS['reasoning_effort'] == 'medium' and KWARGS['enable_thinking'] is True


async def do_row(row, svc, tok, K, sem):
    f = OUT / 'rows' / (row['row_id'].replace(':', '_') + '.json')
    if f.exists():
        return 'skip'
    s_ids = render_prompt(tok, [{'role': 'system', 'content': SYS_FORMAT}] + row['student_messages'])['token_ids']
    seeds = [int.from_bytes(row['row_id'].encode()[-6:], 'big') % (2 ** 31) + k for k in range(K)]

    async def one(seed):
        async with sem:
            try:
                p = await svc.propose(s_ids, [], MAX_TOKENS, seed)
            except Exception as e:
                return {'seed': seed, 'infra_error': str(e)[:300]}
            ids = p['ids']
            reasons, flags, extra = gen.gates(tok, row, ids)
            if not reasons:
                q = row['student_messages'][0]['content']
                rr = {'reasoning': extra['reasoning'], 'content': extra['content'], 'train_tokens': extra['train_tokens']}
                fin = finalize_rows.MAX_TOKENS
                finalize_rows.MAX_TOKENS = 10 ** 9   # no length cap for v2; length is reported, not filtered
                reasons = finalize_rows.extra_reasons(rr, q, None)
                finalize_rows.MAX_TOKENS = fin
                if INSTR_REF.search(extra['reasoning'] + '\n' + extra['content']):
                    reasons.append('instruction_reference')
            return {'seed': seed, 'finish': p['finish'], 'n_tokens': len(ids), 'latency': round(p['latency'], 1),
                    'reasons': reasons, 'flags': flags, 'accepted': not reasons,
                    'reasoning': extra.get('reasoning'), 'content': extra.get('content'),
                    'train_tokens': extra.get('train_tokens'), 'ids': ids if not reasons else None}
    samples = await asyncio.gather(*[one(s) for s in seeds])
    if any('infra_error' in s for s in samples):
        return 'infra'
    rec = {'row_id': row['row_id'], 'source_key': row['source_key'], 'composition': row['composition'],
           'sampling_prompt_tokens': len(s_ids), 'sampling_system': SYS_FORMAT, 'kwargs': KWARGS, 'sampling': gen.SAMPLING, 'max_tokens': MAX_TOKENS,
           'samples': samples, 'n_accepted': sum(s['accepted'] for s in samples)}
    tmp = f.with_suffix('.tmp')
    tmp.write_text(json.dumps(rec, ensure_ascii=False)); tmp.rename(f)
    return 'ok'


async def main(url, shard, nshard, K, limit):
    (OUT / 'rows').mkdir(exist_ok=True)
    tok = gen.load_tokenizer(PB.parent)
    rows = [json.loads(l) for l in open(PB.parent / 'inputs/reference_generation_candidates.jsonl')]
    assert all(r['split'] == 'train' for r in rows)
    rows.sort(key=lambda r: r['row_id'])
    random.Random(20261105).shuffle(rows)
    if limit:
        rows = rows[:limit]
    rows = rows[shard::nshard]
    sem = asyncio.Semaphore(int(os.environ.get('RS_CONC', '32')))
    t0 = time.time(); c = {'ok': 0, 'skip': 0, 'infra': 0}
    async with aiohttp.ClientSession() as session:
        svc = gen.Service(url, session)
        tasks = [asyncio.create_task(do_row(r, svc, tok, K, sem)) for r in rows]
        for i, t in enumerate(asyncio.as_completed(tasks), 1):
            c[await t] += 1
            if i % 20 == 0 or i == len(tasks):
                print(json.dumps({'t': round(time.time() - t0), 'done': i, 'of': len(tasks), **c}), flush=True)


if __name__ == '__main__':
    a = sys.argv
    asyncio.run(main(a[1], int(a[2]), int(a[3]), int(a[4]) if len(a) > 4 else 4, int(a[5]) if len(a) > 5 else 0))
