"""DeepSeek chemistry / consistency audit of S1 training rows (independent of the earlier sub-agent review:
the prior verdict is NOT shown to the model). Model: deepseek-v4-flash (key from env MINTCN_DSV4_FLASH_API_KEY).
Usage: python chem_audit.py inputs.jsonl out.jsonl [concurrency] [model]"""
import asyncio, json, subprocess, sys, time, re
import httpx

BASE = os.environ["MINTCN_BASE_URL"]

SYSTEM = """You are a strict organic-chemistry reviewer auditing training data for a student model that screens single-step retrosynthesis precursor proposals. Each item has: the QUESTION (product SMILES + numbered candidate precursor sets), the REFERENCE decisions (trusted labels), RDKIT FACTS computed for every SMILES (molecular formula, heavy atoms, ring sizes, aromatic ring count; these are ground truth), and the model's THINKING and FINAL answer.

The final keep/reject decisions already match the reference; do NOT re-decide them. Judge only the quality of the reasoning that would be taught:
1. structural errors: wrong ring size, wrong substituent position (ortho/meta/para, ring locants), wrong chain length or atom count, wrong functional group, misidentified heterocycle, wrong stereo claim, wrong bond formed/broken. Use the RDKit facts to check every counting or formula claim.
2. decision support: for each candidate, is the final decision justified by the thinking's OWN chemistry? Flag a candidate if the thinking concluded the opposite and then flipped without a chemical reason, or never analysed it, or justified it with a wrong structural claim.
3. final-answer reasons: does any candidate reason line in FINAL contain a factual chemistry error?
4. leak: any mention of a hidden/reference/provided answer or teacher.

Severity:
- "critical": an error that a reader would rely on for a decision (wrong structure/position/count that the decision rests on), an unsupported decision, an error in a FINAL reason line, or any leak.
- "minor": an incidental slip that is explicitly corrected later in the thinking, or a wrong statement that no decision depends on.
Verdict: "fail" if any critical issue; "minor" if only minor issues; "pass" otherwise. Be concrete: quote the wrong claim and give the correct fact.

Return ONLY JSON:
{"verdict": "pass|minor|fail",
 "issues": [{"severity": "critical|minor", "type": "structure|count|position|functional_group|stereo|mechanism|unsupported_decision|final_reason_error|leak|other", "candidate": <int or null>, "claim": "<quoted, <=25 words>", "correct": "<<=25 words>"}],
 "unsupported_candidates": [<ints>],
 "summary": "<=40 words"}"""


def user_msg(x):
    return (f"QUESTION:\n{x['question']}\n\nREFERENCE:\n{x['reference']}\n\nRDKIT FACTS:\n{json.dumps(x['rdkit'], ensure_ascii=False)}\n\n"
            f"THINKING:\n{x['reasoning']}\n\nFINAL:\n{x['final']}")


def get_key():
    key = os.environ.get('MINTCN_DSV4_FLASH_API_KEY')
    if not key:
        raise SystemExit('set MINTCN_DSV4_FLASH_API_KEY')
    return key


async def one(c, key, x, sem, model):
    async with sem:
        body = {'model': model, 'temperature': 0, 'max_tokens': 24000,
                'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user_msg(x)}],
                'response_format': {'type': 'json_object'}}
        err = 'retries exhausted'
        for attempt in range(5):
            t = time.time()
            try:
                r = await c.post(BASE + '/chat/completions', headers={'Authorization': f'Bearer {key}'}, json=body)
                if r.status_code in (429, 500, 502, 503, 504):
                    err = f'http {r.status_code}'
                    await asyncio.sleep(2 ** attempt * 4); continue
                d = r.json()
                ch = d['choices'][0]
                txt = ch['message'].get('content') or ''
                mm = re.search(r'\{.*\}', txt, re.S)
                if mm is None:
                    err = f"no_json finish={ch.get('finish_reason')}"
                    body['max_tokens'] = min(body['max_tokens'] * 2, 64000); continue
                return {'row_id': x['row_id'], 'audit': json.loads(mm.group(0)), 'latency': round(time.time() - t, 1),
                        'usage': d.get('usage'), 'model': model}
            except Exception as e:
                err = f'{type(e).__name__}: {str(e)[:200]}'
                await asyncio.sleep(2 ** attempt * 4)
        return {'row_id': x['row_id'], 'error': err, 'model': model}


async def main(inp, outp, conc, model):
    key = get_key()
    xs = [json.loads(l) for l in open(inp)]
    done = set()
    try:
        done = {json.loads(l)['row_id'] for l in open(outp) if 'audit' in json.loads(l)}
    except FileNotFoundError:
        pass
    xs = [x for x in xs if x['row_id'] not in done]
    sem = asyncio.Semaphore(conc)
    t0 = time.time(); n = 0
    async with httpx.AsyncClient(trust_env=False, timeout=900) as c:
        with open(outp, 'a') as fh:
            for fut in asyncio.as_completed([one(c, key, x, sem, model) for x in xs]):
                r = await fut
                fh.write(json.dumps(r, ensure_ascii=False) + '\n'); fh.flush(); n += 1
    print('done', n, 'skipped(already)', len(done), 'wall_s', round(time.time() - t0))


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 32,
                     sys.argv[4] if len(sys.argv) > 4 else 'deepseek-v4-flash'))
