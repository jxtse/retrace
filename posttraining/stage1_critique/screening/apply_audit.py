"""Apply LLM leak-span audits mechanically and re-gate (runs on the GPU host in retrace-continual venv, CPU only).
For each audited row:
  kept text = concatenation of original segments [keep_from..end] minus delete_segments (verbatim slices),
  then: regex LEAK must be absent, original gen.gates() must pass (complete, parsable final, decisions == reference,
  RDKit SMILES), and a dangling-reference regex check on the first kept segments.
Writes applied.jsonl + prints a summary by pilot kind. Does NOT touch any training data.
Usage: python apply_audit.py pilot_inputs.jsonl pilot_audit.jsonl applied.jsonl"""
import json, re, sys
from pathlib import Path


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
PB = Path(RETRACE_SHARE + '/stage1-selfdistill/projection_batch_v1')
sys.path.insert(0, str(PB.parent)); sys.path.insert(0, str(PB))
import gen
from gen import LEAK

DANGLE = re.compile(r"\b(?:as (?:computed|noted|shown|discussed|established|determined|mentioned|found) (?:above|earlier|before)|"
                    r"(?:above|earlier|previous(?:ly)?) (?:analysis|computation|count|parse|parsing)|as before|again,|"
                    r"let me re-?(?:examine|check|read|consider|verify)|re-?examin|reconsider|second look|on second thought)\b", re.I)


def main(inp, aud, outp):
    tok = gen.load_tokenizer(PB.parent)
    src = {json.loads(l)['row_id']: json.loads(l) for l in open(PB.parent / 'inputs/reference_generation_candidates.jsonl')}
    X = {json.loads(l)['row_id']: json.loads(l) for l in open(inp)}
    out, summ = [], {}
    for l in open(aud):
        a = json.loads(l)
        x = X[a['row_id']]
        rec = {'row_id': a['row_id'], 'kind': a['kind']}
        if 'audit' not in a:
            rec['status'] = 'api_error'
        else:
            au = a['audit']; v = au.get('verdict')
            rec['verdict'] = v
            segs = x['segments']
            if v == 'unsalvageable':
                rec['status'] = 'drop_unsalvageable'
            else:
                k = int(au.get('keep_from') or 0)
                dele = set(int(i) for i in au.get('delete_segments') or [])
                if v == 'confirm_only':
                    dele |= set(int(i) for i in au.get('leak_segments') or [])
                keep_idx = [i for i in range(k, len(segs)) if i not in dele]
                text = ''.join(x['reasoning'][segs[i][0]:segs[i][1]] for i in keep_idx)
                text = re.sub(r'\n{3,}', '\n\n', text).strip()
                rec.update(keep_from=k, n_segments=len(segs), n_deleted=len(segs) - len(keep_idx),
                           chars_before=len(x['reasoning']), chars_after=len(text))
                head = text[:600]
                rec['dangle_regex'] = bool(DANGLE.search(head))
                rec['llm_dangle'] = bool(au.get('dangling_reference_after_cut'))
                rec['llm_supports_all'] = bool(au.get('remaining_supports_all_decisions'))
                if not text:
                    rec['status'] = 'drop_empty'
                elif LEAK.search(text) or LEAK.search(x['final']):
                    rec['status'] = 'drop_leak_remains'
                    rec['leak_hit'] = (LEAK.search(text) or LEAK.search(x['final'])).group(0)
                else:
                    ids = tok.encode(text + '\n', add_special_tokens=False) + [gen.CLOSE] + \
                        tok.encode('\n\n' + x['final'].strip(), add_special_tokens=False) + [gen.EOS]
                    reasons, flags, extra = gen.gates(tok, src[a['row_id']], ids)
                    rec['gate_reasons'] = reasons
                    if reasons:
                        rec['status'] = 'drop_gates'
                    elif rec['llm_dangle'] or not rec['llm_supports_all']:
                        rec['status'] = 'drop_llm_flag'
                    else:
                        rec['status'] = 'ok_dangle_regex' if rec['dangle_regex'] else 'ok'
                    rec['new_reasoning'] = text
        out.append(rec)
        summ.setdefault(a['kind'], {}).setdefault(rec.get('verdict'), {}).setdefault(rec['status'], 0)
        summ[a['kind']][rec.get('verdict')][rec['status']] += 1
    with open(outp, 'w') as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
