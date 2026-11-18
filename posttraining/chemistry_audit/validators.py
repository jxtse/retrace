"""Verbatim function extraction; see SOURCE_PINS.json. No source main is run."""
import audit_driver as frozen
LABELS = {"VALID", "SUSPECT", "FAKE"}
CANONICAL_MODEL = "dev-anthropic-claude-opus-5"


def require(ok, message):
    if not ok: raise ValueError(message)

def validate_deep(a, indices):
    require(isinstance(a, dict), 'deepread object missing')
    steps = a.get('steps')
    require(isinstance(steps, list) and len(steps)==len(indices), 'step coverage mismatch')
    require(all(isinstance(s, dict) for s in steps), 'invalid step type')
    require(all(type(s.get('step_index')) is int for s in steps), 'step index type')
    require(sorted(s['step_index'] for s in steps)==sorted(indices), 'step ids mismatch')
    for s in steps:
        require(s.get('verdict') in LABELS, 'step verdict invalid')
        require(s.get('defect_code') in frozen.DEFECTS, 'step defect invalid')
        require(isinstance(s.get('reason'),str) and bool(s['reason'].strip()), 'step reason missing')
    nf = sum(s['verdict']=='FAKE' for s in steps)
    ns = sum(s['verdict']=='SUSPECT' for s in steps)
    label = 'FAKE' if nf else ('SUSPECT' if ns else 'VALID')
    require(a.get('route_verdict')==label, 'route/step aggregation mismatch')
    require(type(a.get('n_fake_steps')) is int and a['n_fake_steps']==nf, 'fake count mismatch')
    require(type(a.get('n_suspect_steps')) is int and a['n_suspect_steps']==ns, 'suspect count mismatch')
    require(a.get('worst_defect_code') in frozen.DEFECTS, 'worst defect invalid')
    require(isinstance(a.get('summary'),str) and bool(a['summary'].strip()), 'summary missing')

def validate_adv(a):
    require(isinstance(a,dict), 'adversary object missing')
    require(type(a.get('fake_upheld')) is bool, 'adversary flag is not boolean')
    require(a.get('corrected_verdict') in LABELS, 'corrected verdict invalid')
    require(a['fake_upheld']==(a['corrected_verdict']=='FAKE'), 'adversary verdict inconsistent')
    require(isinstance(a.get('reason'),str) and bool(a['reason'].strip()), 'adversary reason missing')

def response_text(body):
    require(body.get('model')==CANONICAL_MODEL, 'response model mismatch')
    choices=body.get('choices') or []
    require(len(choices)==1, 'response choices missing/ambiguous')
    require(choices[0].get('finish_reason')=='stop', 'response not naturally complete')
    text=choices[0].get('message',{}).get('content')
    require(isinstance(text,str) and bool(text.strip()), 'empty response')
    return text
