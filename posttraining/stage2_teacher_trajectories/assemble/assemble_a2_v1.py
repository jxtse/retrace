"""Assemble stage2 A2 dataset: VALID-only trajectories, full history input, loss only on selector-kept assistant turns.
Inputs: full_v1/{INDEX.jsonl,tok/}, selector/final_v4_s{1,2}/, route_audit_v1/{verdicts.jsonl,route_members.json}, FROZEN_v4.json."""
import json,hashlib,collections,sys
from pathlib import Path
D=Path(__file__).resolve().parent;SEL=D/'selector';A=D/'route_audit_v1';import os
OUT=D/os.environ.get('A2_OUT','a2_v1')
MAXTOK=131072
VALUE_TH=float(os.environ.get('A2_VALUE_TH','7'))  # SWE-Prime delta=7 on mean of 2 samples; 0 = risk-only (v4)
fz=json.loads((SEL/'FROZEN_v4.json').read_text())
MF=json.loads((D/'mech_flags_v1.json').read_text())
assert hashlib.sha256((SEL/fz['prompt']).read_bytes()).hexdigest()==fz['prompt_sha256'] and hashlib.sha256((D/'mech_flags_v1.py').read_bytes()).hexdigest()==fz['mech_flags_sha256']
mem=json.loads((A/'route_members.json').read_text());verd={}
for l in (A/'verdicts.jsonl').read_text().splitlines():
    v=json.loads(l)
    for u in mem[v['uid']]: verd[u]=v['route_verdict']
IDX={x['uid']:x for x in map(json.loads,(D/'full_v1/INDEX.jsonl').read_text().splitlines())}
def per_turn(o):
    m={}
    for s in o['segments']:
        rt={t['turn']:t for t in s['risky_turns']}
        for a in range(s['start'],s['end']+1): m[a]=(s['overall'],rt[a]['risk'] if a in rt else 0,s['intent'],rt[a]['reason'] if a in rt else '')
    return m
OUT.mkdir(exist_ok=True);(OUT/'tok').mkdir(exist_ok=True)
rows=[];C=collections.Counter()
for u,v in sorted(verd.items()):
    C['verdict',v]+=1
    if v!='VALID': continue
    fu=u.replace(':','_');rep,i=u.split(':');key=f"{rep}_{int(i):03d}";x=IDX[key]
    if x['n_tokens']>MAXTOK: C['drop_overlong']+=1;continue
    ss=[SEL/f'final_v4_s{k}'/f'{fu}.json' for k in (1,2)]
    if not all(p.exists() for p in ss): C['pending_selection']+=1;continue
    ms=[per_turn(json.loads(p.read_text())) for p in ss]
    spans=x['spans'];assert len(spans)==39==len(ms[0])==len(ms[1])
    tok=json.loads((D/'full_v1/tok'/f'{key}.json').read_text());ids=tok['input_ids'];base=tok['loss_mask']
    assert len(ids)==x['n_tokens']
    runs=[];j=0
    while j<len(base):
        if base[j]:
            s0=j
            while j<len(base) and base[j]: j+=1
            runs.append((s0,j))
        else: j+=1
    assert len(runs)==len(spans) and all(e-s==sp['n_loss_tokens'] for (s,e),sp in zip(runs,spans)),('runs',key,len(runs))
    mask=[0]*len(ids);kept=[]
    for k,(sp,(ts,te)) in enumerate(zip(spans,runs),1):
        ov=sum(m[k][0] for m in ms)/2;rk=max(m[k][1] for m in ms)
        mf=[f for f in MF[u] if f['turn']==k]
        keep=rk<=2 and not mf and ov>=VALUE_TH
        if keep:
            for j in range(ts,te): mask[j]=1
        kept.append(dict(turn=k,tok_start=ts,tok_end=te,keep=keep,overall_mean=ov,risk_max=rk,tools=sp['tools'],n_loss_tokens=sp['n_loss_tokens'],intents=[m[k][2] for m in ms],risk_reasons=[m[k][3] for m in ms if m[k][3]],mech_flags=[f['kind']+': '+f['detail'] for f in mf]))
    assert all(base[j]>=mask[j] for j in range(len(ids)))
    nk=sum(t['keep'] for t in kept);C['turns_kept']+=nk;C['turns_total']+=39
    for t in kept:
        for tn in set(t['tools']): C['tool_total',tn]+=1;C['tool_kept',tn]+=t['keep']
    (OUT/'tok'/f'{key}.json').write_text(json.dumps({'input_ids':ids,'loss_mask':mask}))
    rows.append(dict(uid=u,key=key,idx=int(i),rep=rep,record_sha256=x['record_sha256'],text_sha256=x['text_sha256'],n_tokens=len(ids),n_loss_tokens=sum(mask),n_loss_tokens_full=x['n_loss_tokens'],turns_kept=nk,turns=kept))
(OUT/'INDEX.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
summ=dict(value_threshold=VALUE_TH,n_traj=len(rows),n_targets=len({r['idx'] for r in rows}),tokens=sum(r['n_tokens'] for r in rows),loss_tokens=sum(r['n_loss_tokens'] for r in rows),loss_tokens_full=sum(r['n_loss_tokens_full'] for r in rows),
  counts={' '.join(map(str,k)) if isinstance(k,tuple) else k:v for k,v in C.items()},frozen_selector=fz,
  save_kept=f"{C['tool_kept','save_best_route']}/{C['tool_total','save_best_route']}",score_kept=f"{C['tool_kept','score_route']}/{C['tool_total','score_route']}")
(OUT/'SUMMARY.json').write_text(json.dumps(summ,indent=1,ensure_ascii=False));print(json.dumps({k:v for k,v in summ.items() if k!='frozen_selector'},indent=1))
