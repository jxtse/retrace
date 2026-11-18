"""Deterministic per-turn veto flags for stage-2 A2 (no model). Output: mech_flags_v1.json {uid: [{turn,kind,detail}]}.
Kinds:
  save_omits_nonbuyable_leaf: save_best_route whose linked score_route reported non_buyable_leaves not listed in starting_materials (misreports the route's inputs).
  exact_duplicate_call: identical (tool,arguments) already issued earlier in the trajectory (list_best_routes excluded: state changes).
  bad_tool_args: tool-call arguments not valid JSON.
"""
import json
from pathlib import Path
D=Path(__file__).resolve().parent
L=[json.loads(l) for l in (D/'COLLECTION_LEDGER_v1.jsonl').read_text().splitlines()]
out={};C={}
for x in L:
    uid=f"{x['rep']}:{x['idx']}"
    m=json.loads((Path(x['session_dir'])/'conversation/latest_messages.json').read_text())['messages']
    a=0;turn_of={};calls={};seen={};score={};fl=[]
    for msg in m:
        if msg['role']=='assistant':
            a+=1
            for tc in msg.get('tool_calls') or []:
                n,args=tc['function']['name'],tc['function']['arguments'] or '{}'
                turn_of[tc['id']]=a;calls[tc['id']]=(n,args)
                try: json.loads(args)
                except Exception: fl.append(dict(turn=a,kind='bad_tool_args',detail=n))
                if n!='list_best_routes' and (n,args) in seen: fl.append(dict(turn=a,kind='exact_duplicate_call',detail=f'{n} first at A{seen[(n,args)]}'))
                seen.setdefault((n,args),a)
        elif msg['role']=='tool':
            n,args=calls.get(msg['tool_call_id'],(None,None));t=turn_of.get(msg['tool_call_id'])
            try:o=json.loads(msg['content'])
            except Exception:o={}
            if not isinstance(o,dict): continue
            if n=='score_route' and o.get('_result_id'): score[o['_result_id']]=o
            if n=='save_best_route':
                try: arg=json.loads(args)
                except Exception: continue
                sc=score.get(arg.get('score_route_result_id')) or {}
                nb=sc.get('non_buyable_leaves') or []
                sm=set(arg.get('starting_materials') or [])
                miss=[s for s in nb if s not in sm]
                if miss: fl.append(dict(turn=t,kind='save_omits_nonbuyable_leaf',detail=f"{len(miss)} non-buyable leaf not in starting_materials: {miss[0][:80]}"))
    out[uid]=fl
    for f in fl: C[f['kind']]=C.get(f['kind'],0)+1
(D/'mech_flags_v1.json').write_text(json.dumps(out,indent=1))
print(C,'trajs with any',sum(bool(v) for v in out.values()))
