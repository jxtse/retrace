
import json,re,sys
from pathlib import Path
def short(s,n):
    s=s if isinstance(s,str) else json.dumps(s,ensure_ascii=False)
    return s if len(s)<=n else s[:n]+f'…[+{len(s)-n} chars]'
def tool_view(name,content):
    try:o=json.loads(content)
    except Exception: return short(content,600)
    if not isinstance(o,dict): return short(content,600)
    if o.get('error'): return 'ERROR: '+short(str(o['error']),400)
    if name=='score_route':
        keep={k:o.get(k) for k in ['total_score','route_complete','buyable_leaf_fraction','non_buyable_leaves','feasibility_score','cost_score','greenness_score','bottleneck','warnings','route_fp','_result_id']}
        return short(keep,2500)
    if name=='save_best_route':
        sv=o.get('saved') or {}
        return short({'ok':o.get('ok'),'status':o.get('status'),'error':o.get('error'),'message':o.get('message'),'saved_label':sv.get('label'),'saved_score':sv.get('total_score'),'saved_sub_scores':sv.get('sub_scores')},700)
    if name=='list_best_routes': return short([{k:r.get(k) for k in ('label','total_score','num_steps','sub_scores')} for r in o.get('routes',[])],1500)
    if name=='check_buyability': return short([(r.get('input_smiles'),r.get('buyable'),r.get('found_in_stocks')) for r in o.get('results',[])],1500)
    if name=='validate_smiles': return short({k:o.get(k) for k in ('valid','canonical','formula','error')},300)
    if name=='neural_retro':
        return short({'target':o.get('target'),'per_engine_counts':o.get('per_engine_counts'),'predictions':[(p.get('reactants'),round(p.get('score',0),2),p.get('support')) for p in (o.get('predictions') or [])[:10]]},2200)
    if name=='suggest_templates':
        return short({'product':o.get('product'),'suggestions':[(s.get('template_source'),s.get('template_kind'),s.get('reactants')) for s in (o.get('suggestions') or [])[:10]]},2200)
    if name=='explore_branches':
        return short({'ranked':[{k:b.get(k) for k in ('name','reactants','score','feasibility','sub_scores','support','warnings')} for b in o.get('ranked',[])]},2500)
    return short(o,1200)
def compact(session_dir):
    msgs=json.loads((Path(session_dir)/'conversation/latest_messages.json').read_text())['messages']
    names={};a=0;lines=[]
    for k,m in enumerate(msgs):
        if m['role']=='assistant':
            a+=1;lines.append(f"\n### A{a}")
            if (m.get('content') or '').strip(): lines.append('TEXT: '+m['content'].strip())
            for tc in m.get('tool_calls') or []:
                names[tc['id']]=tc['function']['name']
                lim=6000 if tc['function']['name'] in ('save_best_route','score_route','explore_branches') else 1500
                lines.append(f"CALL {tc['function']['name']}: {short(tc['function']['arguments'],lim)}")
        elif m['role']=='tool':
            n=names.get(m['tool_call_id'],'?')
            lines.append(f"  RESULT {n}: {tool_view(n,m['content'])}")
        elif m['role']=='user' and k>1:
            c=m['content']
            if c.startswith('[harness prefetch] '):
                for part in re.split(r'\n?\[harness prefetch\] ',c)[1:]:
                    mm=re.match(r'(\w+): (.*)$',part.strip(),re.S)
                    lines.append(f"\n[HARNESS PREFETCH, visible to agent] {mm.group(1)}: {tool_view(mm.group(1),mm.group(2))}")
            else: lines.append(f"\n[HARNESS USER MESSAGE] {short(c,800)}")
    return '\n'.join(lines),a
