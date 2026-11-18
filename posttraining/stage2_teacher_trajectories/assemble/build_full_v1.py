#!/usr/bin/env python3
"""Stage-2 full-trajectory render (arm A1 base layer). CPU only, no model calls.
Renders each Astra trajectory exactly as the student deployment would see it
(Qwen3.8 template, reasoning_effort=low, enable_thinking=True, empty historical
think), and marks loss spans = visible content + tool calls + <|im_end|> of every
assistant turn. Think header, system/user/tool tokens never carry loss.
Segment selection (A2) later only turns some of these spans off."""
import json,hashlib,sys,os
from pathlib import Path
from multiprocessing import Pool
from transformers import AutoTokenizer

# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
R=Path(RETRACE_SHARE + '/stage2-agentic');O=R/'stage2_data/full_v1'
MODEL=RETRACE_SHARE + '/models/Qwen/Qwen3.8-27B'
KW=dict(enable_thinking=True,reasoning_effort='low')  # matches vLLM build_chat_params for harness requests
HDR='<|im_start|>assistant\n<think>\n\n</think>\n\n'
tok=None
def init():
    global tok; tok=AutoTokenizer.from_pretrained(MODEL)
def vllm_like(msgs):
    out=[]
    for m in msgs:
        m=json.loads(json.dumps(m))
        if m['role']=='assistant':
            if 'tool_calls' in m and not m['tool_calls']: m.pop('tool_calls')
            for tc in m.get('tool_calls') or []:
                a=tc['function'].get('arguments'); tc['function']['arguments']=json.loads(a) if a else {}
        out.append(m)
    return out
def render(msgs,tools): return tok.apply_chat_template(msgs,tools=tools,tokenize=False,**KW)
def one(x):
    sd=Path(x['session_dir'])
    reqs=(sd/'conversation/llm_requests.jsonl').read_text().splitlines()
    last=json.loads(reqs[-1]); tools=last['tools']
    msgs=json.loads((sd/'conversation/latest_messages.json').read_text())['messages']
    assert msgs[:len(last['messages'])]==last['messages']
    vm=vllm_like(msgs); full=render(vm,tools)
    spans=[];prev_len=None
    for k,m in enumerate(vm):
        if m['role']!='assistant':continue
        pre=render(vm[:k],tools); post=render(vm[:k+1],tools)
        assert full.startswith(pre) and full.startswith(post),('prefix',k)
        seg=post[len(pre):]
        assert seg.startswith(HDR),('hdr',k,seg[:80])
        assert seg.endswith('<|im_end|>\n'),('end',k)
        # deployment prompt for this turn ends with '<|im_start|>assistant\n<think>\n'
        gp=tok.apply_chat_template(vm[:k],tools=tools,tokenize=False,add_generation_prompt=True,**KW)
        assert gp==pre+'<|im_start|>assistant\n<think>\n',('genprompt',k)
        spans.append(dict(msg_index=k,loss_start=len(pre)+len(HDR),loss_end=len(post)-1,
             tools=[t['function']['name'] for t in m.get('tool_calls') or []],has_content=bool((m.get('content') or '').strip())))
    enc=tok(full,return_offsets_mapping=True,add_special_tokens=False)
    ids=enc['input_ids'];off=enc['offset_mapping']
    mask=[0]*len(ids)
    for s in spans:
        a,b=s['loss_start'],s['loss_end'];n=0
        for i,(c0,c1) in enumerate(off):
            if c0>=a and c1<=b: mask[i]=1;n+=1
            elif c0<b and c1>a: raise AssertionError(('straddle',s['msg_index'],c0,c1,a,b))
        s['n_loss_tokens']=n
        assert ids[[i for i,(c0,c1) in enumerate(off) if c0>=a and c1<=b][-1]]==tok.convert_tokens_to_ids('<|im_end|>')
    uid=f"{x['rep']}_{x['idx']:03d}"
    rec=dict(uid=uid,rep=x['rep'],idx=x['idx'],session_dir=str(sd),record_sha256=x['record_sha256'],
        text_sha256=hashlib.sha256(full.encode()).hexdigest(),n_tokens=len(ids),n_loss_tokens=sum(mask),n_assistant_turns=len(spans),spans=spans)
    (O/'tok').mkdir(parents=True,exist_ok=True)
    (O/'tok'/f'{uid}.json').write_text(json.dumps({'input_ids':ids,'loss_mask':mask}))
    (O/'msgs').mkdir(parents=True,exist_ok=True)
    (O/'msgs'/f'{uid}.json').write_text(json.dumps({'messages':msgs,'tools':tools,'chat_template_kwargs':KW},ensure_ascii=False))
    return rec
if __name__=='__main__':
    L=[json.loads(l) for l in (R/'stage2_data/COLLECTION_LEDGER_v1.jsonl').read_text().splitlines()]
    if len(sys.argv)>1: L=L[:int(sys.argv[1])]
    O.mkdir(parents=True,exist_ok=True)
    with Pool(48,initializer=init) as p: recs=p.map(one,L,chunksize=1)
    (O/'INDEX.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in recs))
    import statistics as st
    nt=sorted(r['n_tokens'] for r in recs);nl=sorted(r['n_loss_tokens'] for r in recs)
    q=lambda v:[v[0],v[len(v)//2],v[int(.9*(len(v)-1))],v[-1]]
    print(json.dumps({'n':len(recs),'tokens_min_med_p90_max':q(nt),'loss_tokens_min_med_p90_max':q(nl),'total_tokens':sum(nt),'total_loss_tokens':sum(nl),'over_131072':sum(t>131072 for t in nt),'kwargs':KW}))
