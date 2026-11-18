"""Hash-bound ms-swift 4.6 cached-dataset/custom-template interface, offline only.
Engineering controls only. Never a scientific selector or a training launcher.
"""
import os

# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import copy, hashlib, inspect, json, socket
from pathlib import Path
from types import SimpleNamespace
O=Path(__file__).resolve().parent
V1=O.parent/'v1'
MODES=('eligible_control','pending_zero','alternating_runs_control')
INDEX={r['uid']:r for r in map(json.loads,(V1/'control/INDEX.jsonl').read_text().splitlines())}
def require(ok,why):
    if not ok: raise ValueError(why)
def reference(uid):
    require(type(uid) is str and uid in INDEX,'unknown uid')
    p=V1/'control/tok'/f'{uid}.json'
    raw=p.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==INDEX[uid]['tokens_sha256'],'frozen token hash changed')
    return json.loads(raw)
def expected_mask(ref,mode):
    require(mode in MODES,'unknown control mode; scientific release is forbidden')
    if mode=='pending_zero': return ref['selected_loss_mask']
    if mode=='eligible_control': return ref['loss_mask']
    # Deliberately non-scientific partial-mask fixture: retain alternating contiguous
    # eligible token runs. This is NOT a turn-selection or route-validity decision.
    run=-1; prev=0; out=[]
    for x in ref['loss_mask']:
        if x and not prev: run+=1
        out.append(int(bool(x) and run%2==0)); prev=x
    return out
def record(index,mode):
    ref=reference(index['uid']); mask=expected_mask(ref,mode); ids=ref['input_ids']
    return dict(uid=index['uid'],control_mode=mode,input_ids=ids,
                labels=[i if m else -100 for i,m in zip(ids,mask)],lengths=[len(ids)])
from swift.template import TEMPLATE_MAPPING, register_template
Base=TEMPLATE_MAPPING['qwen3_8'].template_cls
class FrozenPretokenizedTemplate(Base):
    def _get_model(self):
        raise RuntimeError('offline adapter forbids even meta-model construction')
    def _get_get_rope_index(self):
        # Native Qwen collator ordinarily constructs a dummy model on meta just
        # to call this pure helper. Bind the installed helper to config only;
        # no nn.Module, model constructor, or weights are touched. Text only.
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
        cfg=json.loads((Path(self.model_info.model_dir)/'config.json').read_text())
        holder=SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(**cfg['vision_config'])))
        def text_only(*args,**kwargs):
            require(kwargs.get('image_grid_thw') is None and kwargs.get('video_grid_thw') is None,'multimodal input forbidden')
            kinds=kwargs.get('mm_token_type_ids')
            require(kinds is not None and not bool(kinds.any()),'non-text token types forbidden')
            return Qwen3_5Model.get_rope_index(holder,*args,**kwargs)
        return text_only
    def encode(self,inputs,return_template_inputs=False,return_length=False):
        require(isinstance(inputs,dict),'pretokenized dict required')
        require(not return_template_inputs,'template input reconstruction unsupported')
        require(set(inputs)=={'uid','control_mode','input_ids','labels','lengths'},'unexpected or missing cached columns')
        ref=reference(inputs['uid']); ids=inputs['input_ids']; labels=inputs['labels']
        require(type(ids) is list and all(type(x) is int for x in ids),'integer IDs required')
        require(type(labels) is list and all(type(x) is int for x in labels),'integer labels required')
        require(ids==ref['input_ids'],'exact full input IDs required; truncation forbidden')
        require(inputs['lengths']==[len(ids)] and len(ids)<=131072,'invalid full-context length')
        mask=expected_mask(ref,inputs['control_mode'])
        require(labels==[i if m else -100 for i,m in zip(ids,mask)],'exact selected labels required')
        require(all(s<=b for s,b in zip(mask,ref['loss_mask'])),'mask outside frozen eligibility')
        require(not mask[0],'causal first token must be ignored')
        # Hard reasoning exclusion independently checked over actual IDs, including
        # all whitespace between the thinking delimiters, not just delimiters.
        opening=self.tokenizer.convert_tokens_to_ids('<think>')
        closing=self.tokenizer.convert_tokens_to_ids('</think>')
        inside=False
        for i,x in enumerate(ids):
            if x==opening: inside=True
            if inside or x==closing: require(mask[i]==0,'reasoning loss prohibited')
            if x==closing: inside=False
        out={'input_ids':ids.copy(),'labels':labels.copy()}
        if return_length: out['lengths']=[len(ids)]
        return out
meta=copy.deepcopy(TEMPLATE_MAPPING['qwen3_8'])
meta.template_type='retrace_s3_frozen_pretokenized_v2'; meta.template_cls=FrozenPretokenizedTemplate
register_template(meta,exist_ok=True)
def make_template():
    from swift import get_processor,get_template
    processor=get_processor(RETRACE_SHARE + '/models/Qwen/Qwen3.8-27B')
    template=get_template(processor,template_type=meta.template_type,loss_scale='ignore_empty_think',
                          add_non_thinking_prefix=True,enable_thinking=True,template_backend='swift')
    template.set_mode('train')
    return template
def verify_parent():
    manifest=json.loads((V1/'ARTIFACT_HASHES.json').read_text())
    # Manifest structure is detected, never silently skip any entry.
    if isinstance(manifest,dict) and 'files' in manifest: manifest=manifest['files']
    require(isinstance(manifest,dict),'unknown v1 manifest format')
    for path,digest in manifest.items():
        require(type(digest) is str,'unknown v1 hash record')
        require(hashlib.sha256((V1/path).read_bytes()).hexdigest()==digest,'v1 payload changed: '+path)
    return len(manifest)
def run_acceptance():
    import swift
    from datasets import Dataset,Features,Sequence,Value
    from swift.pipelines.utils import get_cached_dataset
    from swift.pipelines.train.sft import SwiftSft
    from swift.dataset import LazyLLMDataset
    parent_count=verify_parent()
    template=make_template()
    # Fail closed on accidental outbound connections throughout dataset/read/collate.
    original_connect=socket.socket.connect; original_connect_ex=socket.socket.connect_ex
    def forbidden(*a,**kw): raise RuntimeError('offline acceptance forbids socket connections')
    socket.socket.connect=forbidden; socket.socket.connect_ex=forbidden
    cases=[]; counters={m:0 for m in MODES}; pair_checks=0
    features=Features({'uid':Value('string'),'control_mode':Value('string'),
                       'input_ids':Sequence(Value('int64')),'labels':Sequence(Value('int64')),
                       'lengths':Sequence(Value('int64'))})
    try:
        for mode in MODES:
            path=O/'cached_controls'/mode
            if not path.exists():
                ds=Dataset.from_generator(lambda: (record(r,mode) for r in INDEX.values()),features=features,
                                          cache_dir=str(O/'.datasets_cache'))
                ds.save_to_disk(str(path))
            args=SimpleNamespace(cached_dataset=[str(path)],cached_val_dataset=[],data_seed=42,
                                 max_length=131072,truncation_strategy='raise',dataset_shuffle=False,
                                 streaming=False,strict=True,packing=False)
            train,val=get_cached_dataset(args)
            require(len(train)==1 and not val,'unexpected native cached split')
            # Exercise actual SFT postprocessor without calling constructor/run or
            # model/tokenizer preparation. Native LazyLLMDataset invokes registered encode.
            native=SwiftSft._post_process_datasets(SimpleNamespace(args=args,template=template),[train[0],None])[0]
            require(isinstance(native,LazyLLMDataset),'native SFT reader not exercised')
            require(len(native)==len(INDEX),'native row count changed')
            previous=None
            for n,(uid,index) in enumerate(INDEX.items()):
                raw=train[0][n]; expected=record(index,mode)
                require(raw==expected,'native cache loader changed exact row')
                enc=native[n]; ids=enc['input_ids']; labels=enc['labels']; L=len(ids)
                require(ids==expected['input_ids'] and labels==expected['labels'],'native encode parity failed')
                batch=template.data_collator([enc]); bi=batch['input_ids'][0].tolist(); bl=batch['labels'][0].tolist()
                require(bi[:L]==ids and bl[:L]==labels,'native collator changed tokens/labels')
                require(all(x==-100 for x in bl[L:]),'padding label leak')
                require(template.model is None and template.dummy_model is None,'model unexpectedly constructed')
                require(batch['position_ids'].shape[0]==3,'native Qwen position rank changed')
                require(all(axis[0][:L]==list(range(L)) for axis in batch['position_ids'].tolist()),'native full-text position mismatch')
                count=sum(x!=-100 for x in bl[1:]); counters[mode]+=count
                require(count==sum(x!=-100 for x in labels),'causal shift changed mask')
                if mode=='pending_zero': require(count==0,'zero-mask suffix leak')
                if previous is not None:
                    pair=template.data_collator([previous,enc])
                    for j,e in enumerate([previous,enc]):
                        length=len(e['input_ids']); am=pair['attention_mask'][j].tolist()
                        active=[k for k,v in enumerate(am) if v]
                        pi=pair['input_ids'][j].tolist(); pl=pair['labels'][j].tolist()
                        require([pi[k] for k in active]==e['input_ids'],'padded batch token mismatch')
                        require([pl[k] for k in active]==e['labels'],'padded batch label mismatch')
                        require(all(pl[k]==-100 for k,v in enumerate(am) if not v),'batch padding leak')
                    pair_checks+=1
                previous=enc
                cases.append(dict(uid=uid,source_key=index['source_key'],mode=mode,tokens=L,loss_tokens=count,
                                  token_hash=index['tokens_sha256'],exact_full_tokens=True,exact_labels=True,
                                  reasoning_loss_zero=True,causal_shift_exact=True,collator_exact=True))
    finally:
        socket.socket.connect=original_connect; socket.socket.connect_ex=original_connect_ex
    require(verify_parent()==parent_count,'v1 manifest changed')
    (O/'NATIVE_READER_CASES.jsonl').write_text(''.join(json.dumps(c,sort_keys=True)+'\n' for c in cases))
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    sources={name:dict(path=inspect.getfile(obj),sha256=hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest())
             for name,obj in [('get_cached_dataset',get_cached_dataset),('sft_postprocessor',SwiftSft),('native_lazy_reader',LazyLLMDataset),('template',Base),('text_rope_helper',Qwen3_5Model.get_rope_index)]}
    report=dict(swift_version=swift.__version__,unique_real_rows=len(INDEX),cases=len(cases),
                full_tokens_once=sum(r['n_tokens'] for r in INDEX.values()),
                exact_full_tokens_and_masks=True,loss_tokens_by_control=counters,
                pending_selected_loss_tokens=counters['pending_zero'],paired_padding_batches=pair_checks,
                v1_payload_hashes_verified_before_and_after=parent_count,native_sources=sources,
                interface='native get_cached_dataset -> SFT._post_process_datasets -> LazyLLMDataset -> registered custom template -> native collator',
                template_type=meta.template_type,network_connections_forbidden=True,model_loaded=False,
                model_constructed=False,meta_model_constructed=False,native_text_rope_without_model=True,
                inference_calls=0,optimizer_steps=0,scientific_selector_run=False,training_ready=False,
                warning='All datasets are engineering controls, never admitted training data; no route VALID or selector receipts exist.')
    (O/'NATIVE_READER_REPORT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report
if __name__=='__main__': print(json.dumps(run_acceptance(),sort_keys=True))
