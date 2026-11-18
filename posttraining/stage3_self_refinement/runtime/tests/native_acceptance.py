"""CPU-only native parity, controls kept outside the production release path."""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from contract import require, sha, read_json, write_new, validate_tokens, digest
from evidence_gates import code_pins, ref, canonical_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old-v1', type=Path, required=True)
    ap.add_argument('--old-v2', type=Path, required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--pins', type=Path, required=True)
    ap.add_argument('--admission-selected', type=Path)
    ap.add_argument('--admission-root', type=Path)
    ap.add_argument('--admission-build-sha256')
    ap.add_argument('--admission-upstream-map', type=Path)
    ap.add_argument('--tokenizer-source', type=Path)
    ap.add_argument('--bridge-fixture', action='store_true')
    ap.add_argument('--manifest')
    ap.add_argument('--manifest-sha256')
    a = ap.parse_args()
    a.out=a.out.resolve();a.pins=a.pins.resolve()
    require(a.out.is_relative_to(Path(__file__).resolve().parents[1]),'native output must be runtime-only')
    require(sum(bool(x) for x in (a.manifest,a.admission_selected,a.bridge_fixture))<=1,'native modes are exclusive')
    pins_sha=code_pins(a.pins)
    a.out.mkdir(parents=True, exist_ok=False)
    sys.dontwritebytecode=True
    before = os.environ.get('CUDA_VISIBLE_DEVICES')
    import pretokenized_v3 as prod
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == before, 'production changed CUDA environment')
    import torch
    from datasets import Dataset, Features, Sequence, Value
    from swift import get_processor, get_template
    from swift.template import TEMPLATE_MAPPING, register_template
    from swift.pipelines.utils import get_cached_dataset
    from swift.pipelines.train.sft import SwiftSft
    from swift.dataset import LazyLLMDataset
    from swift.sequence_parallel.sequence_parallel import SequenceParallel
    # Fail before any attempted model construction, weight loading, optimizer, or socket call.
    from transformers import PreTrainedModel
    def forbidden(*args, **kwargs):
        raise RuntimeError('standalone CPU acceptance forbids model/optimizer/network')
    PreTrainedModel.__init__ = forbidden
    PreTrainedModel.from_pretrained = forbidden
    torch.load = forbidden
    torch.optim.Optimizer.__init__ = forbidden
    socket.socket.connect = forbidden
    socket.socket.connect_ex = forbidden
    torch.set_num_threads(2)
    index = {r['uid']: r for r in map(json.loads, (a.old_v1 / 'control/INDEX.jsonl').read_text().splitlines())}
    class Controls(prod.TextOnlyQwen):
        def encode(self, inputs, return_template_inputs=False, return_length=False):
            uid, mode = inputs['uid'], inputs['control_mode']
            path = a.old_v1 / 'control/tok' / (uid + '.json')
            require(sha(path) == index[uid]['tokens_sha256'], 'control source changed')
            r = read_json(path)
            eligible = r['loss_mask']
            if mode == 'eligible_control':
                mask = eligible
            elif mode == 'pending_zero':
                mask = [0] * len(eligible)
            elif mode == 'alternating_runs_control':
                mask, run, prev = [], -1, 0
                for e in eligible:
                    if e and not prev:
                        run += 1
                    mask.append(int(bool(e) and run % 2 == 0))
                    prev = e
            else:
                raise ValueError('unknown engineering control')
            p = dict(input_ids=r['input_ids'], labels=[x if s else -100 for x,s in zip(r['input_ids'], mask)],
                     eligible_mask=eligible, selected_mask=mask)
            validate_tokens(p, [self.tokenizer.convert_tokens_to_ids(t) for t in ('<think>', '</think>')])
            require(inputs['input_ids'] == p['input_ids'] and inputs['labels'] == p['labels'], 'cache changed')
            return dict(input_ids=p['input_ids'], labels=p['labels'], length=len(p['input_ids']))
    meta = copy.deepcopy(TEMPLATE_MAPPING['qwen3_8'])
    meta.template_type, meta.template_cls = 'retrace_s3_ENGINEERING_ONLY', Controls
    register_template(meta, exist_ok=True)
    processor = get_processor(a.model)
    template = get_template(processor, template_type=meta.template_type, padding_free=False,
                            sequence_parallel_size=4, max_length=131072, truncation_strategy='raise')
    template.set_mode('train')
    class LocalRank(SequenceParallel):
        @property
        def sp_rank(self):
            return self.test_rank
    sp = LocalRank()
    sp.world_size = sp.sp_world_size = 4
    sp.rp_world_size = 1
    sp.tokenizer = template.tokenizer
    cases = []
    def check_batch(encs, label):
        for e in encs:
            e['length'] = len(e['input_ids'])
        for padding_free in (False, True):
            template.padding_free = padding_free
            batch = template.data_collator(copy.deepcopy(encs))
            ids, labels = batch['input_ids'], batch['labels']
            if padding_free:
                expected_ids = sum([e['input_ids'] for e in encs], [])
                expected_labels = sum([e['labels'] for e in encs], [])
                positions = sum([list(range(len(e['input_ids']))) for e in encs], [])
                require(ids[0].tolist() == expected_ids and labels[0].tolist() == expected_labels, 'concatenation parity')
                require(batch['text_position_ids'][0].tolist() == positions, 'sequence boundary positions')
                require(all(axis[0].tolist() == positions for axis in batch['position_ids']), 'native Qwen positions')
                # Actual installed SP pad / roll / split, with only rank identity injected;
                # no distributed initialization, collectives, model or attention execution.
                parts = []
                for rank in range(4):
                    sp.test_rank = rank
                    part = sp.pad_and_split_inputs(ids, None, labels, batch['position_ids'], None, None,
                                                    real_position_ids=batch['text_position_ids'])
                    parts.append(part)
                padded_ids = expected_ids + [template.tokenizer.pad_token_id] * ((-len(expected_ids)) % 4)
                padded_labels = expected_labels + [-100] * ((-len(expected_labels)) % 4)
                require(torch.cat([p[0] for p in parts], dim=-1)[0].tolist() == padded_ids, 'SP4 IDs')
                shifted = padded_labels[1:] + padded_labels[:1]
                require(torch.cat([p[2] for p in parts], dim=-1)[0].tolist() == shifted, 'SP4 causal labels')
                boundaries, offset = [], 0
                for e in encs:
                    boundaries.append(offset)
                    require(e['labels'][0] == -100, 'boundary first label')
                    offset += len(e['input_ids'])
                for start in boundaries[1:]:
                    require(shifted[start-1] == -100, 'cross-sequence causal leak')
                require(sum(x != -100 for x in shifted) == sum(x != -100 for x in expected_labels), 'SP loss count')
            else:
                for j, e in enumerate(encs):
                    n = len(e['input_ids'])
                    require(ids[j, :n].tolist() == e['input_ids'] and labels[j, :n].tolist() == e['labels'], 'padded parity')
                    require(all(x == -100 for x in labels[j, n:].tolist()), 'padding leak')
            require(template.model is None and template.dummy_model is None, 'model constructed')
        cases.append(dict(case=label, lengths=[len(e['input_ids']) for e in encs], exact=True))
    modes = ('eligible_control', 'pending_zero', 'alternating_runs_control')
    counters = {}
    for mode in modes:
        args = SimpleNamespace(cached_dataset=[str(a.old_v2 / 'cached_controls' / mode)], cached_val_dataset=[],
                               data_seed=42, max_length=131072, truncation_strategy='raise', dataset_shuffle=False,
                               streaming=False, strict=True, packing=False)
        train, val = get_cached_dataset(args)
        native = SwiftSft._post_process_datasets(SimpleNamespace(args=args, template=template), [train[0], None])[0]
        require(isinstance(native, LazyLLMDataset) and len(native) == len(index), 'native reader not exercised')
        previous, counters[mode] = None, 0
        for n in range(len(native)):
            enc = native[n]
            counters[mode] += sum(x != -100 for x in enc['labels'])
            check_batch([enc], mode + ':' + str(n))
            if previous is not None:
                check_batch([previous, enc], mode + ':pair:' + str(n))
            previous = enc
    # No-claim synthetic selector subset exercises all remainders mod 4 and boundaries.
    for n in range(5, 13):
        e = dict(input_ids=list(range(100, 100+n)), labels=[-100] + [100+i if i % 2 else -100 for i in range(1,n)])
        z = dict(input_ids=[201,202,203], labels=[-100]*3)
        check_batch([e,z], 'NO_CLAIM_SYNTHETIC:' + str(n))
    payloads={};binding={};data_index=[];selected_cases=[];mode='controls';release=None
    think=[template.tokenizer.convert_tokens_to_ids(t) for t in ('<think>','</think>')]
    if a.admission_selected:
        from admission_candidate import validate_candidate
        payloads,binding=validate_candidate(a.admission_root,a.admission_selected,a.admission_build_sha256,
                                           a.tokenizer_source,a.admission_upstream_map)
        data_index=binding.pop('canonical_index');mode='candidate'
    elif a.manifest:
        from contract import Release
        release=Release(a.manifest,a.manifest_sha256)
        payloads={u:release.payload(u) for u in release.rows}
        data_index=canonical_index(release);mode='production'
        os.environ.update(S3_MANIFEST=str(release.path),S3_MANIFEST_SHA256=a.manifest_sha256)
    elif a.bridge_fixture:
        # Exercise both classes on real control bytes, with an in-memory test double
        # only. Never produce a scientific release or actual-native admission report.
        for uid in [min(index,key=lambda u:index[u]['n_tokens']),max(index,key=lambda u:index[u]['n_tokens'])]:
            original=read_json(a.old_v1/'control/tok'/f'{uid}.json');ids=original['input_ids'];eligible=original['loss_mask']
            mask=[int(bool(x) and i%2==0) for i,x in enumerate(eligible)]
            payloads[uid]=dict(uid=uid,source_key=index[uid]['source_key'],input_ids=ids,eligible_mask=eligible,
                               selected_mask=mask,labels=[x if m else -100 for x,m in zip(ids,mask)])
        data_index=[dict(uid=u,source_key=p['source_key'],payload_sha256=digest(p),full_tokens=len(p['input_ids']),
                         selected_tokens=sum(p['selected_mask'])) for u,p in payloads.items()];mode='bridge_fixture'
    class CandidateTemplate(prod.TextOnlyQwen):
        def encode(self,inputs,return_template_inputs=False,return_length=False):
            require(inputs['mode'] in ('CANDIDATE_SELECTED','ZERO_CONTROL','PARTIAL_CONTROL'),'candidate cache mode')
            p=copy.deepcopy(payloads[inputs['uid']]);mask=p['selected_mask']
            if inputs['mode']=='ZERO_CONTROL':mask=[0]*len(mask)
            elif inputs['mode']=='PARTIAL_CONTROL':mask=[int(bool(x) and i%2==0) for i,x in enumerate(mask)]
            p['selected_mask']=mask;p['labels']=[x if m else -100 for x,m in zip(p['input_ids'],mask)]
            validate_tokens(p,think)
            expected=dict(uid=p['uid'],mode=inputs['mode'],input_ids=p['input_ids'],labels=p['labels'],lengths=[len(mask)])
            require(inputs==expected,'candidate frozen payload mismatch')
            return dict(input_ids=p['input_ids'],labels=p['labels'],length=len(mask))
    if payloads:
        cm=copy.deepcopy(TEMPLATE_MAPPING['qwen3_8']);cm.template_type='retrace_s3_candidate_CPU_ONLY';cm.template_cls=CandidateTemplate
        register_template(cm,exist_ok=True)
        candidate_template=get_template(processor,template_type=cm.template_type,sequence_parallel_size=4)
        candidate_template.set_mode('train')
        production=get_template(processor,template_type=prod.TEMPLATE,sequence_parallel_size=4);production.set_mode('train')
        if a.bridge_fixture:
            class FixtureRelease:
                manifest={'think_token_ids':think}
                def payload(self,u):return copy.deepcopy(payloads[u])
                def record(self,u):
                    p=self.payload(u);return dict(uid=u,mode='FINAL_SELECTED',input_ids=p['input_ids'],labels=p['labels'],lengths=[len(p['input_ids'])])
            production._release=FixtureRelease()
        features=Features({'uid':Value('string'),'mode':Value('string'),'input_ids':Sequence(Value('int64')),
                           'labels':Sequence(Value('int64')),'lengths':Sequence(Value('int64'))})
        index_hash=binding.get('selected_index_sha256',digest(data_index))
        for variant,cache_mode in [('selected','CANDIDATE_SELECTED'),('zero_control','ZERO_CONTROL'),('partial_control','PARTIAL_CONTROL')]:
            records=[]
            for uid,p in payloads.items():
                mask=p['selected_mask']
                if variant=='zero_control':mask=[0]*len(mask)
                elif variant=='partial_control':mask=[int(bool(x) and i%2==0) for i,x in enumerate(mask)]
                records.append(dict(uid=uid,mode=cache_mode,input_ids=p['input_ids'],labels=[x if m else -100 for x,m in zip(p['input_ids'],mask)],lengths=[len(mask)]))
            template=candidate_template
            if variant=='selected' and mode in ('production','bridge_fixture'):
                template=production
                for r in records:r['mode']='FINAL_SELECTED'
            cache=a.out/(variant+'-cache');Dataset.from_list(records,features=features).save_to_disk(str(cache))
            args.cached_dataset=[str(cache)];train,_=get_cached_dataset(args)
            native=SwiftSft._post_process_datasets(SimpleNamespace(args=args,template=template),[train[0],None])[0]
            require(isinstance(native,LazyLLMDataset) and len(native)==len(payloads),'candidate/production native reader coverage')
            previous=None
            for n,uid in enumerate(payloads):
                enc=native[n];require(enc['input_ids']==records[n]['input_ids'] and enc['labels']==records[n]['labels'],'native row parity')
                check_batch([enc],mode+':'+variant+':'+uid)
                selected_cases.append(dict(uid=uid,mode=variant,status='PASS',index_sha256=index_hash,
                    full_tokens=len(enc['input_ids']),selected_tokens=sum(x!=-100 for x in enc['labels']),
                    input_ids_sha256=digest(enc['input_ids']),labels_sha256=digest(enc['labels'])))
                if variant=='selected' and previous is not None:
                    check_batch([previous,enc],mode+':padded_pair:'+uid)
                    selected_cases.append(dict(uids=list(payloads)[n-1:n+1],mode='padded_pair',status='PASS',index_sha256=index_hash))
                previous=enc
        if a.bridge_fixture:
            # The candidate selected encode also gets a genuine native cache path.
            template=candidate_template
            records=[dict(uid=u,mode='CANDIDATE_SELECTED',input_ids=p['input_ids'],labels=p['labels'],lengths=[len(p['input_ids'])]) for u,p in payloads.items()]
            cache=a.out/'candidate-selected-fixture-cache';Dataset.from_list(records,features=features).save_to_disk(str(cache))
            args.cached_dataset=[str(cache)];train,_=get_cached_dataset(args)
            native=SwiftSft._post_process_datasets(SimpleNamespace(args=args,template=template),[train[0],None])[0]
            for n in range(len(native)):check_batch([native[n]],'candidate-selected-fixture:'+str(n))
    require(not torch.cuda.is_initialized(),'CUDA initialized')
    require(code_pins(a.pins)==pins_sha,'source pins changed during native run')
    if a.admission_selected:require(sha(a.admission_selected)==binding['selected_index_sha256'],'selected index changed during native run')
    write_new(a.out/'cases.json',cases)
    case_path=a.out/'NATIVE_CASES.jsonl'
    with case_path.open('x') as f:
        for c in selected_cases:f.write(json.dumps(c,sort_keys=True)+'\n')
    write_new(a.out/'DATA_INDEX.json',data_index)
    import inspect,platform,importlib.metadata
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    objects={'get_cached_dataset':get_cached_dataset,'sft_postprocessor':SwiftSft,'native_lazy_reader':LazyLLMDataset,
             'template':prod.Base,'text_rope_helper':Qwen3_5Model.get_rope_index,'adapter':prod.FinalSelectedTemplate,
             'acceptance_runner':main,'sequence_parallel':SequenceParallel}
    runtime=dict(python=sys.version,executable=str(Path(sys.executable).resolve()),executable_sha256=sha(Path(sys.executable).resolve()),
                 platform=platform.platform(),hostname=platform.node(),ms_swift_version=importlib.metadata.version('ms_swift'),
                 packages={n:importlib.metadata.version(n) for n in ('ms_swift','torch','transformers','datasets')})
    report=dict(schema='s3.native-evidence.v4',passed=True,mode=mode,
        data_origin='actual_native_run' if mode in ('candidate','production') else 'engineering_fixture',
        unique_real_rows=len(payloads) if mode in ('candidate','production') else 0,
        selected_ids=list(payloads),canonical_data_index=ref(a.out/'DATA_INDEX.json'),case_artifact=ref(case_path),
        control_case_artifact=ref(a.out/'cases.json'),runtime_identity=runtime,
        source_pins=ref(a.pins),source_pins_sha256=pins_sha,native_sources={n:ref(inspect.getfile(obj)) for n,obj in objects.items()},
        real_control_rows=len(index),controls=counters,cases=len(cases),cuda_import_preserved=True,cuda_initialized=False,
        model_loaded=False,no_model_constructed=True,no_optimizer=True,optimizer_steps=0,inference_calls=0,
        distributed_collectives_tested=False,sp4_native_pad_shift_split=True,training_ready=False,
        actual_manifest_sha256=a.manifest_sha256,actual_final_rows=len(payloads) if mode=='production' else 0,
        exact_full_tokens_and_masks=True,reasoning_loss_zero=True,causal_shift_exact=True,collator_exact=True,
        padding_exact=True,zero_and_partial_controls_passed=True,**binding)
    write_new(a.out/'report.json',report);write_new(a.out/'NATIVE_READER_REPORT.json',report)
    print(json.dumps(report))

if __name__ == '__main__':
    main()
