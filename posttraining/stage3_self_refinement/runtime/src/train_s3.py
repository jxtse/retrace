"""Owner-invoked gated official swift SFT launcher; default is read-only plan."""
import argparse
import os
from pathlib import Path
import subprocess
from evidence_gates import verify_native, verify_optimizer
from contract import Release, require, sha, read_json, write_new, PREDECESSOR_SHA, CONFIG_SHA, verify_ref


# --- paths and endpoints come from the environment; defaults are placeholders ---
import os
RETRACE_SHARE = os.environ.get("RETRACE_SHARE", "/path/to/retrace-share")
RETRACE_DATA_ROOT = os.environ.get("RETRACE_DATA_ROOT", "/path/to/data-root")
RETRACE_OUTPUTS = os.environ.get("RETRACE_OUTPUTS", "/path/to/outputs")
RETRACE_HOST = os.environ.get("RETRACE_HOST", "gpu-host")
ROOT = Path(__file__).resolve().parents[1]
BASE = Path(RETRACE_SHARE + '')
MODEL = BASE / 'models/Qwen/Qwen3.8-27B'
SWIFT = BASE / 'swift-env/venv-swift/bin/swift'
PREDECESSOR = BASE / 'stage2-train/a1v2_full_sp4_zero3/final-adapter'


def verify_pins(path):
    pins = read_json(path)
    for ref in pins['files']:
        verify_ref(ROOT, ref)
    require(pins['files'], 'empty source pins')
    return sha(path)


def plan(manifest, manifest_sha, stage, out, pins, acceptance):
    out = Path(out).resolve()
    require(out.is_relative_to(ROOT) and not out.exists(), 'new runtime output required')
    release = Release(manifest, manifest_sha)
    require(sha(PREDECESSOR / 'adapter_model.safetensors') == PREDECESSOR_SHA, 'predecessor changed')
    require(sha(PREDECESSOR / 'adapter_config.json') == CONFIG_SHA, 'predecessor config changed')
    cfg = read_json(PREDECESSOR / 'adapter_config.json')
    require(cfg['r'] == 8 and cfg['lora_alpha'] == 16, 'LoRA lineage mismatch')
    pins_sha = verify_pins(pins)
    verify_native(acceptance, release, pins)
    selected = list(release.rows)
    if stage == 'smoke':
        selected = sorted(selected, key=lambda u: (-len(release.payload(u)['input_ids']), u))[:8]
    # max_steps=2 intentionally repeats only if fewer than eight admitted rows exist.
    argv = [str(SWIFT), 'sft', '--model', str(MODEL), '--template', 'retrace_s3_final_selected_v3',
        '--external_plugins', str(ROOT/'src/pretokenized_v3.py'), str(ROOT/'src/training_trace.py'), '--callbacks', 's3_trace_v4',
        '--remove_unused_columns', 'false', '--adapters', str(PREDECESSOR),
        '--cached_dataset', str(out/'cache'), '--dataset_shuffle', 'false', '--split_dataset_ratio', '0',
        '--strict', 'true', '--packing', 'false', '--loss_scale', 'ignore_empty_think', '--add_non_thinking_prefix', 'true',
        '--max_length', '131072', '--truncation_strategy', 'raise', '--attn_impl', 'flash_attn', '--padding_free', 'true',
        '--learning_rate', '2e-5', '--lr_scheduler_type', 'constant', '--warmup_ratio', '0', '--warmup_steps', '0',
        '--weight_decay', '0', '--max_grad_norm', '1.0', '--num_train_epochs', '1', '--per_device_train_batch_size', '1',
        '--gradient_accumulation_steps', '4', '--gradient_checkpointing', 'true', '--use_liger_kernel', 'false',
        '--sequence_parallel_size', '4', '--optim', 'adamw_torch', '--adam_beta1', '0.9', '--adam_beta2', '0.999',
        '--adam_epsilon', '1e-8', '--bf16', 'true', '--torch_dtype', 'bfloat16', '--seed', '42', '--data_seed', '42',
        '--logging_steps', '1', '--save_strategy', 'steps', '--save_steps', '1' if stage=='smoke' else '10',
        '--save_total_limit', '3', '--save_only_model', 'true', '--report_to', 'none', '--dataloader_num_workers', '0',
        '--add_version', 'false', '--deepspeed', 'zero3', '--output_dir', str(out/'checkpoints')]
    if stage == 'smoke':
        argv += ['--max_steps', '2']
    env = dict(CELOSS_PARALLEL_SIZE='2048', CUDA_VISIBLE_DEVICES='0,1,2,3', NPROC_PER_NODE='4', MASTER_PORT='29579',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TMPDIR=str(out/'tmp'), MODELSCOPE_CACHE=str(out/'msc'), HF_HOME=str(out/'hf'),
        HF_DATASETS_CACHE=str(out/'datasets-cache'), PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=str(ROOT/'src'), S3_MANIFEST=str(release.path), S3_MANIFEST_SHA256=manifest_sha,
        S3_EXPOSURE_DIR=str(out/'exposure'), S3_EVIDENCE_DIR=str(out/'optimizer-evidence'))
    return release, dict(stage=stage, argv=argv, env=env, selected_uids=selected,
        manifest_sha256=manifest_sha, predecessor_sha256=PREDECESSOR_SHA, pins_sha256=pins_sha,
        native_acceptance_sha256=sha(acceptance), source_pins_path=str(Path(pins).resolve()), fresh_optimizer=True, resume_from_checkpoint=None,
        planned_full_tokens=sum(len(release.payload(u)['input_ids']) for u in selected),
        planned_selected_tokens=sum(sum(release.payload(u)['selected_mask']) for u in selected))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest', required=True); p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--stage', choices=['smoke','full'], required=True); p.add_argument('--out', required=True)
    p.add_argument('--pins', default=str(ROOT/'SOURCE_PINS.fix-v4-final2.json')); p.add_argument('--native-acceptance', required=True)
    p.add_argument('--optimizer-gate'); p.add_argument('--execute', action='store_true')
    a=p.parse_args()
    release, rec=plan(a.manifest,a.manifest_sha256,a.stage,a.out,a.pins,a.native_acceptance)
    if a.stage == 'full':
        require(a.optimizer_gate, 'full requires independent smoke/save/reload gate')
        verify_optimizer(a.optimizer_gate, a.pins, a.manifest_sha256, 'smoke')
        rec['optimizer_gate_sha256']=sha(a.optimizer_gate)
    if not a.execute:
        import json
        print(json.dumps(rec,indent=2)); return
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    for name in ('tmp','msc','hf','datasets-cache','exposure','optimizer-evidence'):
        (out/name).mkdir()
    write_new(out/'launch.json',rec)
    os.environ.update(rec['env'])
    from datasets import Dataset, Features, Sequence, Value
    features=Features({'uid':Value('string'),'mode':Value('string'),'input_ids':Sequence(Value('int64')),
                       'labels':Sequence(Value('int64')),'lengths':Sequence(Value('int64'))})
    Dataset.from_list([release.record(u) for u in rec['selected_uids']],features=features).save_to_disk(str(out/'cache'))
    with (out/'train.log').open('xb') as log:
        result=subprocess.run(['timeout','36000']+rec['argv'],env=dict(os.environ,**rec['env']),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    write_new(out/'exit.json',{'returncode':result.returncode})
    raise SystemExit(result.returncode)

if __name__=='__main__': main()
